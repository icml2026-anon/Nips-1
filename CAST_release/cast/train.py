import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import time

from cast.config import CASTConfig
from cast.models.cast_model import CASTModel
from cast.models.causal_discovery import train_causal_discovery
from cast.models.hyperbolic import train_hyperbolic_embedding, RiemannianSGD
from cast.utils.metrics import compute_binary_metrics, print_metrics

from cast.data.dataset import SparseSequenceDataset, default_transform


class CASTTrainer:

    def __init__(self, config, model=None):
        self.config = config
        self.device = torch.device(config.device)
        if model is None:
            self.model = CASTModel(config).to(self.device)
        else:
            self.model = model.to(self.device)

    def _build_dataloader(self, X, y, masks=None, shuffle=True,
                          apply_transform=True):
        transform_fn = default_transform if apply_transform else None
        dataset = SparseSequenceDataset(
            X, y, masks=masks, transform_fn=transform_fn
        )
        return DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=shuffle, drop_last=False, num_workers=0
        )

    def _get_raw_ts_matrix(self, X):
        if X.ndim == 3:
            if X.shape[-1] == 1:
                return X.squeeze(-1)
            return X.reshape(X.shape[0], -1)
        return X

    def phase1_pretrain(self, X_train):
        print("=" * 60)
        print("Phase 1: Pre-training Causal Discovery + Hyperbolic Embedding")
        print("=" * 60)

        X_raw = self._get_raw_ts_matrix(X_train)

        print("\n[Stage 1] Training Temporal Causal Discovery...")
        t0 = time.time()
        A = train_causal_discovery(
            self.model.causal_discovery, X_raw, self.config, self.device
        )
        self.model.update_causal_structure(A)
        num_edges = (A.abs() > 0).sum().item()
        print(f"  Discovered {num_edges} causal edges in {time.time() - t0:.1f}s")
        print(f"  DAG constraint h(A) = {self.model.causal_discovery.dag_constraint(A).item():.6f}")
        if num_edges > 0:
            print(f"  Edge weight stats: mean={A[A.abs()>0].abs().mean():.4f}, max={A.abs().max():.4f}")
        else:
            print(f"  Edge weight stats: mean=nan, max={A.abs().max():.4f}")

        self.model.freeze_causal_discovery()
        print("  Causal discovery module frozen.")

        if self.config.ablate_no_hyperbolic:
            print("\n[Stage 2] ABLATION: Using Euclidean embeddings (skipping hyperbolic)")
            with torch.no_grad():
                T_c = self.model.T_causal
                d = self.config.embedding_dim
                if A.abs().sum() < 1e-8:
                    eucl_emb = torch.randn(T_c, d, device=self.device) * 0.1
                else:
                    U, S, V = torch.svd(A.cpu().float())
                    k = min(d, T_c)
                    raw = U[:, :k] * S[:k].sqrt().unsqueeze(0)
                    if k < d:
                        raw = torch.cat([raw, torch.zeros(T_c, d - k)], dim=1)
                    raw = raw / (raw.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                    eucl_emb = (raw * 0.3).to(self.device)
                pooled = self.model._pool_causal_embeddings(eucl_emb)
                self.model.poincare_embeddings.copy_(pooled)
            self.model.hyperbolic_embedding.embeddings.requires_grad = False
            norms = pooled.norm(dim=-1)
            print(f"  Euclidean embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
            print("  Hyperbolic embedding frozen (forward uses Euclidean buffer).")
        else:
            print("\n[Stage 2] Training Hyperbolic Causal Embedding...")
            t0 = time.time()
            self.model.hyperbolic_embedding = train_hyperbolic_embedding(
                self.model.hyperbolic_embedding, A, self.config, self.device
            )
            self.model.update_poincare_embeddings()
            print(f"  Embedding training completed in {time.time() - t0:.1f}s")

            e = self.model.hyperbolic_embedding.get_poincare_embeddings()
            norms = e.norm(dim=-1)
            print(f"  Poincare embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    def phase2_finetune(self, X_train, y_train, X_val=None, y_val=None,
                        masks_train=None, masks_val=None,
                        preprocessed=False):
        print("\n" + "=" * 60)
        print("Phase 2: End-to-End Joint Fine-tuning")
        print("=" * 60)

        train_loader = self._build_dataloader(
            X_train, y_train, masks=masks_train,
            shuffle=True, apply_transform=not preprocessed
        )

        hyp_embedding_param = self.model.hyperbolic_embedding.embeddings
        hyp_embedding_id = id(hyp_embedding_param)

        use_rsgd = hyp_embedding_param.requires_grad and not self.config.ablate_no_hyperbolic

        adam_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) != hyp_embedding_id
        ]
        if hyp_embedding_param.requires_grad and not use_rsgd:
            adam_params.append(hyp_embedding_param)

        adam_optimizer = torch.optim.AdamW(
            adam_params, lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        if use_rsgd:
            rsgd_optimizer = RiemannianSGD(
                [hyp_embedding_param],
                lr=self.config.rsgd_lr
            )
        else:
            rsgd_optimizer = None
            if self.config.ablate_no_causal:
                print("  [A1] RSGD disabled (embeddings frozen at zero)")
            elif self.config.ablate_no_hyperbolic:
                print("  [A2] RSGD disabled (Euclidean embeddings frozen in buffer)")

        total_epochs = self.config.finetune_epochs
        warmup_epochs = 5
        base_lr = self.config.learning_rate

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-6 / base_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(adam_optimizer, lr_lambda)

        best_val_f1 = 0.0
        best_val_auc = 0.0
        best_state = None
        patience = 0
        max_patience = 25

        for epoch in range(self.config.finetune_epochs):
            self.model.train()
            epoch_losses = {"total": 0, "task": 0, "hyperbolic": 0}
            n_batches = 0

            for batch in train_loader:
                x = batch["ts_data"].to(self.device)
                masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                adam_optimizer.zero_grad()
                if rsgd_optimizer is not None:
                    rsgd_optimizer.zero_grad()

                total_loss, logits, loss_dict = self.model.compute_total_loss(
                    x, labels, masks
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(adam_params, 1.0)
                adam_optimizer.step()
                if rsgd_optimizer is not None:
                    rsgd_optimizer.step()

                for k in epoch_losses:
                    if k in loss_dict:
                        epoch_losses[k] += loss_dict[k].item()
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            scheduler.step()

            cur_lr = adam_optimizer.param_groups[0]["lr"]

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(
                    X_val, y_val, masks=masks_val,
                    apply_transform=not preprocessed
                )
                val_f1 = val_metrics.get("f1", 0)
                val_auc = val_metrics.get("auc", 0)
                if val_f1 != val_f1:
                    val_f1 = 0
                if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_auc > best_val_auc):
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

            do_print = (epoch + 1) % max(1, self.config.finetune_epochs // 20) == 0 or epoch == 0 or epoch == self.config.finetune_epochs - 1
            if do_print:
                msg = (
                    f"Epoch {epoch + 1}/{self.config.finetune_epochs} | "
                    f"Task: {epoch_losses['task']:.4f} | "
                    f"Hyp: {epoch_losses['hyperbolic']:.4f} | "
                    f"LR: {cur_lr:.1e}"
                )
                if X_val is not None:
                    msg += f" | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                print(msg)

            if patience >= max_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            print(f"\nRestored best model (Val F1: {best_val_f1:.4f}, Val AUC: {best_val_auc:.4f})")

    def train(self, X_train, y_train, X_val=None, y_val=None,
              X_train_pp=None, X_val_pp=None,
              masks_train=None, masks_val=None,
              skip_phase1=False):
        if isinstance(y_train, np.ndarray):
            pos_ratio = y_train.mean()
        else:
            pos_ratio = y_train.float().mean().item()
        if pos_ratio > 0 and pos_ratio < 1:
            pw = (1.0 - pos_ratio) / pos_ratio
            self.model._pos_weight = pw
            print(f"Class balance: pos_ratio={pos_ratio:.4f}, pos_weight={pw:.2f}")

        if skip_phase1 or self.config.ablate_no_causal:
            print("\n" + "=" * 60)
            print("ABLATION A1: Skipping Phase 1 (zero embeddings, no causal structure)")
            print("=" * 60)
            self.model.freeze_causal_discovery()
            with torch.no_grad():
                self.model.poincare_embeddings.zero_()
                emb = self.model.hyperbolic_embedding.embeddings.data
                emb.zero_()
                emb[:, 0] = 1.0
            self.model.hyperbolic_embedding.embeddings.requires_grad = False
            print("  Hyperbolic embeddings frozen at zero.")
        else:
            self.phase1_pretrain(X_train)

        if X_train_pp is not None:
            self.phase2_finetune(X_train_pp, y_train,
                                X_val_pp if X_val_pp is not None else X_val,
                                y_val,
                                masks_train=masks_train,
                                masks_val=masks_val,
                                preprocessed=True)
        else:
            self.phase2_finetune(X_train, y_train, X_val, y_val,
                                preprocessed=False)

    @torch.no_grad()
    def evaluate(self, X, y, masks=None, apply_transform=True):
        self.model.eval()
        loader = self._build_dataloader(
            X, y, masks=masks, shuffle=False,
            apply_transform=apply_transform
        )
        all_probs = []
        all_labels = []

        for batch in loader:
            x = batch["ts_data"].to(self.device)
            batch_masks = batch["mask"].to(self.device)
            labels = batch["label"]

            logits = self.model(x, batch_masks)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        return compute_binary_metrics(all_labels, all_probs)

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()

        dataset = SparseSequenceDataset(X_tensor, torch.zeros(len(X_tensor)), transform_fn=default_transform)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        all_probs = []

        for batch in loader:
            x = batch["ts_data"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(x, masks)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs)

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")
