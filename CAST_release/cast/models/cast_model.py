import torch
import torch.nn as nn
import torch.nn.functional as F

from cast.models.causal_discovery import CausalDiscovery
from cast.models.hyperbolic import HyperbolicEmbedding, hyperboloid_to_poincare
from cast.models.cast_kan import (
    CASTKANSequenceModel,
    SampleAdaptiveSparsityGate,
    CausalGate,
    CausalPhaseRotation
)


class CASTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        T = config.seq_len
        C_raw = getattr(config, 'raw_channels', 1)
        T_causal = T * C_raw
        d = config.embedding_dim

        self.T = T
        self.C_raw = C_raw
        self.T_causal = T_causal

        self.causal_discovery = CausalDiscovery(
            T=T_causal,
            hidden_dim=config.sem_hidden_dim,
            sigma_init=config.sigma_z
        )

        self.hyperbolic_embedding = HyperbolicEmbedding(
            T=T_causal,
            embedding_dim=d,
            khop=config.khop,
            lambda_g=config.lambda_g,
            damping=config.pagerank_damping
        )

        self.sparsity_gate = SampleAdaptiveSparsityGate(
            seq_len=T,
            embedding_dim=d,
            hidden_dim=config.sparsity_mlp_hidden
        )

        self.sequence_model = CASTKANSequenceModel(config)

        if config.ablate_no_sparsity_gate:
            for layer in self.sequence_model.kan_layers:
                if hasattr(layer, 'causal_gate'):
                    layer.causal_gate.disabled = True
                if hasattr(layer, 'phase_rot'):
                    layer.phase_rot.disabled = True

        if config.ablate_no_causal_attn:
            for layer in self.sequence_model.kan_layers:
                if hasattr(layer, 'causal_gate'):
                    layer.causal_gate.disabled = True
                if hasattr(layer, 'phase_rot'):
                    layer.phase_rot.disabled = True

        self.register_buffer("adjacency", torch.zeros(T_causal, T_causal))
        self.register_buffer("poincare_embeddings", torch.zeros(T, d))
        self._dag_frozen = False
        self._focal_gamma = config.focal_gamma
        self._pos_weight = 1.0

    def update_causal_structure(self, A):
        self.adjacency.copy_(A.detach())

    def _pool_causal_embeddings(self, e):
        if self.C_raw > 1:
            return e.reshape(self.T, self.C_raw, -1).mean(dim=1)
        return e

    def update_poincare_embeddings(self):
        with torch.no_grad():
            e = self.hyperbolic_embedding.get_poincare_embeddings()
            self.poincare_embeddings.copy_(self._pool_causal_embeddings(e))

    def freeze_causal_discovery(self):
        for p in self.causal_discovery.parameters():
            p.requires_grad = False
        self._dag_frozen = True

    def forward(self, x, masks=None):
        if masks is None:
            if x.dim() == 2:
                masks = (x > 0).float()
            elif x.dim() == 3:
                masks = (x.abs().sum(dim=-1) > 0).float()

        if self.config.ablate_no_hyperbolic:
            phi = self.poincare_embeddings
        else:
            phi_full = self.hyperbolic_embedding.get_poincare_embeddings()
            phi = self._pool_causal_embeddings(phi_full)

        if self.config.ablate_no_sparsity_gate:
            phi_adapted = phi
        else:
            phi_adapted = self.sparsity_gate(masks, phi)

        logits = self.sequence_model(x, phi_adapted)
        return logits

    def compute_task_loss(self, logits, labels):
        class_weight = self._pos_weight * labels + 1.0 * (1 - labels)

        if self.config.ablate_no_focal:
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            return (class_weight * bce).mean()

        ls = labels * 0.97 + 0.015
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        focal_weight = (1 - p_t) ** self._focal_gamma
        bce = F.binary_cross_entropy_with_logits(logits, ls, reduction="none")
        return (class_weight * focal_weight * bce).mean()

    def compute_hyperbolic_loss(self):
        return self.hyperbolic_embedding(self.adjacency)

    def compute_total_loss(self, x, labels, masks=None):
        if masks is None:
            if x.dim() == 2:
                masks = (x > 0).float()
            elif x.dim() == 3:
                masks = (x.abs().sum(dim=-1) > 0).float()

        logits = self.forward(x, masks)
        task_loss = self.compute_task_loss(logits, labels)

        if self.config.ablate_no_hyperbolic:
            hyp_loss = torch.tensor(0.0, device=task_loss.device)
        else:
            hyp_loss = self.compute_hyperbolic_loss()

        total = task_loss + self.config.lambda_hyp * hyp_loss

        loss_dict = {
            "total": total,
            "task": task_loss,
            "hyperbolic": hyp_loss,
        }
        return total, logits, loss_dict
