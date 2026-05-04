
import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from einops import rearrange, reduce

from cast.utils.metrics import compute_binary_metrics


def _extract_common_features(X):
    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    B, T, C = X.shape
    parts = []

    for c in range(C):
        x = X[:, :, c]
        nz_mask = (x > 0).astype(np.float32)
        nz_count = nz_mask.sum(axis=1, keepdims=True).clip(min=1)

        parts.append(x.mean(axis=1, keepdims=True))
        parts.append(x.std(axis=1, keepdims=True))
        parts.append(x.max(axis=1, keepdims=True))
        parts.append(x.sum(axis=1, keepdims=True))
        parts.append(nz_mask.mean(axis=1, keepdims=True))
        parts.append(nz_mask.sum(axis=1, keepdims=True))
        parts.append((x * nz_mask).sum(axis=1, keepdims=True) / nz_count)

        last_nz = np.full((B, 1), -1.0)
        for i in range(B):
            idx = np.where(x[i] > 0)[0]
            if len(idx) > 0:
                last_nz[i] = T - 1 - idx[-1]
        parts.append(last_nz)

        t_axis = np.arange(T, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = ((t_axis - t_mean) ** 2).sum()
        slope = np.zeros((B, 1))
        if t_var > 0:
            for i in range(B):
                slope[i] = ((t_axis - t_mean) * (x[i] - x[i].mean())).sum() / t_var
        parts.append(slope)

        parts.append(x[:, -1:])

    return np.concatenate(parts, axis=1)


def _croston_params(series, alpha=0.1):
    nz_idx = np.where(series > 0)[0]
    T = len(series)

    if len(nz_idx) == 0:
        return 0.0, 0.0, float(T)
    if len(nz_idx) == 1:
        return series[nz_idx[0]] / T, series[nz_idx[0]], float(T)

    z = series[nz_idx[0]]
    p = float(nz_idx[0] + 1)
    prev = nz_idx[0]

    for idx in nz_idx[1:]:
        interval = float(idx - prev)
        z = alpha * series[idx] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        prev = idx

    return z / max(p, 1e-8), z, p


def _sba_params(series, alpha=0.1):
    rate, z, p = _croston_params(series, alpha)
    return rate * (1 - alpha / 2), z, p


def _zip_params(series):
    x_mean = series.mean()
    x_var = series.var()

    if x_mean < 1e-8:
        return 1.0, 0.0

    lam = x_mean + x_var / max(x_mean, 1e-8) - 1
    lam = max(lam, 1e-8)
    pi = np.clip(1 - x_mean / lam, 0.0, 1.0)
    return pi, lam


def _hurdle_params(series):
    nz = series[series > 0]
    p_nz = len(nz) / max(len(series), 1)

    if len(nz) == 0:
        return p_nz, 0.0, 0.0

    cond_mean = nz.mean()
    cond_cv = nz.std() / max(cond_mean, 1e-8) if len(nz) > 1 else 0.0
    return p_nz, cond_mean, cond_cv


class _BaseBaseline:

    name = "base"

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"
        )

    def _model_features(self, X):
        raise NotImplementedError

    def _build_features(self, X):
        common = _extract_common_features(X)
        model = self._model_features(X)
        feat = np.concatenate([common, model], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X_train, y_train):
        feat = self.scaler.fit_transform(self._build_features(X_train))
        self.clf.fit(feat, y_train)

    def predict_proba(self, X):
        feat = self.scaler.transform(self._build_features(X))
        return self.clf.predict_proba(feat)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        return compute_binary_metrics(y, proba)


class CrostonBaseline(_BaseBaseline):

    name = "Croston"

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def _model_features(self, X):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        B, T, C = X.shape
        parts = []
        for c in range(C):
            rates = np.zeros((B, 1))
            demands = np.zeros((B, 1))
            intervals = np.zeros((B, 1))
            for i in range(B):
                r, d, p = _croston_params(X[i, :, c], self.alpha)
                rates[i], demands[i], intervals[i] = r, d, p
            parts.extend([rates, demands, intervals])
        return np.concatenate(parts, axis=1)


class SBABaseline(_BaseBaseline):

    name = "SBA"

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def _model_features(self, X):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        B, T, C = X.shape
        parts = []
        for c in range(C):
            rates = np.zeros((B, 1))
            demands = np.zeros((B, 1))
            intervals = np.zeros((B, 1))
            for i in range(B):
                r, d, p = _sba_params(X[i, :, c], self.alpha)
                rates[i], demands[i], intervals[i] = r, d, p
            parts.extend([rates, demands, intervals])
        return np.concatenate(parts, axis=1)


class ZIPBaseline(_BaseBaseline):

    name = "ZIP"

    def _model_features(self, X):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        B, T, C = X.shape
        parts = []
        for c in range(C):
            pis = np.zeros((B, 1))
            lams = np.zeros((B, 1))
            for i in range(B):
                pi, lam = _zip_params(X[i, :, c])
                pis[i], lams[i] = pi, lam
            parts.extend([pis, lams])
        return np.concatenate(parts, axis=1)


class HurdleBaseline(_BaseBaseline):

    name = "Hurdle"

    def _model_features(self, X):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        B, T, C = X.shape
        parts = []
        for c in range(C):
            p_nz = np.zeros((B, 1))
            c_mean = np.zeros((B, 1))
            c_cv = np.zeros((B, 1))
            for i in range(B):
                pn, cm, cc = _hurdle_params(X[i, :, c])
                p_nz[i], c_mean[i], c_cv[i] = pn, cm, cc
            parts.extend([p_nz, c_mean, c_cv])
        return np.concatenate(parts, axis=1)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_per_sample(X):
    X = X.astype(np.float32).copy()
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    mu = X.mean(axis=1, keepdims=True)
    std = np.clip(X.std(axis=1, keepdims=True), 1e-6, None)
    return ((X - mu) / std).astype(np.float32)


def _make_loader(X, y, batch_size, shuffle=False, device=None):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    if device is not None:
        X_t = X_t.to(device)
        y_t = y_t.to(device)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


class _NNBaseline:

    name = "nn_base"

    def __init__(self, epochs=500, batch_size=64, lr=1e-3, patience=30,
                 weight_decay=0.0, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = device or _get_device()
        self.model = None

    def _build_model(self, seq_len, in_channels):
        raise NotImplementedError

    def _preprocess(self, X):
        X = _normalize_per_sample(X)
        return X.transpose(0, 2, 1)

    def _pos_weight(self, y):
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        return torch.tensor([max(n_neg / max(n_pos, 1), 1.0)],
                            dtype=torch.float32, device=self.device)

    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = self.model(xb)
            loss = criterion(logits.squeeze(-1), yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            logits = self.model(xb)
            loss = criterion(logits.squeeze(-1), yb)
            if torch.isnan(loss):
                return float("inf")
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_tr = self._preprocess(X_train)
        _, C, T = X_tr.shape
        self.model = self._build_model(seq_len=T, in_channels=C).to(self.device)

        tr_loader = _make_loader(X_tr, y_train, self.batch_size,
                                 shuffle=True, device=self.device)
        val_loader = None
        if X_val is not None:
            X_vl = self._preprocess(X_val)
            val_loader = _make_loader(X_vl, y_val, self.batch_size,
                                      device=self.device)

        pw = self._pos_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self._train_epoch(tr_loader, optimizer, criterion)

            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        X_pp = self._preprocess(X)
        loader = _make_loader(X_pp, np.zeros(len(X_pp)), self.batch_size,
                              device=self.device)
        probs = []
        for xb, _ in loader:
            logits = self.model(xb)
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        out = np.concatenate(probs)
        return np.nan_to_num(out, nan=0.5)

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        return compute_binary_metrics(y, proba)


class _SparseCNNLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, density=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2)
        self.register_buffer("mask", torch.ones_like(self.conv.weight))
        self.density = density
        self._init_sparse()

    def _init_sparse(self):
        with torch.no_grad():
            w = self.conv.weight.abs().flatten()
            k = max(int(w.numel() * self.density), 1)
            thr = torch.topk(w, k).values[-1]
            self.mask.copy_((self.conv.weight.abs() >= thr).float())
            self.conv.weight.mul_(self.mask)

    def forward(self, x):
        self.conv.weight.data.mul_(self.mask)
        return self.conv(x)


class _SparseCNNModule(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, density):
        super().__init__()
        self.sparse_conv = _SparseCNNLayer(in_ch, out_ch, kernel_size, density)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.sparse_conv(x)))
        x = F.relu(self.bn2(self.conv1x1(x)))
        return x


class _DSNNet(nn.Module):

    def __init__(self, in_channels, seq_len, ch_size=47, kernel_size=39,
                 depth=4, density=0.2):
        super().__init__()
        self.depth = depth
        self.density = density

        modules = []
        c_in = in_channels
        for _ in range(min(depth - 1, 3)):
            modules.append(_SparseCNNModule(c_in, ch_size, kernel_size, density))
            c_in = ch_size
        self.sparse_modules = nn.ModuleList(modules)

        self.final_sparse = _SparseCNNLayer(ch_size, ch_size, kernel_size,
                                            density)
        self.final_bn = nn.BatchNorm1d(ch_size)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(ch_size * 2, 1)

    def forward(self, x):
        for mod in self.sparse_modules:
            x = mod(x)
        x = F.relu(self.final_bn(self.final_sparse(x)))
        x_avg = self.avg_pool(x).squeeze(-1)
        x_max = self.max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        return self.classifier(x)

    def get_sparse_layers(self):
        layers = []
        for mod in self.sparse_modules:
            layers.append(mod.sparse_conv)
        layers.append(self.final_sparse)
        return layers


def _dsn_prune_and_regrow(sparse_layers, iteration, total_iters, alpha=0.5):
    update_frac = alpha / 2 * (1 + math.cos(math.pi * iteration / max(total_iters, 1)))
    if update_frac < 1e-6:
        return

    for layer in sparse_layers:
        with torch.no_grad():
            mask = layer.mask
            w = layer.conv.weight * mask
            active_idx = mask.flatten().nonzero(as_tuple=True)[0]
            n_active = len(active_idx)
            n_update = max(int(n_active * update_frac), 1)
            if n_active <= n_update:
                continue

            active_vals = w.flatten()[active_idx].abs()
            _, prune_local = torch.topk(active_vals, n_update, largest=False)
            prune_idx = active_idx[prune_local]

            flat_mask = mask.flatten()
            flat_mask[prune_idx] = 0.0

            inactive_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(inactive_idx) > 0:
                n_regrow = min(n_update, len(inactive_idx))
                perm = torch.randperm(len(inactive_idx),
                                      device=mask.device)[:n_regrow]
                flat_mask[inactive_idx[perm]] = 1.0

            mask.copy_(flat_mask.view_as(mask))
            layer.conv.weight.data.mul_(mask)


class DSNBaseline(_NNBaseline):

    name = "DSN"

    def __init__(self, density=0.2, ch_size=47, kernel_size=39, depth=4,
                 epochs=500, batch_size=64, lr=1e-3, patience=30,
                 device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, device=device)
        self.density = density
        self.ch_size = ch_size
        self.kernel_size = kernel_size
        self.depth = depth
        self._global_step = 0
        self._total_steps = 0

    def _build_model(self, seq_len, in_channels):
        ks = min(self.kernel_size, seq_len // 2 * 2 + 1)
        if ks % 2 == 0:
            ks -= 1
        ks = max(ks, 3)
        return _DSNNet(in_channels, seq_len, ch_size=self.ch_size,
                       kernel_size=ks, depth=self.depth,
                       density=self.density)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_tr = self._preprocess(X_train)
        n_batches = max(len(X_tr) // self.batch_size, 1)
        self._total_steps = n_batches * self.epochs
        self._global_step = 0
        super().fit(X_train, y_train, X_val, y_val)

    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        sparse_layers = self.model.get_sparse_layers()
        total_loss = 0.0
        delta_t = 100

        for xb, yb in loader:
            optimizer.zero_grad()
            logits = self.model(xb)
            loss = criterion(logits.squeeze(-1), yb)
            loss.backward()

            for layer in sparse_layers:
                if layer.conv.weight.grad is not None:
                    layer.conv.weight.grad.mul_(layer.mask)

            optimizer.step()
            total_loss += loss.item() * len(yb)

            self._global_step += 1
            if self._global_step % delta_t == 0:
                _dsn_prune_and_regrow(sparse_layers, self._global_step,
                                      self._total_steps)

        return total_loss / len(loader.dataset)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class _MoEBlock(nn.Module):

    def __init__(self, dim, num_experts, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)
        self.norm = _RMSNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)

        logits = F.softmax(self.gate(x_flat), dim=-1)
        top_val, top_idx = logits.topk(1, dim=-1)

        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask_i = (top_idx.squeeze(-1) == i)
            if mask_i.any():
                out[mask_i] = self.experts[i](x_flat[mask_i])

        y = x_flat + out * top_val
        y = self.norm(y.view(B, N, D))

        importance = logits.sum(0)
        eps = 1e-10
        load_loss = importance.float().var() / (importance.float().mean() ** 2 + eps)

        return F.gelu(y), load_loss


class _InceptionModule1D(nn.Module):

    NF = 32

    def __init__(self, in_dim, ks=40):
        super().__init__()
        nf = self.NF
        ks_list = [max(ks // (2 ** i), 1) for i in range(3)]
        ks_list = [k if k % 2 != 0 else k - 1 for k in ks_list]
        ks_list = [max(k, 1) for k in ks_list]

        self.bottleneck = (nn.Conv1d(in_dim, nf, 1, bias=False)
                           if in_dim > 1 else nn.Identity())
        bn_out = nf if in_dim > 1 else in_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_out, nf, k, padding=k // 2, bias=False)
            for k in ks_list
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_dim, nf, 1, bias=False)
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.GELU()

    def forward(self, x):
        T = x.shape[2]
        x_bn = self.bottleneck(x)
        outs = [conv(x_bn) for conv in self.convs] + [self.maxpool_conv(x)]
        out = torch.cat(outs, dim=1)
        if T > 1:
            out = self.bn(out)
        return self.act(out)


class _SoftShapeLayer(nn.Module):

    def __init__(self, dim, moe, attn_head):
        super().__init__()
        self.norm1 = _RMSNorm(dim)
        self.norm2 = _RMSNorm(dim)
        self.attn_head = attn_head
        self.moe = moe
        self.inception = _InceptionModule1D(dim, ks=min(40, dim))
        incep_out_dim = _InceptionModule1D.NF * 4
        self.incep_proj = nn.Linear(incep_out_dim, dim)
        self.drop = nn.Dropout(0.15)

    def forward(self, x, remain_ratio=1.0, is_last=False):
        B, N, D = x.shape
        x_n = self.norm1(x)
        scores = self.attn_head(x_n)

        if remain_ratio < 1.0 and N > 2:
            left_k = max(math.ceil(remain_ratio * N), 1)
            _, top_idx = torch.topk(scores.squeeze(-1), left_k, dim=1)
            sorted_idx, _ = torch.sort(top_idx, dim=1)

            left_index = sorted_idx.unsqueeze(-1).expand(-1, -1, D)
            left_x = torch.gather(x_n * scores, 1, left_index)

            comp_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            comp_mask.scatter_(1, sorted_idx, False)
            all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            comp_idx = all_idx[comp_mask].view(B, N - left_k)
            comp_index = comp_idx.unsqueeze(-1).expand(-1, -1, D)
            non_topk = torch.gather(x_n * scores, 1, comp_index)
            extra_token = non_topk.sum(dim=1, keepdim=True)

            x = torch.cat([left_x, extra_token], dim=1)

            x_n2 = self.norm2(x)
            moe_out, moe_loss = self.moe(x_n2)
            incep_out = self.inception(x_n2.permute(0, 2, 1))
            incep_out = self.incep_proj(incep_out.permute(0, 2, 1))
            x = x + moe_out + incep_out
        else:
            x = x_n * scores
            moe_loss = 0.0
            incep_out = self.inception(self.norm2(x).permute(0, 2, 1))
            incep_out = self.incep_proj(incep_out.permute(0, 2, 1))
            x = x + incep_out

        end_scores = None
        if is_last:
            x = self.drop(x)
            end_scores = self.attn_head(x)

        return F.gelu(x), moe_loss, end_scores


class _SoftShapeNet(nn.Module):

    def __init__(self, in_channels, seq_len, emb_dim=128, depth=2,
                 sparse_rate=0.5, shape_size=8, stride=4, num_experts=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.sparse_rate = sparse_rate

        shape_size = min(shape_size, seq_len)
        stride = min(stride, shape_size)
        num_patches = max((seq_len - shape_size) // stride + 1, 1)

        self.shape_embed = nn.Conv1d(in_channels, emb_dim,
                                     kernel_size=shape_size, stride=stride)
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(0.15)

        self.attn_head = nn.Sequential(
            nn.Linear(emb_dim, 8), nn.Tanh(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.moe = _MoEBlock(emb_dim, num_experts=num_experts)

        self.sparse_schedule = [
            x.item() for x in torch.linspace(0, sparse_rate, depth)
        ]
        self.blocks = nn.ModuleList([
            _SoftShapeLayer(emb_dim, self.moe, self.attn_head)
            for _ in range(depth)
        ])

        self.head = nn.Linear(emb_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch=100, warm_up=50):
        x = self.shape_embed(x)
        x = x.transpose(1, 2)

        N = x.shape[1]
        if N != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=N,
                mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed

        x = x + pos
        x = self.pos_drop(x)

        total_moe_loss = 0.0
        end_scores = None

        for i, blk in enumerate(self.blocks):
            ratio = 1.0 - self.sparse_schedule[i]
            if epoch < warm_up:
                ratio = 1.0
            is_last = (i == self.depth - 1)
            x, moe_loss, end_scores = blk(x, remain_ratio=ratio,
                                           is_last=is_last)
            total_moe_loss = total_moe_loss + moe_loss

        if end_scores is not None:
            logits = self.head(x)
            weighted = logits * end_scores
            cls_logits = weighted.mean(dim=1)
        else:
            cls_logits = self.head(x.mean(dim=1))

        return cls_logits, total_moe_loss


class SoftShapeBaseline(_NNBaseline):

    name = "SoftShape"

    def __init__(self, emb_dim=128, depth=2, sparse_rate=0.5, shape_size=8,
                 stride=4, warm_up_epoch=50, moe_loss_rate=0.001,
                 epochs=500, batch_size=64, lr=1e-3, patience=30,
                 device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, device=device)
        self.emb_dim = emb_dim
        self.depth = depth
        self.sparse_rate = sparse_rate
        self.shape_size = shape_size
        self.stride = stride
        self.warm_up_epoch = warm_up_epoch
        self.moe_loss_rate = moe_loss_rate
        self._current_epoch = 0

    def _build_model(self, seq_len, in_channels):
        return _SoftShapeNet(
            in_channels, seq_len, emb_dim=self.emb_dim, depth=self.depth,
            sparse_rate=self.sparse_rate, shape_size=self.shape_size,
            stride=self.stride, num_experts=2
        )

    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, moe_loss = self.model(
                xb, epoch=self._current_epoch, warm_up=self.warm_up_epoch
            )
            loss = criterion(logits.squeeze(-1), yb)
            if isinstance(moe_loss, torch.Tensor):
                loss = loss + self.moe_loss_rate * moe_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        self._current_epoch += 1
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            logits, _ = self.model(
                xb, epoch=self._current_epoch, warm_up=self.warm_up_epoch
            )
            loss = criterion(logits.squeeze(-1), yb)
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        X_pp = self._preprocess(X)
        loader = _make_loader(X_pp, np.zeros(len(X_pp)), self.batch_size,
                              device=self.device)
        probs = []
        for xb, _ in loader:
            logits, _ = self.model(
                xb, epoch=999, warm_up=self.warm_up_epoch
            )
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        return np.concatenate(probs)


class _Lookahead:

    def __init__(self, base_optimizer, alpha=0.5, k=6):
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self._step_count = 0
        self.state = defaultdict(dict)

    def _update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(fast_p.data - slow, alpha=self.alpha)
            fast_p.data.copy_(slow)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group in self.param_groups:
                self._update_slow(group)
        return loss


def _moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (
        torch.max(col) * torch.max(row))
    I = torch.eye(x.shape[-1], device=device).unsqueeze(0)
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class _NystromAttn(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, num_landmarks=256,
                 pinv_iterations=6, residual=True, residual_conv_kernel=33,
                 eps=1e-8, dropout=0.):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads, (residual_conv_kernel, 1),
                padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        m = self.num_landmarks
        iters = self.pinv_iterations
        eps = self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if mask is not None:
                mask = F.pad(mask, (padding, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale
        l_val = ceil(q.shape[2] / m)
        eq = '... (n l) d -> ... n d'
        q_land = reduce(q, eq, 'sum', l=l_val)
        k_land = reduce(k, eq, 'sum', l=l_val)

        divisor = l_val
        if mask is not None:
            mask_land_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l_val)
            divisor = mask_land_sum[..., None] + eps

        q_land = q_land / divisor
        k_land = k_land / divisor

        sim1 = torch.einsum('... i d, ... j d -> ... i j', q, k_land)
        sim2 = torch.einsum('... i d, ... j d -> ... i j', q_land, k_land)
        sim3 = torch.einsum('... i d, ... j d -> ... i j', q_land, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1),
                                   (sim1, sim2, sim3))
        attn2_inv = _moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        return out


class _TMTransLayer(nn.Module):

    def __init__(self, dim=512, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = _NystromAttn(
            dim=dim, dim_head=dim // 8, heads=8,
            num_landmarks=dim // 2, pinv_iterations=6,
            residual=True, dropout=dropout)

    def forward(self, x):
        return x + self.attn(self.norm(x))


def _tm_manual_pad(x, min_length):
    pad_amount = min_length - x.shape[-1]
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    return F.pad(x, [pad_left, pad_right], mode="constant", value=0.)


class _TMInceptionModule(nn.Module):

    def __init__(self, in_ch, out_ch=32, bn_ch=32, padding_mode="replicate"):
        super().__init__()
        if in_ch > 1:
            self.bottleneck = nn.Conv1d(in_ch, bn_ch, 1, padding="same",
                                        padding_mode=padding_mode)
        else:
            self.bottleneck = nn.Identity()
            bn_ch = 1
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_ch, out_ch, k, padding="same",
                      padding_mode=padding_mode)
            for k in [10, 20, 40]
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, padding=1, stride=1),
            nn.Conv1d(in_ch, out_ch, 1, padding="same",
                      padding_mode=padding_mode))
        self.act = nn.Sequential(nn.BatchNorm1d(4 * out_ch), nn.ReLU())

    def forward(self, x):
        x_bn = self.bottleneck(x)
        z = torch.cat([c(x_bn) for c in self.convs]
                      + [self.maxpool_conv(x)], dim=1)
        return self.act(z)


class _TMInceptionBlock(nn.Module):

    def __init__(self, in_ch, out_ch=32, bn_ch=32,
                 padding_mode="replicate", n_modules=3):
        super().__init__()
        mods = []
        for i in range(n_modules):
            mods.append(_TMInceptionModule(
                in_ch if i == 0 else out_ch * 4,
                out_ch, bn_ch, padding_mode))
        self.mods = nn.Sequential(*mods)
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, 4 * out_ch, 1, padding="same",
                      padding_mode=padding_mode),
            nn.BatchNorm1d(4 * out_ch))

    def forward(self, x):
        return F.relu(self.mods(x) + self.residual(x))


class _TMInceptionTimeExtractor(nn.Module):

    def __init__(self, n_in_channels, out_channels=32,
                 padding_mode="replicate"):
        super().__init__()
        self.encoder = nn.Sequential(
            _TMInceptionBlock(n_in_channels, out_channels, padding_mode=padding_mode),
            _TMInceptionBlock(out_channels * 4, out_channels, padding_mode=padding_mode))

    def forward(self, x):
        min_len = 21
        if x.shape[-1] < min_len:
            x = _tm_manual_pad(x, min_len)
        return self.encoder(x)


def _mexican_hat_wavelet(size, scale, shift):
    device = scale.device
    half = (size[1] - 1) // 2
    x = torch.linspace(-half, half, size[1], device=device)
    x = x.unsqueeze(0).expand(size[0], -1)
    x = x - shift
    C = 2 / (3 ** 0.5 * math.pi ** 0.25)
    wavelet = C * (1 - (x / scale) ** 2) * torch.exp(
        -(x / scale) ** 2 / 2) / (torch.abs(scale) ** 0.5)
    return wavelet


class _TMWaveletEncoding(nn.Module):

    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3):
        cls_token, feat = x[:, 0], x[:, 1:]
        xt = feat.transpose(1, 2)
        D = xt.shape[1]

        wk1 = _mexican_hat_wavelet((D, 19), wave1[0], wave1[1])
        wk2 = _mexican_hat_wavelet((D, 19), wave2[0], wave2[1])
        wk3 = _mexican_hat_wavelet((D, 19), wave3[0], wave3[1])

        pos1 = F.conv1d(xt, wk1.unsqueeze(1), groups=D, padding='same')
        pos2 = F.conv1d(xt, wk2.unsqueeze(1), groups=D, padding='same')
        pos3 = F.conv1d(xt, wk3.unsqueeze(1), groups=D, padding='same')

        pos_sum = (pos1 + pos2 + pos3).transpose(1, 2)
        feat = feat + self.proj(pos_sum)
        return torch.cat((cls_token.unsqueeze(1), feat), dim=1)


def _init_tm_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def _make_wave_param(dim, zero_shift=False):
    w = torch.zeros(2, dim, 1) if zero_shift else torch.randn(2, dim, 1)
    w[0] = torch.ones(dim, 1) + torch.randn(dim, 1)
    return nn.Parameter(w)


class _TimeMILNet(nn.Module):

    def __init__(self, in_channels, mDim=128, max_seq_len=400, dropout=0.2):
        super().__init__()
        self.feature_extractor = _TMInceptionTimeExtractor(
            n_in_channels=in_channels, out_channels=mDim // 4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = _make_wave_param(mDim)
        self.wave2 = _make_wave_param(mDim, zero_shift=True)
        self.wave3 = _make_wave_param(mDim, zero_shift=True)
        self.wave1_ = _make_wave_param(mDim)
        self.wave2_ = _make_wave_param(mDim, zero_shift=True)
        self.wave3_ = _make_wave_param(mDim, zero_shift=True)
        self.pos_layer = _TMWaveletEncoding(mDim)
        self.pos_layer2 = _TMWaveletEncoding(mDim)
        self.layer1 = _TMTransLayer(dim=mDim, dropout=dropout)
        self.layer2 = _TMTransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        self.head = nn.Sequential(
            nn.Linear(mDim, mDim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mDim, 1))
        self.alpha = nn.Parameter(torch.ones(1))
        _init_tm_weights(self)

    def forward(self, x, warmup=False):
        x = self.feature_extractor(x.transpose(1, 2))
        x = x.transpose(1, 2)
        global_token = x.mean(dim=1)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        x = self.layer1(x)
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        x = self.layer2(x)

        x = x[:, 0]
        if warmup:
            x = 0.1 * x + 0.99 * global_token
        return self.head(x)


class TimeMILBaseline(_NNBaseline):

    name = "TimeMIL"

    def __init__(self, epochs=300, batch_size=64, lr=5e-3, patience=30,
                 weight_decay=1e-4, dropout=0.2, dropout_patch=0.5,
                 warmup_epochs=10, embed_dim=128, device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.dropout_node = dropout
        self.dropout_patch = dropout_patch
        self.warmup_epochs = warmup_epochs
        self.embed_dim = embed_dim
        self._current_epoch = 0

    def _preprocess(self, X):
        return _normalize_per_sample(X)

    def _build_model(self, seq_len, in_channels):
        return _TimeMILNet(
            in_channels=in_channels, mDim=self.embed_dim,
            max_seq_len=seq_len, dropout=self.dropout_node)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_tr = self._preprocess(X_train)
        B, T, C = X_tr.shape
        self.model = self._build_model(seq_len=T, in_channels=C).to(
            self.device)

        tr_loader = _make_loader(X_tr, y_train, self.batch_size,
                                 shuffle=True, device=self.device)
        val_loader = None
        if X_val is not None:
            X_vl = self._preprocess(X_val)
            val_loader = _make_loader(X_vl, y_val, self.batch_size,
                                      device=self.device)

        pw = self._pos_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        base_opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        optimizer = _Lookahead(base_opt)

        best_val_loss = float("inf")
        best_state = None
        wait = 0
        self._current_epoch = 0

        for epoch in range(self.epochs):
            self._current_epoch = epoch
            self._train_epoch(tr_loader, optimizer, criterion)

            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        warmup = self._current_epoch < self.warmup_epochs
        total_loss = 0.0
        for xb, yb in loader:
            if self.dropout_patch > 0 and self.training_active():
                B_cur, T_cur, _ = xb.shape
                n_windows = 10
                interval = max(T_cur // n_windows, 1)
                n_drop = int(self.dropout_patch * n_windows)
                drop_idxs = random.sample(range(n_windows), n_drop)
                for idx in drop_idxs:
                    start = idx * interval
                    end = min(start + interval, T_cur)
                    xb[:, start:end, :] = torch.randn(1, device=xb.device)

            optimizer.zero_grad()
            logits = self.model(xb, warmup=warmup)
            loss = criterion(logits.squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    def training_active(self):
        return self.model.training

    @torch.no_grad()
    def _eval_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            logits = self.model(xb, warmup=False)
            loss = criterion(logits.squeeze(-1), yb)
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        X_pp = self._preprocess(X)
        loader = _make_loader(X_pp, np.zeros(len(X_pp)), self.batch_size,
                              device=self.device)
        probs = []
        for xb, _ in loader:
            logits = self.model(xb, warmup=False)
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        return np.concatenate(probs)


class _UniTSDropPath(nn.Module):

    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device).add_(keep).floor_()
        return x / keep * mask


class _UniTSGate(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):
        return x * self.proj(x).sigmoid()


class _UniTSSeqAttn(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = dropout
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class _UniTSVarAttn(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = dropout
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, V, P, C = x.shape
        qkv = self.qkv(x).reshape(B, V, P, 3, self.num_heads,
                                   self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q = q.mean(dim=1)
        k = k.mean(dim=1)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, V, -1)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.)
        x = x.view(B, self.num_heads, V, -1, P
                    ).permute(0, 2, 4, 1, 3).reshape(B, V, P, C)
        return self.proj_drop(self.proj(x))


class _UniTSCrossAttn(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, query):
        B, N, C = x.shape
        Q = query.shape[1]
        q = self.q_proj(query).reshape(
            B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(x).reshape(
            B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Q, C)
        return self.drop(self.out_proj(out))


class _UniTSSeqAttBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0., drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = _UniTSSeqAttn(dim, num_heads, dropout)
        self.gate = _UniTSGate(dim)
        self.dp = _UniTSDropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, V, L, C = x.shape
        h = self.attn(self.norm(x).reshape(B * V, L, C)).reshape(B, V, L, C)
        return x + self.dp(self.gate(h))


class _UniTSVarAttBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0., drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = _UniTSVarAttn(dim, num_heads, dropout)
        self.gate = _UniTSGate(dim)
        self.dp = _UniTSDropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dp(self.gate(self.attn(self.norm(x))))


class _UniTSFFNBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=8., dropout=0., drop_path=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Conv1d(dim, hidden, 3, padding=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)
        self.gate = _UniTSGate(dim)
        self.dp = _UniTSDropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, V, L, C = x.shape
        h = self.norm(x).reshape(B * V, L, C).transpose(1, 2)
        h = self.drop1(self.act(self.fc1(h)))
        h = self.drop2(self.fc2(self.ln(h.transpose(1, 2)))).reshape(B, V, L, C)
        return x + self.dp(self.gate(h))


class _UniTSBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=8., dropout=0., drop_path=0.):
        super().__init__()
        self.seq_att = _UniTSSeqAttBlock(dim, num_heads, dropout, drop_path)
        self.var_att = _UniTSVarAttBlock(dim, num_heads, dropout, drop_path)
        self.ffn = _UniTSFFNBlock(dim, mlp_ratio, dropout, drop_path)

    def forward(self, x):
        return self.ffn(self.var_att(self.seq_att(x)))


class _UniTSCLSHead(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.cross_att = _UniTSCrossAttn(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim))
        self.gate = _UniTSGate(dim)

    def forward(self, x, category_tokens):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x_flat = x.reshape(B * V, L, C)
        cls_tok = self.cross_att(x_flat, query=x_flat[:, -1:])
        cls_tok = cls_tok.reshape(B, V, 1, C)
        cls_tok = cls_tok + self.gate(self.mlp(self.norm(cls_tok)))
        M = category_tokens.shape[2]
        cls_exp = cls_tok.expand(B, V, M, C)
        cat_exp = category_tokens.expand(B, V, M, C)
        dist = torch.einsum('bvmc,bvmc->bm', cls_exp, cat_exp) / V
        return dist


class _UniTSNet(nn.Module):

    def __init__(self, in_channels, seq_len, d_model=64, n_heads=4,
                 e_layers=3, patch_len=8, prompt_num=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels

        actual_pl = min(patch_len, max(2, seq_len))
        remainder = seq_len % actual_pl
        self.pad_len = (actual_pl - remainder) if remainder != 0 else 0
        num_patches = (seq_len + self.pad_len) // actual_pl
        self.actual_pl = actual_pl

        self.patch_embed = nn.Linear(actual_pl, d_model, bias=False)

        pe = torch.zeros(1, 1, num_patches, d_model)
        pos = torch.arange(num_patches).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float()
               * -(math.log(10000.0) / d_model)).exp()
        pe[0, 0, :, 0::2] = torch.sin(pos * div)
        pe[0, 0, :, 1::2] = torch.cos(pos * div)
        self.pos_embed = nn.Parameter(pe)

        self.prompt_tokens = nn.Parameter(
            torch.randn(1, 1, prompt_num, d_model) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, 1, d_model) * 0.02)
        self.category_tokens = nn.Parameter(
            torch.randn(1, 1, 2, d_model) * 0.02)
        self.prompt_num = prompt_num
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _UniTSBlock(d_model, n_heads, mlp_ratio=8.,
                        dropout=dropout) for _ in range(e_layers)])
        self.cls_head = _UniTSCLSHead(d_model, n_heads, dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, T, C = x.shape

        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        x = x.unfold(-1, self.actual_pl, self.actual_pl)
        nP = x.shape[2]
        x = self.patch_embed(
            x.reshape(B * C, nP, self.actual_pl))
        x = x.reshape(B, C, nP, self.d_model)

        x = x + self.pos_embed[:, :, :nP, :]
        prompt = self.prompt_tokens.expand(B, C, -1, -1)
        cls_tok = self.cls_token.expand(B, C, -1, -1)
        x = torch.cat([prompt, x, cls_tok], dim=2)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        cat_tok = self.category_tokens.expand(1, C, -1, -1)
        logits_2 = self.cls_head(x, cat_tok)
        out = logits_2[:, 1:2] - logits_2[:, 0:1]
        return out.clamp(-30, 30)


class UniTSBaseline(_NNBaseline):

    name = "UniTS"

    def __init__(self, d_model=64, n_heads=4, e_layers=3, patch_len=8,
                 prompt_num=5, epochs=500, batch_size=64, lr=1e-3,
                 patience=30, weight_decay=1e-4, device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.patch_len = patch_len
        self.prompt_num = prompt_num

    def _build_model(self, seq_len, in_channels):
        pl = min(self.patch_len, max(2, seq_len // 3))
        return _UniTSNet(in_channels, seq_len, d_model=self.d_model,
                         n_heads=self.n_heads, e_layers=self.e_layers,
                         patch_len=pl, prompt_num=self.prompt_num,
                         dropout=0.1)

    def _preprocess(self, X):
        X = X.astype(np.float32).copy()
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.log1p(np.clip(X, 0, None))
        return X.transpose(0, 2, 1)


class _MambaSelectiveScan(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        d_inner = int(d_model * expand)
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True)

        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())

        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)

        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = F.silu(x_branch).transpose(1, 2)

        ssm_params = self.x_proj(x_branch)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        dt_raw = ssm_params[:, :, -1:]
        dt = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log)

        y = self._scan(x_branch, dt, A, B_param, C_param)

        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.drop(self.out_proj(y))

    def _scan(self, x, dt, A, B_in, C_in):
        B_sz, L, d_inner = x.shape
        N = self.d_state

        h = torch.zeros(B_sz, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :]
            B_t = B_in[:, t, :]
            C_t = C_in[:, t, :]
            x_t = x[:, t, :]

            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = h * dA + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)


class _MambaBlock(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _MambaSelectiveScan(d_model, d_state, d_conv, expand, dropout)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class _MambaNet(nn.Module):

    def __init__(self, in_channels, seq_len, d_model=64, d_state=16,
                 d_conv=4, expand=2, n_layers=4, patch_len=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        actual_pl = min(patch_len, max(2, seq_len))
        remainder = seq_len % actual_pl
        self.pad_len = (actual_pl - remainder) if remainder != 0 else 0
        self.actual_pl = actual_pl
        num_patches = (seq_len + self.pad_len) // actual_pl

        self.patch_embed = nn.Linear(actual_pl * in_channels, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.fwd_blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)])
        self.bwd_blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, T = x.shape
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        x = x.unfold(-1, self.actual_pl, self.actual_pl)
        nP = x.shape[2]
        x = x.permute(0, 2, 1, 3).reshape(B, nP, -1)
        x = self.patch_embed(x) + self.pos_embed[:, :nP, :]
        x = self.drop(x)

        x_fwd = x
        for blk in self.fwd_blocks:
            x_fwd = blk(x_fwd)

        x_bwd = torch.flip(x, dims=[1])
        for blk in self.bwd_blocks:
            x_bwd = blk(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        x = self.norm(x_fwd + x_bwd)
        x = x.mean(dim=1)
        return self.head(x)


class MambaBaseline(_NNBaseline):

    name = "Mamba"

    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2,
                 n_layers=4, patch_len=4, epochs=500, batch_size=64,
                 lr=1e-3, patience=30, weight_decay=1e-4, device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.patch_len = patch_len

    def _build_model(self, seq_len, in_channels):
        pl = min(self.patch_len, max(2, seq_len // 3))
        return _MambaNet(in_channels, seq_len, d_model=self.d_model,
                         d_state=self.d_state, d_conv=self.d_conv,
                         expand=self.expand, n_layers=self.n_layers,
                         patch_len=pl, dropout=0.1)


class _GRUDCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_in = input_size * 2 + hidden_size
        self.W_z = nn.Linear(gate_in, hidden_size)
        self.W_r = nn.Linear(gate_in, hidden_size)
        self.W_h = nn.Linear(gate_in, hidden_size)

        self.W_gamma_x = nn.Linear(input_size, input_size, bias=False)
        self.W_gamma_h = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x_t, m_t, delta_t, h_prev, x_mean):
        gamma_x = torch.exp(-F.relu(self.W_gamma_x(delta_t)))
        gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta_t)))

        x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_t + (1 - gamma_x) * x_mean)

        h_decayed = gamma_h * h_prev

        combined = torch.cat([x_hat, m_t, h_decayed], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))

        combined_r = torch.cat([x_hat, m_t, r * h_decayed], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        h_new = (1 - z) * h_decayed + z * h_tilde
        return h_new


class _GRUDNet(nn.Module):

    def __init__(self, in_channels, seq_len, hidden_size=64,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.cells = nn.ModuleList([
            _GRUDCell(in_channels if i == 0 else hidden_size, hidden_size)
            for i in range(n_layers)])

        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        B, C, T = x.shape
        x = x.permute(0, 2, 1)

        mask = (x != 0).float()

        delta = torch.zeros_like(x)
        for t in range(1, T):
            delta[:, t, :] = torch.where(
                mask[:, t - 1, :] == 1,
                torch.ones_like(delta[:, t, :]),
                delta[:, t - 1, :] + 1)

        obs_sum = (x * mask).sum(dim=1)
        obs_cnt = mask.sum(dim=1).clamp(min=1)
        x_mean = (obs_sum / obs_cnt).mean(dim=0)

        h = [torch.zeros(B, self.hidden_size, device=x.device)
             for _ in range(self.n_layers)]

        for t in range(T):
            inp = x[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    h[layer_idx] = cell(inp, m_t, d_t, h[layer_idx], x_mean)
                else:
                    h_in = self.drop(h[layer_idx - 1])
                    m_ones = torch.ones(B, self.hidden_size, device=x.device)
                    d_zeros = torch.zeros(B, self.hidden_size, device=x.device)
                    h_mean = torch.zeros(self.hidden_size, device=x.device)
                    h[layer_idx] = cell(h_in, m_ones, d_zeros,
                                        h[layer_idx], h_mean)

        return self.head(self.drop(h[-1]))


class GRUDBaseline(_NNBaseline):

    name = "GRU-D"

    def __init__(self, hidden_size=64, n_layers=2, epochs=500,
                 batch_size=64, lr=1e-3, patience=30, weight_decay=1e-4,
                 device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def _build_model(self, seq_len, in_channels):
        return _GRUDNet(in_channels, seq_len,
                        hidden_size=self.hidden_size,
                        n_layers=self.n_layers, dropout=0.1)

    def _preprocess(self, X):
        X = X.astype(np.float32).copy()
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.transpose(0, 2, 1)


class _MambaSLBlock(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 tv_dt=True, tv_B=True, tv_C=True, use_D=True,
                 dropout=0.1):
        super().__init__()
        d_inner = int(d_model * expand)
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        dt_rank = max(1, math.ceil(d_model / 16))
        self.dt_rank = dt_rank
        self.tv_dt = tv_dt
        self.tv_B = tv_B
        self.tv_C = tv_C

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        if d_conv > 0:
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=d_conv,
                padding=d_conv - 1, groups=d_inner, bias=True)
        else:
            self.conv1d = None

        proj_dim = 0
        self.tv_proj_dims = [0, 0, 0]
        if tv_dt:
            self.tv_proj_dims[0] = dt_rank
            proj_dim += dt_rank
        if tv_B:
            self.tv_proj_dims[1] = d_state
            proj_dim += d_state
        if tv_C:
            self.tv_proj_dims[2] = d_state
            proj_dim += d_state
        self.x_proj = nn.Linear(d_inner, proj_dim, bias=False) if proj_dim > 0 else None

        if not tv_B:
            self.B_const = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        if not tv_C:
            self.C_const = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)

        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())

        self.D = nn.Parameter(torch.ones(d_inner)) if use_D else None

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B_sz, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        if self.conv1d is not None:
            x_branch = x_branch.transpose(1, 2)
            x_branch = self.conv1d(x_branch)[:, :, :L]
            x_branch = self.act(x_branch).transpose(1, 2)
        else:
            x_branch = self.act(x_branch)

        if self.x_proj is not None:
            x_dbl = self.x_proj(x_branch.reshape(B_sz * L, -1))
            splits = [d for d in self.tv_proj_dims if d > 0]
            parts = torch.split(x_dbl, splits, dim=-1)
            idx = 0

        if self.tv_dt:
            dt_raw = parts[idx].reshape(B_sz, L, self.dt_rank)
            dt = F.softplus(
                torch.einsum('bld,od->blo', dt_raw, self.dt_proj.weight)
                + self.dt_proj.bias)
            idx += 1
        else:
            dt = F.softplus(self.dt_proj.bias).unsqueeze(0).unsqueeze(0).expand(B_sz, L, -1)

        if self.tv_B:
            B_param = parts[idx].reshape(B_sz, L, self.d_state)
            idx += 1
        else:
            B_param = self.B_const

        if self.tv_C:
            C_param = parts[idx].reshape(B_sz, L, self.d_state)
        else:
            C_param = self.C_const

        A = -torch.exp(self.A_log)

        y = self._scan(x_branch, dt, A, B_param, C_param)

        if self.D is not None:
            y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(z)
        y = self.out_proj(y)
        return self.drop(self.norm(self.act(y)))

    def _scan(self, x, dt, A, B_in, C_in):
        B_sz, L, d_inner = x.shape
        N = self.d_state
        tv_B = isinstance(B_in, torch.Tensor) and B_in.dim() == 3
        tv_C = isinstance(C_in, torch.Tensor) and C_in.dim() == 3

        h = torch.zeros(B_sz, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :]
            x_t = x[:, t, :]
            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))

            if tv_B:
                B_t = B_in[:, t, :]
                dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            else:
                dB = dt_t.unsqueeze(-1) * B_in.unsqueeze(0)

            h = h * dA + dB * x_t.unsqueeze(-1)

            if tv_C:
                C_t = C_in[:, t, :]
                y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            else:
                y_t = (h * C_in.unsqueeze(0)).sum(dim=-1)

            ys.append(y_t)
        return torch.stack(ys, dim=1)


class _MambaSLNet(nn.Module):

    def __init__(self, in_channels, seq_len, d_model=64, d_state=16,
                 d_conv=4, expand=2, d_kernel=3, n_heads=4,
                 tv_dt=True, tv_B=True, tv_C=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Conv1d(in_channels, d_model, kernel_size=d_kernel,
                                   padding=(d_kernel - 1) // 2, bias=False)
        nn.init.kaiming_normal_(self.token_emb.weight, mode='fan_in',
                                nonlinearity='leaky_relu')
        self.pos_emb = nn.Parameter(torch.zeros(1, max(5000, seq_len), d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.emb_drop = nn.Dropout(dropout)

        self.mamba = _MambaSLBlock(d_model, d_state=d_state, d_conv=d_conv,
                                   expand=expand, tv_dt=tv_dt, tv_B=tv_B,
                                   tv_C=tv_C, dropout=dropout)

        self.out_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1, bias=False))
        nn.init.xavier_uniform_(self.out_layer[1].weight)

        self.attn_weight = nn.Sequential(
            nn.Linear(d_model, n_heads, bias=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Softmax(dim=1))
        for m in self.attn_weight.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1.0)

    def forward(self, x):
        x = self.token_emb(x).transpose(1, 2)
        T_out = x.shape[1]
        x = x + self.pos_emb[:, :T_out, :]
        x = self.emb_drop(x)

        x = self.mamba(x)

        logit = self.out_layer(x)
        w = self.attn_weight(x)
        out = (logit * w).sum(dim=1)
        return out.clamp(-30, 30)


class MambaSLBaseline(_NNBaseline):

    name = "MambaSL"

    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2,
                 d_kernel=3, n_heads=4, epochs=500, batch_size=64,
                 lr=1e-3, patience=30, weight_decay=1e-4, device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_kernel = d_kernel
        self.n_heads = n_heads

    def _build_model(self, seq_len, in_channels):
        dk = min(self.d_kernel, max(2, seq_len // 2))
        return _MambaSLNet(in_channels, seq_len, d_model=self.d_model,
                           d_state=self.d_state, d_conv=self.d_conv,
                           expand=self.expand, d_kernel=dk,
                           n_heads=self.n_heads, dropout=0.1)


class _ShapeletModule(nn.Module):

    def __init__(self, n_channels, shapelet_len, num_shapelets=5, eps=1.0):
        super().__init__()
        self.n = num_shapelets
        self.length = shapelet_len
        self.eps = eps
        self.weights = nn.Parameter(
            torch.randn(num_shapelets, n_channels, shapelet_len) * 0.1)

    def forward(self, x):
        x_unf = x.unfold(2, self.length, 1)
        x_unf = x_unf.permute(0, 2, 1, 3).unsqueeze(2)
        d = (x_unf - self.weights).abs().mean(dim=-1)
        p = torch.exp(-(self.eps * d) ** 2)
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.0)
        soft = torch.softmax(p, dim=1)
        onehot = hard + soft - soft.detach()
        max_p = (onehot * p).sum(dim=1)
        min_d = d.min(dim=1).values
        return max_p.flatten(1), min_d.flatten(1)


class _ShapeletBottleneckModel(nn.Module):

    def __init__(self, n_channels, seq_len, num_class=2,
                 num_shapelet=(5, 5, 5, 5),
                 shapelet_frac=(0.1, 0.2, 0.3, 0.5),
                 eps=1.0, lambda_reg=0.01, lambda_div=0.01, dropout=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div

        self.shapelets = nn.ModuleList()
        for ns, frac in zip(num_shapelet, shapelet_frac):
            sl = max(3, int(math.ceil(frac * seq_len)))
            self.shapelets.append(
                _ShapeletModule(n_channels, sl, num_shapelets=ns, eps=eps))

        total_feat = sum(num_shapelet) * n_channels
        self.classifier = nn.Linear(total_feat, num_class, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        probs, dists = [], []
        for s in self.shapelets:
            p, d = s(x_norm)
            probs.append(p)
            dists.append(d)
        feat = torch.cat(probs, dim=-1)
        logits = self.classifier(self.drop(feat))
        return logits, feat

    def reg_loss(self):
        loss = self.classifier.weight.abs().mean() * self.lambda_reg
        if self.lambda_div > 0:
            for s in self.shapelets:
                sh = s.weights.permute(1, 0, 2)
                d = torch.cdist(sh, sh)
                mask = 1.0 - torch.eye(sh.shape[1], device=d.device).unsqueeze(0)
                loss = loss + (torch.exp(-d) * mask).mean() * self.lambda_div
        return loss


class _FCNDeep(nn.Module):

    def __init__(self, in_channels, seq_len, num_class=2):
        super().__init__()
        if seq_len <= 10:
            ks = (3, 3, 2)
        else:
            ks = (8, 5, 3)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, ks[0], padding=(ks[0] - 1) // 2),
            nn.BatchNorm1d(128), nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, ks[1], padding=(ks[1] - 1) // 2),
            nn.BatchNorm1d(256), nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, ks[2], padding=(ks[2] - 1) // 2),
            nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class _InterpGNNet(nn.Module):

    def __init__(self, in_channels, seq_len, num_shapelet=(5, 5, 5, 5),
                 shapelet_frac=(0.1, 0.2, 0.3, 0.5), eps=1.0,
                 lambda_reg=0.01, lambda_div=0.01, dropout=0.1):
        super().__init__()
        self.sbm = _ShapeletBottleneckModel(
            in_channels, seq_len, num_class=2,
            num_shapelet=num_shapelet, shapelet_frac=shapelet_frac,
            eps=eps, lambda_reg=lambda_reg, lambda_div=lambda_div,
            dropout=dropout)
        self.fcn = _FCNDeep(in_channels, seq_len, num_class=2)

    def forward(self, x):
        sbm_logits, _ = self.sbm(x)
        fcn_logits = self.fcn(x)

        p = F.softmax(sbm_logits, dim=-1)
        gini = p.pow(2).sum(-1, keepdim=True)
        sbm_util = (2.0 * gini - 1.0)
        deep_util = 1.0 - sbm_util

        combined = sbm_util * sbm_logits + deep_util * fcn_logits
        out = combined[:, 1:2] - combined[:, 0:1]
        return out.clamp(-30, 30)

    def aux_loss(self):
        return self.sbm.reg_loss()


class InterpGNBaseline(_NNBaseline):

    name = "InterpGN"

    def __init__(self, num_shapelet=(5, 5, 5, 5),
                 shapelet_frac=(0.1, 0.2, 0.3, 0.5), eps=1.0,
                 lambda_reg=0.01, lambda_div=0.01,
                 epochs=500, batch_size=64, lr=1e-3,
                 patience=30, weight_decay=1e-4, device=None):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr,
                         patience=patience, weight_decay=weight_decay,
                         device=device)
        self.num_shapelet = num_shapelet
        self.shapelet_frac = shapelet_frac
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div

    def _build_model(self, seq_len, in_channels):
        return _InterpGNNet(in_channels, seq_len,
                            num_shapelet=self.num_shapelet,
                            shapelet_frac=self.shapelet_frac,
                            eps=self.eps, lambda_reg=self.lambda_reg,
                            lambda_div=self.lambda_div, dropout=0.1)

    def _train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = self.model(xb)
            loss = criterion(logits.squeeze(-1), yb)
            if hasattr(self.model, 'aux_loss'):
                loss = loss + self.model.aux_loss()
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)


STAT_BASELINES = [CrostonBaseline, SBABaseline, ZIPBaseline, HurdleBaseline]
NN_BASELINES = [DSNBaseline, SoftShapeBaseline, TimeMILBaseline,
                UniTSBaseline, MambaBaseline, GRUDBaseline,
                MambaSLBaseline, InterpGNBaseline]
ALL_BASELINES = STAT_BASELINES + NN_BASELINES


def run_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test,
                      label="", print_fn=print):
    from cast.utils.metrics import print_metrics

    results = []

    for cls in STAT_BASELINES:
        model = cls()
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_m = model.evaluate(X_train, y_train)
        val_m = model.evaluate(X_val, y_val)
        test_m = model.evaluate(X_test, y_test)

        print_fn(f"\n{'='*60}")
        print_fn(f"Baseline: {model.name} | {label} | fit {elapsed:.1f}s")
        print_fn(f"{'='*60}")
        print_metrics(train_m, prefix="[Train] ")
        print_metrics(val_m, prefix="[Val] ")
        print_metrics(test_m, prefix="[Test] ")

        results.append({
            "model": model.name,
            "label": label,
            "elapsed": round(elapsed, 2),
            "train": train_m,
            "val": val_m,
            "test": test_m,
        })

    for cls in NN_BASELINES:
        model = cls()
        t0 = time.time()
        model.fit(X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0

        train_m = model.evaluate(X_train, y_train)
        val_m = model.evaluate(X_val, y_val)
        test_m = model.evaluate(X_test, y_test)

        print_fn(f"\n{'='*60}")
        print_fn(f"Baseline: {model.name} | {label} | fit {elapsed:.1f}s")
        print_fn(f"{'='*60}")
        print_metrics(train_m, prefix="[Train] ")
        print_metrics(val_m, prefix="[Val] ")
        print_metrics(test_m, prefix="[Test] ")

        results.append({
            "model": model.name,
            "label": label,
            "elapsed": round(elapsed, 2),
            "train": train_m,
            "val": val_m,
            "test": test_m,
        })
