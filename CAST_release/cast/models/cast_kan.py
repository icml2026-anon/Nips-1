import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cast.models.kan_layers import build_kan_layer


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).float()
        return x * mask / keep


class CausalPhaseRotation(nn.Module):

    def __init__(self, dim, scale=math.pi / 4.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.disabled = False

    def forward(self, x, phi):
        if self.disabled:
            return x
        d = phi.shape[-1]
        D = x.shape[-1]
        if D != 2 * d:
            if D < 2 * d:
                x = F.pad(x, (0, 2 * d - D))
            else:
                x = x[..., :2 * d]
            D = 2 * d
        angles = self.scale * phi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        min_d = min(x1.shape[-1], cos_a.shape[-1])
        x1 = x1[..., :min_d]
        x2 = x2[..., :min_d]
        cos_a = cos_a[..., :min_d]
        sin_a = sin_a[..., :min_d]
        y1 = x1 * cos_a - x2 * sin_a
        y2 = x1 * sin_a + x2 * cos_a
        out = torch.stack([y1, y2], dim=-1).reshape(*x.shape[:-1], min_d * 2)
        return out


class CausalGate(nn.Module):

    def __init__(self, embedding_dim, num_basis):
        super().__init__()
        self.gate_linear = nn.Linear(embedding_dim, num_basis)
        self.disabled = False

    def forward(self, phi):
        if self.disabled:
            return torch.ones(*phi.shape[:-1], self.gate_linear.out_features,
                              device=phi.device, dtype=phi.dtype)
        return torch.sigmoid(self.gate_linear(phi))


class SampleAdaptiveSparsityGate(nn.Module):

    def __init__(self, seq_len, embedding_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.adapt_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, mask, phi):
        context = self.mlp(mask)
        gate = torch.sigmoid(self.adapt_linear(context))
        if phi.dim() == 2 and gate.dim() == 2:
            if phi.shape[0] != gate.shape[0]:
                adapted = phi.unsqueeze(0) * gate.unsqueeze(1)
            else:
                adapted = phi * gate
        elif phi.dim() == 2 and gate.dim() == 3:
            adapted = phi.unsqueeze(0) * gate
        else:
            adapted = phi * gate
        return adapted


class CASTKANLayer(nn.Module):

    def __init__(self, in_features, out_features, embedding_dim, config,
                 drop_path_rate=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kan = build_kan_layer(config.kan_type, in_features, out_features, config)
        self.causal_gate = CausalGate(embedding_dim, self.kan.basis_size())
        self.phase_rot = CausalPhaseRotation(embedding_dim, config.phase_scale)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, phi):
        residual = x
        x = self.norm(x)
        gates = self.causal_gate(phi)
        if x.dim() == 3:
            B, T, C = x.shape
            if phi.dim() == 3 and C == phi.shape[-1] * 2:
                phi_flat = phi.reshape(B * T, -1)
                x_flat = x.reshape(B * T, C)
                x_rot = self.phase_rot(x_flat, phi_flat).reshape(B, T, -1)
            elif phi.dim() == 2 and C == phi.shape[-1] * 2:
                if phi.shape[0] == T:
                    phi_exp = phi.unsqueeze(0).expand(B, -1, -1).reshape(B * T, -1)
                else:
                    phi_exp = phi.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                x_flat = x.reshape(B * T, C)
                x_rot = self.phase_rot(x_flat, phi_exp).reshape(B, T, -1)
            else:
                x_rot = x

            if gates.dim() == 3:
                g_flat = gates.reshape(B * T, -1)
            elif gates.dim() == 2 and gates.shape[0] == T:
                g_flat = gates.unsqueeze(0).expand(B, -1, -1).reshape(B * T, -1)
            else:
                g_flat = gates.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

            xr_flat = x_rot.reshape(B * T, -1)
            out_flat = self.kan(xr_flat, gates=g_flat)
            out = out_flat.reshape(B, T, self.out_features)
        else:
            x_rot = self.phase_rot(x, phi) if x.shape[-1] == phi.shape[-1] * 2 else x
            out = self.kan(x_rot, gates=gates)

        out = self.dropout(out)
        if out.shape == residual.shape:
            out = residual + self.drop_path(out)
        return out


class CASTConvLayer(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, config, drop_path_rate=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        h3 = hidden_dim // 3
        h5 = hidden_dim // 3
        h7 = hidden_dim - h3 - h5
        self.conv3 = nn.Conv1d(hidden_dim, h3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, h5, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(hidden_dim, h7, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.causal_gate = CausalGate(embedding_dim, hidden_dim)
        self.phase_rot = CausalPhaseRotation(embedding_dim, config.phase_scale)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, phi):
        residual = x
        x = self.norm(x)
        xt = x.transpose(1, 2)
        c3 = self.conv3(xt)
        c5 = self.conv5(xt)
        c7 = self.conv7(xt)
        x_conv = torch.cat([c3, c5, c7], dim=1)
        x_conv = self.bn(x_conv).transpose(1, 2)
        x_conv = F.gelu(x_conv)
        gates = self.causal_gate(phi)
        if gates.dim() == 2:
            gates = gates.unsqueeze(0)
        x_conv = x_conv * gates
        x_conv = self.dropout(x_conv)
        return residual + self.drop_path(x_conv)


class MeanMaxPooling(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, phi=None):
        mean_out = x.mean(dim=1)
        max_out = x.max(dim=1)[0]
        return self.out_proj(torch.cat([mean_out, max_out], dim=-1))


class CausalAttentionPooling(nn.Module):

    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.causal_bias = nn.Linear(embedding_dim, 1)
        self.scale = hidden_dim ** 0.5
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, phi):
        B = x.shape[0]
        q = self.cls_query.expand(B, -1, -1)
        k = self.key(x)
        attn = (q * k).sum(dim=-1) / self.scale
        if phi is not None:
            if phi.dim() == 3:
                causal_weight = self.causal_bias(phi).squeeze(-1)
            elif phi.dim() == 2:
                causal_weight = self.causal_bias(phi).squeeze(-1)
                if causal_weight.dim() == 1:
                    causal_weight = causal_weight.unsqueeze(0).expand(B, -1)
            else:
                causal_weight = torch.zeros(B, x.shape[1], device=x.device)
            attn = attn + causal_weight
        attn = torch.softmax(attn, dim=-1)
        attn_out = (x * attn.unsqueeze(-1)).sum(dim=1)
        max_out = x.max(dim=1)[0]
        return self.out_proj(torch.cat([attn_out, max_out], dim=-1))


class CASTKANSequenceModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_channels, config.kan_hidden_dim)
        self.input_norm = nn.LayerNorm(config.kan_hidden_dim)

        n = config.kan_num_layers
        dpr = [0.1 * i / max(n - 1, 1) for i in range(n)]
        layers = []
        for i in range(n):
            if config.kan_type == "conv":
                layers.append(
                    CASTConvLayer(
                        config.kan_hidden_dim, config.embedding_dim,
                        config, drop_path_rate=dpr[i],
                    )
                )
            else:
                layers.append(
                    CASTKANLayer(
                        config.kan_hidden_dim, config.kan_hidden_dim,
                        config.embedding_dim, config,
                        drop_path_rate=dpr[i],
                    )
                )
        self.kan_layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(config.kan_hidden_dim)
        if getattr(config, 'ablate_no_causal_attn', False):
            self.attn_pool = MeanMaxPooling(config.kan_hidden_dim)
        else:
            self.attn_pool = CausalAttentionPooling(config.kan_hidden_dim, config.embedding_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.kan_hidden_dim),
            nn.Linear(config.kan_hidden_dim, config.kan_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.kan_hidden_dim, config.kan_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.kan_hidden_dim // 2, 1)
        )

    def forward(self, x, phi):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.config.kan_hidden_dim:
            x = self.input_norm(self.input_proj(x))
        for layer in self.kan_layers:
            x = layer(x, phi)
        x = self.final_norm(x)
        if x.dim() == 3:
            h = self.attn_pool(x, phi)
        else:
            h = x
        logits = self.classifier(h).squeeze(-1)
        return logits
