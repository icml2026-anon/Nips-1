import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TaylorKANLayer(nn.Module):

    def __init__(self, in_features, out_features, order=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, order) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        powers = torch.stack(
            [x_flat.pow(k) for k in range(self.order)], dim=-1
        )
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.order:
                g = g[..., :self.order]
            powers = powers * g.unsqueeze(-2)
        out = torch.einsum("bik,oik->bo", powers, self.coeffs) + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.order


class BSplineKANLayer(nn.Module):

    def __init__(self, in_features, out_features, num_knots=10, spline_order=3,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.num_bases = num_knots + spline_order - 1

        h = (grid_range[1] - grid_range[0]) / num_knots
        knots = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            num_knots + 2 * spline_order + 1
        )
        self.register_buffer("knots", knots)
        self.coeffs = nn.Parameter(
            torch.empty(in_features, out_features, self.num_bases)
        )
        nn.init.normal_(self.coeffs, mean=0.0, std=1.0 / (in_features * self.num_bases))
        self.base_weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def _bspline_basis(self, x):
        t = self.knots
        p = self.spline_order
        n_basis = len(t) - p - 1
        x_expand = x.unsqueeze(-1)
        bases = ((x_expand >= t[:-1].unsqueeze(0).unsqueeze(0)) &
                 (x_expand < t[1:].unsqueeze(0).unsqueeze(0))).float()
        bases = bases[..., :len(t) - 1]
        for k in range(1, p + 1):
            n_k = len(t) - k - 1
            left_num = x_expand - t[:n_k].unsqueeze(0).unsqueeze(0)
            left_den = (t[k:k + n_k] - t[:n_k]).unsqueeze(0).unsqueeze(0)
            left = torch.where(left_den.abs() > 1e-8, left_num / left_den, torch.zeros_like(left_num))
            right_num = t[k + 1:k + 1 + n_k].unsqueeze(0).unsqueeze(0) - x_expand
            right_den = (t[k + 1:k + 1 + n_k] - t[1:1 + n_k]).unsqueeze(0).unsqueeze(0)
            right = torch.where(right_den.abs() > 1e-8, right_num / right_den, torch.zeros_like(right_num))
            bases = left * bases[..., :n_k] + right * bases[..., 1:n_k + 1]
        return bases[..., :n_basis]

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        lo = self.knots[self.spline_order]
        hi = self.knots[-self.spline_order - 1]
        x_clamped = (torch.tanh(x_flat) + 1.0) * 0.5 * (hi - lo) + lo
        base_out = torch.einsum("bi,io->bo", F.silu(x_flat), self.base_weight)
        bases = self._bspline_basis(x_clamped)
        bases = bases[..., :self.num_bases]
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.num_bases:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.num_bases, mode='linear', align_corners=True
                ).squeeze(1)
            bases = bases * g.unsqueeze(-2)
        spline_out = torch.einsum("bik,iok->bo", bases, self.coeffs)
        out = base_out + spline_out + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.num_bases


class FourierKANLayer(nn.Module):

    def __init__(self, in_features, out_features, num_frequencies=8, omega=math.pi):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.omega = omega
        self.a0 = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.a0, a=math.sqrt(5))
        self.a_cos = nn.Parameter(
            torch.empty(in_features, out_features, num_frequencies)
        )
        nn.init.normal_(self.a_cos, mean=0.0, std=1.0 / math.sqrt(in_features * num_frequencies))
        self.b_sin = nn.Parameter(
            torch.empty(in_features, out_features, num_frequencies)
        )
        nn.init.normal_(self.b_sin, mean=0.0, std=1.0 / math.sqrt(in_features * num_frequencies))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)
        freqs = torch.arange(1, self.num_frequencies + 1, device=x.device, dtype=x.dtype)
        angles = x_norm.unsqueeze(-1) * freqs * self.omega
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.num_frequencies:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.num_frequencies, mode='linear', align_corners=True
                ).squeeze(1)
            cos_vals = cos_vals * g.unsqueeze(-2)
            sin_vals = sin_vals * g.unsqueeze(-2)
        dc = torch.einsum("bi,io->bo", x_norm, self.a0)
        cos_term = torch.einsum("bif,iof->bo", cos_vals, self.a_cos)
        sin_term = torch.einsum("bif,iof->bo", sin_vals, self.b_sin)
        out = dc + cos_term + sin_term + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.num_frequencies


class ChebyKANLayer(nn.Module):

    def __init__(self, in_features, out_features, degree=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)
        cheby = torch.ones(x_flat.shape[0], self.in_features, self.degree + 1,
                           device=x.device, dtype=x.dtype)
        if self.degree > 0:
            cheby[:, :, 1] = x_norm
        for n in range(2, self.degree + 1):
            cheby[:, :, n] = 2 * x_norm * cheby[:, :, n - 1].clone() - cheby[:, :, n - 2].clone()
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.degree + 1:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.degree + 1, mode='linear', align_corners=True
                ).squeeze(1)
            cheby = cheby * g.unsqueeze(-2)
        out = torch.einsum("bid,iod->bo", cheby, self.cheby_coeffs) + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.degree + 1


class JacobiKANLayer(nn.Module):

    def __init__(self, in_features, out_features, degree=8, a=1.0, b=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.a = a
        self.b = b
        self.jacobi_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1.0 / math.sqrt(in_features * (degree + 1)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        norms = self._precompute_norms()
        self.register_buffer("basis_norms", norms)

    def _precompute_norms(self):
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, 500)
            basis = torch.ones(500, self.degree + 1)
            if self.degree > 0:
                basis[:, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
            for i in range(2, self.degree + 1):
                theta_k = (
                    (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1)
                    / (2 * i * (i + self.a + self.b))
                )
                theta_k1 = (
                    (2 * i + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2)
                    / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
                )
                theta_k2 = (
                    (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b)
                    / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
                )
                basis[:, i] = (theta_k * x + theta_k1) * basis[:, i - 1] - theta_k2 * basis[:, i - 2]
            return basis.abs().amax(dim=0).clamp(min=1.0)

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)
        jacobi = torch.ones(x_flat.shape[0], self.in_features, self.degree + 1,
                            device=x.device, dtype=x.dtype)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x_norm) / 2
        for i in range(2, self.degree + 1):
            theta_k = (
                (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            jacobi[:, :, i] = (
                (theta_k * x_norm + theta_k1) * jacobi[:, :, i - 1].clone()
                - theta_k2 * jacobi[:, :, i - 2].clone()
            )
        jacobi = jacobi / self.basis_norms
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.degree + 1:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.degree + 1, mode='linear', align_corners=True
                ).squeeze(1)
            jacobi = jacobi * g.unsqueeze(-2)
        out = torch.einsum("bid,iod->bo", jacobi, self.jacobi_coeffs) + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.degree + 1


class RBFKANLayer(nn.Module):

    def __init__(self, in_features, out_features, num_centers=8, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        centers = torch.linspace(grid_range[0], grid_range[1], num_centers)
        self.centers = nn.Parameter(
            centers.unsqueeze(0).expand(in_features, num_centers).clone()
        )
        grid_spacing = (grid_range[1] - grid_range[0]) / max(num_centers - 1, 1)
        self.log_sigma = nn.Parameter(
            torch.full((in_features, num_centers), math.log(grid_spacing * 0.5))
        )
        self.coeffs = nn.Parameter(torch.empty(in_features, out_features, num_centers))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (in_features * num_centers))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)
        diff = x_norm.unsqueeze(-1) - self.centers.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).clamp(min=0.01)
        bases = torch.exp(-0.5 * diff.pow(2) / sigma.unsqueeze(0).pow(2))
        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.num_centers:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.num_centers, mode='linear', align_corners=True
                ).squeeze(1)
            bases = bases * g.unsqueeze(-2)
        out = torch.einsum("bik,iok->bo", bases, self.coeffs) + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.num_centers


class WaveletKANLayer(nn.Module):

    def __init__(self, in_features, out_features, num_wavelets=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets

        self.scale = nn.Parameter(
            torch.linspace(0.5, 2.0, num_wavelets)
            .unsqueeze(0).expand(in_features, num_wavelets).clone()
        )
        self.translation = nn.Parameter(
            torch.linspace(-1.0, 1.0, num_wavelets)
            .unsqueeze(0).expand(in_features, num_wavelets).clone()
        )
        self.coeffs = nn.Parameter(
            torch.empty(in_features, out_features, num_wavelets)
        )
        nn.init.normal_(self.coeffs, mean=0.0, std=1.0 / math.sqrt(in_features * num_wavelets))
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)

        s = self.scale.clamp(min=0.1)
        t = (x_norm.unsqueeze(-1) - self.translation.unsqueeze(0)) / s.unsqueeze(0)
        wavelet_bases = (1.0 - t.pow(2)) * torch.exp(-0.5 * t.pow(2))

        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.num_wavelets:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.num_wavelets, mode='linear', align_corners=True
                ).squeeze(1)
            wavelet_bases = wavelet_bases * g.unsqueeze(-2)

        base_out = torch.einsum("bi,io->bo", F.silu(x_flat), self.base_weight)
        wavelet_out = torch.einsum("bik,iok->bo", wavelet_bases, self.coeffs)
        out = base_out + wavelet_out + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.num_wavelets


class LegendreKANLayer(nn.Module):

    def __init__(self, in_features, out_features, degree=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.legendre_coeffs = nn.Parameter(
            torch.empty(in_features, out_features, degree + 1)
        )
        nn.init.normal_(self.legendre_coeffs, mean=0.0,
                        std=1.0 / math.sqrt(in_features * (degree + 1)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_norm = torch.tanh(x_flat)

        legendre = torch.ones(
            x_flat.shape[0], self.in_features, self.degree + 1,
            device=x.device, dtype=x.dtype
        )
        if self.degree > 0:
            legendre[:, :, 1] = x_norm
        for n in range(1, self.degree):
            legendre[:, :, n + 1] = (
                (2 * n + 1) * x_norm * legendre[:, :, n].clone()
                - n * legendre[:, :, n - 1].clone()
            ) / (n + 1)

        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.degree + 1:
                g = F.interpolate(
                    g.unsqueeze(1), size=self.degree + 1, mode='linear', align_corners=True
                ).squeeze(1)
            legendre = legendre * g.unsqueeze(-2)

        out = torch.einsum("bid,iod->bo", legendre, self.legendre_coeffs) + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.degree + 1


class CausalHermiteKANLayer(nn.Module):

    def __init__(self, in_features, out_features, order=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order

        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, order) * 0.01
        )
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.log_sigma = nn.Parameter(torch.zeros(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nf = torch.ones(order)
        for n in range(1, order):
            nf[n] = nf[n - 1] * math.sqrt(n)
        self.register_buffer("norm_factors", nf)

    def forward(self, x, gates=None):
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        sigma = torch.exp(self.log_sigma).clamp(min=0.1, max=10.0)
        z = x_flat / sigma.unsqueeze(0)

        hermite = torch.ones(
            x_flat.shape[0], self.in_features, self.order,
            device=x.device, dtype=x.dtype
        )
        if self.order > 1:
            hermite[:, :, 1] = z
        for n in range(2, self.order):
            hermite[:, :, n] = (
                z * hermite[:, :, n - 1].clone()
                - (n - 1) * hermite[:, :, n - 2].clone()
            )

        hermite = hermite / self.norm_factors

        if gates is not None:
            g = gates
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] != self.order:
                g = g[..., :self.order]
            hermite = hermite * g.unsqueeze(-2)

        hermite_out = torch.einsum("bik,oik->bo", hermite, self.coeffs)
        base_out = torch.einsum("bi,io->bo", F.silu(x_flat), self.base_weight)

        out = base_out + hermite_out + self.bias
        return out.reshape(*batch_dims, self.out_features)

    def basis_size(self):
        return self.order


def build_kan_layer(kan_type, in_features, out_features, config):
    if kan_type == "taylor":
        return TaylorKANLayer(in_features, out_features, order=config.kan_order)
    elif kan_type == "bspline":
        return BSplineKANLayer(
            in_features, out_features,
            num_knots=config.kan_order, spline_order=config.bspline_order
        )
    elif kan_type == "fourier":
        return FourierKANLayer(
            in_features, out_features,
            num_frequencies=config.kan_order, omega=config.fourier_omega
        )
    elif kan_type == "chebykan":
        return ChebyKANLayer(
            in_features, out_features, degree=config.kan_order
        )
    elif kan_type == "jacobikan":
        return JacobiKANLayer(
            in_features, out_features, degree=config.kan_order,
            a=config.jacobi_a, b=config.jacobi_b
        )
    elif kan_type == "rbfkan":
        return RBFKANLayer(
            in_features, out_features, num_centers=config.kan_order
        )
    elif kan_type == "hermitekan":
        return CausalHermiteKANLayer(
            in_features, out_features, order=config.kan_order
        )
    elif kan_type == "waveletkan":
        return WaveletKANLayer(
            in_features, out_features, num_wavelets=config.kan_order
        )
    elif kan_type == "legendrekan":
        return LegendreKANLayer(
            in_features, out_features, degree=config.kan_order
        )
    else:
        raise ValueError(f"Unknown KAN type: {kan_type}")
