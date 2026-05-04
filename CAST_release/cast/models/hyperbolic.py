import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def lorentz_inner(u, v):
    return -u[..., 0:1] * v[..., 0:1] + (u[..., 1:] * v[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_distance(u, v):
    inner = lorentz_inner(u, v).squeeze(-1)
    return torch.acosh((-inner).clamp(min=1.0 + 1e-7))


def project_to_hyperboloid(x):
    spatial = x[..., 1:]
    t = torch.sqrt(1.0 + (spatial ** 2).sum(dim=-1, keepdim=True))
    return torch.cat([t, spatial], dim=-1)


def hyperboloid_to_poincare(p):
    return p[..., 1:] / (p[..., 0:1] + 1.0)


def exp_map(p, v, eps=1e-7):
    v_norm_sq = lorentz_inner(v, v).squeeze(-1).clamp(min=eps)
    v_norm = torch.sqrt(v_norm_sq)
    coeff_p = torch.cosh(v_norm)
    coeff_v = torch.sinh(v_norm) / v_norm.clamp(min=eps)
    return coeff_p.unsqueeze(-1) * p + coeff_v.unsqueeze(-1) * v


def project_to_tangent(p, v):
    inner = lorentz_inner(p, v).squeeze(-1)
    return v + inner.unsqueeze(-1) * p


class RiemannianSGD:

    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            euclidean_grad = p.grad.data
            lorentz_metric = torch.ones_like(euclidean_grad)
            lorentz_metric[..., 0] = -1.0
            riemannian_grad = euclidean_grad * lorentz_metric
            tangent_grad = project_to_tangent(p.data, riemannian_grad)
            direction = -self.lr * tangent_grad
            p.data = exp_map(p.data, direction)
            p.data = project_to_hyperboloid(p.data)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def compute_pagerank(A, damping=0.15):
    T = A.shape[0]
    A_abs = A.abs()
    d_in = A_abs.sum(dim=0).clamp(min=1e-8)
    P_t = A_abs / d_in.unsqueeze(0)
    P = P_t.t()
    P_hat = (1.0 - damping) * P + damping / T * torch.ones_like(P)
    pi = torch.ones(T, device=A.device) / T
    for _ in range(100):
        pi_new = P_hat.t() @ pi
        pi_new = pi_new / pi_new.sum()
        if (pi_new - pi).abs().max() < 1e-8:
            break
        pi = pi_new
    return pi


class HyperbolicEmbedding(nn.Module):

    def __init__(self, T, embedding_dim, khop=2, lambda_g=0.1, damping=0.15):
        super().__init__()
        self.T = T
        self.d = embedding_dim
        self.khop = khop
        self.lambda_g = lambda_g
        self.damping = damping
        init_spatial = torch.randn(T, embedding_dim) * 0.1
        init_t = torch.sqrt(1.0 + (init_spatial ** 2).sum(dim=-1, keepdim=True))
        self.embeddings = nn.Parameter(torch.cat([init_t, init_spatial], dim=-1))

    def get_hyperboloid_embeddings(self):
        return project_to_hyperboloid(self.embeddings)

    def get_poincare_embeddings(self):
        p = self.get_hyperboloid_embeddings()
        return hyperboloid_to_poincare(p)

    def _get_khop_neighbors(self, A):
        A_sym = (A.abs() + A.abs().t() > 0).float()
        reachable = A_sym.clone()
        A_power = A_sym.clone()
        for _ in range(self.khop - 1):
            A_power = torch.matmul(A_power, A_sym).clamp(max=1.0)
            reachable = (reachable + A_power).clamp(max=1.0)
        reachable.fill_diagonal_(0.0)
        return reachable

    def contrastive_loss(self, A):
        p = self.get_hyperboloid_embeddings()
        origin = torch.zeros(self.d + 1, device=p.device)
        origin[0] = 1.0

        positive_mask = self._get_khop_neighbors(A)
        negative_mask = 1.0 - positive_mask
        negative_mask.fill_diagonal_(0.0)

        inner_matrix = -lorentz_inner(
            p.unsqueeze(1).expand(-1, self.T, -1),
            p.unsqueeze(0).expand(self.T, -1, -1)
        ).squeeze(-1)
        dist_matrix = torch.acosh(inner_matrix.clamp(min=1.0 + 1e-7))

        weight_matrix = A.abs() + A.abs().t()
        weight_matrix = torch.where(
            positive_mask > 0, weight_matrix.clamp(min=0.1), torch.zeros_like(weight_matrix)
        )

        neg_exp = torch.exp(-dist_matrix) * negative_mask
        neg_sum = neg_exp.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        pos_exp = torch.exp(-dist_matrix) * positive_mask
        log_softmax = torch.log(pos_exp / (pos_exp + neg_sum) + 1e-8)

        has_pos = (positive_mask.sum(dim=-1) > 0)
        has_neg = (negative_mask.sum(dim=-1) > 0)
        valid = has_pos & has_neg

        if valid.sum() > 0:
            weighted_loss = -(weight_matrix * log_softmax)
            per_node = weighted_loss.sum(dim=-1)
            normalizer = weight_matrix.sum(dim=-1).clamp(min=1e-8)
            loss_con = (per_node[valid] / normalizer[valid]).mean()
        else:
            loss_con = torch.tensor(0.0, device=p.device)

        pagerank = compute_pagerank(A, self.damping)
        dist_to_origin = lorentz_distance(p, origin.unsqueeze(0).expand_as(p))
        reg = (pagerank * dist_to_origin).mean()
        return loss_con + self.lambda_g * reg

    def forward(self, A):
        return self.contrastive_loss(A)


def train_hyperbolic_embedding(model, A, config, device):
    model = model.to(device)
    A = A.to(device)
    hyp_lr = config.rsgd_lr * 10.0
    rsgd = RiemannianSGD([model.embeddings], lr=hyp_lr)
    other_params = [p for name, p in model.named_parameters() if name != "embeddings"]
    if other_params:
        adam = torch.optim.Adam(other_params, lr=config.learning_rate)
    else:
        adam = None

    n_epochs = max(config.pretrain_epochs, 200)
    for epoch in range(n_epochs):
        rsgd.zero_grad()
        if adam is not None:
            adam.zero_grad()
        loss = model(A)
        loss.backward()
        rsgd.step()
        if adam is not None:
            adam.step()

    return model
