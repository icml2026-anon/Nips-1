import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class NonlinearSEM(nn.Module):

    def __init__(self, T, hidden_dim=64):
        super().__init__()
        self.T = T
        self.encoder_fc1 = nn.Linear(1, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, 1)
        self.decoder_fc1 = nn.Linear(1, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x.unsqueeze(-1)))
        return self.encoder_fc2(h).squeeze(-1)

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z.unsqueeze(-1)))
        return self.decoder_fc2(h).squeeze(-1)


class ZeroInflatedHead(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.linear = nn.Linear(T, T)

    def forward(self, f_x):
        return torch.sigmoid(self.linear(f_x))


class CausalDiscovery(nn.Module):

    def __init__(self, T, hidden_dim=64, sigma_init=1.0):
        super().__init__()
        self.T = T
        self.A_raw = nn.Parameter(torch.randn(T, T) * 0.01)
        self.sem = NonlinearSEM(T, hidden_dim)
        self.zi_head = ZeroInflatedHead(T)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
        self.log_sigma_z = nn.Parameter(torch.tensor(0.0))
        temporal_mask = torch.triu(torch.ones(T, T), diagonal=1)
        self.register_buffer("temporal_mask", temporal_mask)

    def get_adjacency(self):
        return self.A_raw * self.temporal_mask

    def dag_constraint(self, A):
        M = A * A
        expm = torch.matrix_exp(M)
        return torch.trace(expm) - self.T

    def forward(self, X):
        A = self.get_adjacency()
        f_X = self.sem.encode(X)
        I_mat = torch.eye(self.T, device=X.device)
        I_minus_A = I_mat - A
        Z_mean = torch.matmul(f_X, I_minus_A)
        sigma_z = torch.exp(self.log_sigma_z).clamp(min=1e-4)
        Z = Z_mean + torch.randn_like(Z_mean) * sigma_z
        I_minus_A_inv = torch.inverse(I_minus_A)
        f_X_hat = torch.matmul(Z, I_minus_A_inv)
        X_hat = self.sem.decode(f_X_hat)
        pi_hat = self.zi_head(f_X)
        return X_hat, pi_hat, Z_mean, sigma_z, A

    def zero_inflated_log_likelihood(self, X, X_hat, pi_hat):
        sigma = torch.exp(self.log_sigma).clamp(min=1e-4)
        mask = (X > 0).float()
        var = sigma ** 2
        log_norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma)
        gaussian_ll = log_norm - 0.5 * (X - X_hat) ** 2 / var
        gaussian_at_zero = log_norm - 0.5 * X_hat ** 2 / var
        pi_clamped = pi_hat.clamp(1e-7, 1.0 - 1e-7)
        ll_zero = torch.log(pi_clamped + (1.0 - pi_clamped) * torch.exp(gaussian_at_zero))
        ll_nonzero = torch.log(1.0 - pi_clamped) + gaussian_ll
        ll = mask * ll_nonzero + (1.0 - mask) * ll_zero
        return ll.sum(dim=-1).mean()

    def compute_kl(self, Z_mean, sigma_z):
        var_z = sigma_z ** 2
        kl = -0.5 * (1 + torch.log(var_z) - Z_mean.pow(2) - var_z)
        return kl.sum(dim=-1).mean()

    def compute_loss(self, X, lambda_s=1.0, rho=1.0, alpha=0.0):
        X_hat, pi_hat, Z_mean, sigma_z, A = self.forward(X)
        recon_ll = self.zero_inflated_log_likelihood(X, X_hat, pi_hat)
        kl = self.compute_kl(Z_mean, sigma_z)
        elbo = recon_ll - kl
        h = self.dag_constraint(A)
        l1 = torch.norm(A, p=1)
        loss = -elbo + lambda_s * l1 + 0.5 * rho * h ** 2 + alpha * h
        return loss, h, A


def train_causal_discovery(model, X, config, device):
    model = model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rho = config.dag_rho_init
    alpha_val = config.dag_alpha_init
    h_prev = float("inf")

    for outer in range(config.dag_outer_iter):
        for inner in range(config.dag_inner_iter):
            n = X_tensor.shape[0]
            idx = torch.randint(0, n, (min(config.batch_size, n),), device=device)
            batch = X_tensor[idx]
            optimizer.zero_grad()
            loss, h, A = model.compute_loss(
                batch, lambda_s=config.sparsity_lambda, rho=rho, alpha=alpha_val
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        with torch.no_grad():
            A = model.get_adjacency()
            h_val = model.dag_constraint(A).item()
            alpha_val = alpha_val + rho * h_val
            if abs(h_val) > 0.25 * abs(h_prev):
                rho = min(rho * 10, 1e16)
            h_prev = h_val

    with torch.no_grad():
        A_final = model.get_adjacency()
        mask = (A_final.abs() > config.dag_threshold).float()
        A_final = A_final * mask

    return A_final.detach()
