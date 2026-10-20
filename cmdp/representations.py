from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    cost: Optional[torch.Tensor] = None


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class PositionalFeature(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        encoded = x.to(freqs.dtype) @ freqs.unsqueeze(0)
        return torch.cat([encoded.cos(), encoded.sin()], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: nn.Module = nn.Mish(),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = activation
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + self.residual(residual)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: nn.Module = nn.Mish(),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        block_dims = (hidden_dims[0],) + tuple(hidden_dims)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(block_dims[i], block_dims[i + 1], activation) for i in range(len(block_dims) - 1)]
        )
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        return self.fc2(x)


def get_noise_schedule(num_noises: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = np.arange(1, num_noises + 1)
    b_max = 10.0
    b_min = 0.1
    alphas = np.exp(-b_min / num_noises - 0.5 * (b_max - b_min) * (2 * t - 1) / num_noises**2)
    betas = 1 - alphas
    alphabars = np.cumprod(alphas, axis=0)
    return (
        torch.as_tensor(betas, dtype=torch.float32),
        torch.as_tensor(alphas, dtype=torch.float32),
        torch.as_tensor(alphabars, dtype=torch.float32),
    )


def _build_rff_head(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class BaseRepresentationLearner(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def compute_feature(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, batch: TransitionBatch) -> Dict[str, float]:
        raise NotImplementedError

    @torch.no_grad()
    def covariance(self, obs: torch.Tensor, action: torch.Tensor, ridge: float) -> torch.Tensor:
        phi = self.compute_feature(obs, action)
        cov = phi.T @ phi / max(phi.shape[0], 1)
        cov = cov + ridge * torch.eye(self.feature_dim, device=phi.device)
        return cov


class SpederRepresentation(BaseRepresentationLearner):
    """
    SPEDER-style factorized transition model aligned with the public
    `spectral-rl` implementation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int,
        phi_hidden_dims: tuple[int, ...] = (512, 512),
        mu_hidden_dims: tuple[int, ...] = (512, 512),
        head_hidden_dims: tuple[int, ...] = (512, 512),
        lam: float = 1024.0,
        lr: float = 5e-6,
        model_coef: float = 1.0,
        reward_coef: float = 1.0,
        cost_coef: float = 0.0,
        num_noises: int = 25,
        device: str = "cpu",
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.lam = lam
        self.model_coef = model_coef
        self.reward_coef = reward_coef
        self.cost_coef = cost_coef
        self.num_noises = num_noises

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )
        self.phi_net = ResidualMLP(obs_dim + action_dim, feature_dim, phi_hidden_dims, nn.Mish())
        self.mu_net = ResidualMLP(
            obs_dim + (128 if self.num_noises > 0 else 0),
            feature_dim,
            mu_hidden_dims,
            nn.Mish(),
        )
        self.reward_head = _build_rff_head(feature_dim, head_hidden_dims, 1)
        self.cost_head = _build_rff_head(feature_dim, head_hidden_dims, 1)

        if self.num_noises > 0:
            betas, alphas, alphabars = get_noise_schedule(self.num_noises)
            self.register_buffer("betas", betas[:, None], persistent=False)
            self.register_buffer("alphas", alphas[:, None], persistent=False)
            self.register_buffer("alphabars", alphabars[:, None], persistent=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def compute_feature(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device).float()
        action = action.to(self.device).float()
        return self.phi_net(torch.cat([obs, action], dim=-1))

    def compute_next_basis(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_obs = next_obs.to(self.device).float()
        return torch.tanh(self.mu_net(next_obs))

    def update(self, batch: TransitionBatch) -> Dict[str, float]:
        obs = batch.obs.to(self.device).float()
        action = batch.action.to(self.device).float()
        next_obs = batch.next_obs.to(self.device).float()
        reward = batch.reward.to(self.device).view(-1, 1)
        cost = None if batch.cost is None else batch.cost.to(self.device).view(-1, 1)

        z_phi = self.compute_feature(obs, action)
        batch_size = next_obs.shape[0]

        if self.num_noises > 0:
            tiled_next_obs = next_obs.unsqueeze(0).repeat(self.num_noises, 1, 1)
            t = torch.arange(0, self.num_noises, device=self.device)
            t = t.repeat_interleave(batch_size).reshape(self.num_noises, batch_size)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(tiled_next_obs)
            noisy_next_obs = alphabars.sqrt() * tiled_next_obs + (1 - alphabars).sqrt() * eps
            z_mu = self.mu_net(torch.cat([noisy_next_obs, self.mlp_t(t.unsqueeze(-1))], dim=-1))
        else:
            z_mu = self.mu_net(next_obs).unsqueeze(0)

        z_mu = torch.tanh(z_mu)
        tiled_phi = z_phi.unsqueeze(0).repeat(z_mu.shape[0], 1, 1)
        inner = torch.bmm(tiled_phi, z_mu.transpose(-1, -2))

        pos_term = torch.diagonal(inner, dim1=-2, dim2=-1)
        neg_sq = inner.pow(2)
        if batch_size > 1:
            neg_term = (
                neg_sq.sum(dim=(-2, -1)) - torch.diagonal(neg_sq, dim1=-2, dim2=-1).sum(dim=-1)
            ) / (batch_size * (batch_size - 1))
        else:
            neg_term = neg_sq.mean(dim=(-2, -1))

        model_loss = self.lam * neg_term.mean() - 2.0 * pos_term.mean()
        reward_loss = F.mse_loss(self.reward_head(z_phi), reward)
        total_loss = self.model_coef * model_loss + self.reward_coef * reward_loss

        if cost is not None and self.cost_coef > 0.0:
            cost_loss = F.mse_loss(self.cost_head(z_phi), cost)
            total_loss = total_loss + self.cost_coef * cost_loss
        else:
            cost_loss = torch.zeros((), device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pos_probs = pos_term.mean(dim=-1)
            neg_mean = (
                inner.sum(dim=(-2, -1)) - pos_term.sum(dim=-1)
            ) / max(batch_size * max(batch_size - 1, 1), 1)

        return {
            "loss/total": float(total_loss.item()),
            "loss/model": float(model_loss.item()),
            "loss/reward": float(reward_loss.item()),
            "loss/cost": float(cost_loss.item()),
            "misc/pos_score": float(pos_probs[0].item()),
            "misc/neg_score": float(neg_mean[0].item()),
            "misc/phi_norm": float(z_phi.abs().mean().item()),
            "misc/phi_std": float(z_phi.std(dim=0).mean().item()),
        }
