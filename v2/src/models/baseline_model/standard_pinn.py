from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.linear1(self.norm(x)))
        return x + self.linear2(hidden)


class StandardPINN(nn.Module):
    """Strong classical PINN baseline for sparse permeability-field prediction.

    The network receives coordinate features, normalized physical conditions,
    and a learnable per-field sample embedding. The sample embedding allows the
    same network to predict held-out points in multiple GSTools permeability
    fields instead of collapsing to a cross-sample mean field.

    This baseline intentionally uses strong classical ingredients: fixed
    Fourier coordinate features and residual MLP blocks. It still uses fixed
    PINN loss weights and has no quantum feature layer, adaptive residual
    weighting, or Monte Carlo neuron screening.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        sample_count: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.sample_embedding = nn.Embedding(sample_count, latent_dim)
        self.register_buffer("fourier_frequencies", torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32))

        feature_dim = input_dim + 2 * 2 * int(self.fourier_frequencies.numel())
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[:, :2]
        angles = coords[:, :, None] * self.fourier_frequencies[None, None, :] * torch.pi
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1).flatten(start_dim=1)
        return torch.cat([x, encoded], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(self.fourier_features(x))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)
