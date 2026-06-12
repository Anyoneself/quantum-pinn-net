from __future__ import annotations

import torch
import torch.nn as nn


class StandardPINN(nn.Module):
    """Baseline PINN for sparse permeability-field prediction.

    The network receives coordinate features, normalized physical conditions,
    and a learnable per-field sample embedding. The sample embedding allows the
    same network to predict held-out points in multiple GSTools permeability
    fields instead of collapsing to a cross-sample mean field.
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

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
