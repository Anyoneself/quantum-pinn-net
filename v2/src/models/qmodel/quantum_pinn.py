from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    """Sinusoidal layer used by the quantum-inspired feature encoder."""

    def __init__(self, in_dim: int, out_dim: int, omega: float = 18.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class QuantumFeaturePINN(nn.Module):
    """Quantum-inspired PINN for permeability-field prediction.

    This model is not a hardware quantum circuit. It uses angle encoding,
    trigonometric phase features, a learnable entanglement mixer, and gated
    residual layers to improve PINN expressivity under sparse observations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        sample_count: int,
        latent_dim: int,
        qubits: int = 6,
    ) -> None:
        super().__init__()
        self.qubits = qubits
        self.sample_embedding = nn.Embedding(sample_count, latent_dim)
        self.angle_encoder = nn.Linear(input_dim, qubits)
        self.quantum_mixer = nn.Parameter(torch.randn(qubits, qubits) * 0.08)

        feature_dim = input_dim + qubits * 4
        self.first_layer = SineLayer(feature_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.angle_encoder(x)
        entangled = torch.sin(angles @ self.quantum_mixer)
        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
                torch.sin(angles + entangled),
                torch.cos(angles - entangled),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x, self.quantum_features(x)], dim=-1)
        hidden = torch.sin(self.gate(features)) * self.first_layer(features)
        for layer in self.hidden_layers:
            hidden = F.silu(layer(hidden)) + hidden
        return self.output(hidden)

    def reset_low_utility_neurons(self, utility: torch.Tensor, threshold: float) -> int:
        low = torch.nonzero(utility < threshold, as_tuple=False).flatten()
        if low.numel() == 0:
            return 0

        with torch.no_grad():
            for idx in low.tolist():
                nn.init.normal_(self.first_layer.linear.weight[idx : idx + 1], mean=0.0, std=0.12)
                self.first_layer.linear.bias[idx].zero_()
        return int(low.numel())


class QuantumLossBalancer:
    """Quantum-inspired probability-amplitude loss weighting.

    Loss components are mapped to a normalized amplitude vector. Squared
    amplitudes act as dynamic weights for data, physics, smoothness, and
    boundary losses.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.amplitude = torch.ones(len(names), dtype=torch.float32) / math.sqrt(len(names))
        self.previous: torch.Tensor | None = None
        self.history: list[dict[str, float]] = []

    def weights(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(iter(losses.values())).device
        amplitude = self.amplitude.to(device)
        probabilities = amplitude.square()
        return {name: probabilities[idx] * len(self.names) for idx, name in enumerate(self.names)}

    def update(self, losses: dict[str, float]) -> dict[str, float]:
        current = torch.tensor([losses[name] for name in self.names], dtype=torch.float32)
        if self.previous is None:
            score = 1.0 / (current + 1e-8)
        else:
            score = torch.clamp((self.previous - current) / (self.previous.abs() + 1e-8), min=-0.5, max=0.5)
            score = score + 1.0 / (current + 1e-8)

        phase = torch.softmax(score, dim=0)
        self.amplitude = torch.sqrt(0.70 * self.amplitude.square() + 0.30 * phase)
        self.amplitude = self.amplitude / torch.linalg.norm(self.amplitude)
        self.previous = current

        probabilities = self.amplitude.square()
        weights = {name: float(probabilities[idx] * len(self.names)) for idx, name in enumerate(self.names)}
        self.history.append(weights)
        return weights
