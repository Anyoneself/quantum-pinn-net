"""Interchangeable classical/quantum feature blocks, identical surrounding PINNs."""

from dataclasses import dataclass
import math

import pennylane as qml
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    features: int = 4
    feature_layers: int = 2
    hidden: int = 24
    pressure_depth: int = 2

    def validate(self):
        if not 2 <= self.features <= 8:
            raise ValueError("features/qubits must be between 2 and 8 for this simulator protocol")
        if min(self.feature_layers, self.hidden, self.pressure_depth) < 1:
            raise ValueError("network dimensions must be positive")


class ClassicalFeatures(nn.Module):
    def __init__(self, count, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(count, count) for _ in range(layers)])

    def forward(self, angles):
        for layer in self.layers:
            angles = torch.tanh(layer(angles))
        return angles


class QuantumFeatures(nn.Module):
    """Exact statevector simulation. No finite shots or hardware speedup claims."""

    def __init__(self, count, layers):
        super().__init__()
        self.rotations = nn.Parameter(0.2 * torch.randn(layers, count, 3, dtype=torch.float64))
        self.forward_calls = 0
        self.sample_evaluations = 0
        device = qml.device("default.qubit", wires=count, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles, rotations):
            for wire in range(count):
                qml.RY(angles[..., wire], wires=wire)
            for layer in range(layers):
                for wire in range(count):
                    qml.Rot(*rotations[layer, wire], wires=wire)
                for wire in range(count - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if count > 2:
                    qml.CNOT(wires=[count - 1, 0])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(count))

        self.circuit = circuit

    def forward(self, angles):
        self.forward_calls += 1
        self.sample_evaluations += len(angles)
        return torch.stack(self.circuit(angles, self.rotations), dim=-1)


class PermeabilityNet(nn.Module):
    def __init__(self, config, kind, seed):
        super().__init__()
        # Common layers use independent RNG scopes, unaffected by block parameter counts.
        with torch.random.fork_rng():
            torch.manual_seed(seed + 11)
            self.encoder = nn.Sequential(nn.Linear(2, config.hidden), nn.Tanh(),
                                         nn.Linear(config.hidden, config.features))
            self.readout = nn.Sequential(nn.Linear(config.features, config.hidden), nn.Tanh(),
                                         nn.Linear(config.hidden, 1))
            nn.init.normal_(self.readout[-1].weight, std=0.05)
            nn.init.zeros_(self.readout[-1].bias)
        with torch.random.fork_rng():
            torch.manual_seed(seed + 23)
            self.features = (QuantumFeatures(config.features, config.feature_layers)
                             if kind == "quantum"
                             else ClassicalFeatures(config.features, config.feature_layers))
        self.register_buffer("feature_mask", torch.ones(config.features, dtype=torch.float64))

    def forward(self, xy, mask=None):
        angles = math.pi * torch.tanh(self.encoder(2 * xy - 1))
        features = self.features(angles)
        return self.readout(features * (self.feature_mask if mask is None else mask)).squeeze(-1)


class PressureNet(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed + 37)
            layers = [nn.Linear(2, config.hidden), nn.Tanh()]
            for _ in range(config.pressure_depth - 1):
                layers.extend([nn.Linear(config.hidden, config.hidden), nn.Tanh()])
            layers.append(nn.Linear(config.hidden, 1))
            self.network = nn.Sequential(*layers)

    def forward(self, xy):
        x = xy[:, 0]
        correction = self.network(2 * xy - 1).squeeze(-1)
        return 1 - x + x * (1 - x) * correction


class DarcyPINN(nn.Module):
    def __init__(self, config, kind, seed):
        super().__init__()
        config.validate()
        if kind not in ("classical", "quantum"):
            raise ValueError("kind must be classical or quantum")
        self.kind = kind
        self.permeability = PermeabilityNet(config, kind, seed)
        self.pressure = PressureNet(config, seed)
        self.double()

    def parameter_counts(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "feature_block": sum(p.numel() for p in self.permeability.features.parameters()),
            "pressure": sum(p.numel() for p in self.pressure.parameters()),
        }
