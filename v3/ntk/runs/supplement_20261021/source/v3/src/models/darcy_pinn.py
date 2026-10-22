"""经典与量子模型共用的 Darcy 网络结构和配置。"""

from dataclasses import dataclass
import math

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


class PermeabilityNet(nn.Module):
    def __init__(self, config, feature_block, seed):
        super().__init__()
        # 共享层使用独立随机数作用域，避免两种特征块的参数量影响初始化。
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
            self.features = feature_block(config.features, config.feature_layers)
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
    def __init__(self, config, kind, seed, feature_block):
        super().__init__()
        config.validate()
        if kind not in ("classical", "quantum"):
            raise ValueError("kind must be classical or quantum")
        self.kind = kind
        self.permeability = PermeabilityNet(config, feature_block, seed)
        self.pressure = PressureNet(config, seed)
        self.double()

    def parameter_counts(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "feature_block": sum(p.numel() for p in self.permeability.features.parameters()),
            "pressure": sum(p.numel() for p in self.pressure.parameters()),
        }
