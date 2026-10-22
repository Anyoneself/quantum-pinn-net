"""V3 经典基线：与量子模型共享 Darcy 网络，仅替换特征块。"""

import torch
from torch import nn

from .darcy_pinn import DarcyPINN


class ClassicalFeatures(nn.Module):
    """经典可训练特征层。"""

    def __init__(self, count, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(count, count) for _ in range(layers)])

    def forward(self, angles):
        for layer in self.layers:
            angles = torch.tanh(layer(angles))
        return angles


class BaselineDarcyPINN(DarcyPINN):
    """构造经典特征的 Darcy 对照模型。"""

    def __init__(self, config, seed):
        super().__init__(config, "classical", seed, ClassicalFeatures)
