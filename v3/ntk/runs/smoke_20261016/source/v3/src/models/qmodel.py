"""V3 量子模型：精确模拟特征线路，共享 Darcy 网络其余部分。"""

import pennylane as qml
import torch
from torch import nn

from .darcy_pinn import DarcyPINN


class QuantumFeatures(nn.Module):
    """精确态矢量模拟，不使用有限采样或真实量子硬件。"""

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


class QDarcyPINN(DarcyPINN):
    """构造量子特征的 Darcy 模型。"""

    def __init__(self, config, seed):
        super().__init__(config, "quantum", seed, QuantumFeatures)
