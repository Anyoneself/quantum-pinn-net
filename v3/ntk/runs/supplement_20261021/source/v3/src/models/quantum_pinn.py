"""兼容旧模型导入入口；新代码直接使用两个独立模型文件。"""

from .baseline_model import ClassicalFeatures
from .darcy_pinn import DarcyPINN as _DarcyPINN, ModelConfig
from .qmodel import QuantumFeatures


class DarcyPINN(_DarcyPINN):
    """保留旧的 kind 参数构造接口，便于已有测试和调用继续使用。"""

    def __init__(self, config, kind, seed):
        feature_block = QuantumFeatures if kind == "quantum" else ClassicalFeatures
        super().__init__(config, kind, seed, feature_block)
