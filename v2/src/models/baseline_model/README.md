# Standard PINN Baseline

本目录对应标准 PINN 基线方法。网络实现位于 `standard_pinn.py` 中的 `StandardPINN`，训练入口 `v2/src/experiment/run_pinn_experiment.py` 会动态加载该模型文件。

网络输入为二维坐标、五维物理条件和可学习样本 embedding，输出对应空间点的归一化 log-permeability。样本 embedding 使 Baseline 能够在每个 GSTools 渗透率场内部根据稀疏观测点预测未观测位置。

为了避免弱 Baseline 造成不公平对比，当前 Baseline 已升级为强经典 PINN：

- 固定 Fourier 坐标特征，用于增强空间高频表达；
- SiLU 残差 MLP 块，用于提高拟合能力和训练稳定性；
- LayerNorm 稳定深层特征尺度；
- 不使用量子特征、动态损失权重、残差自适应采样或 Monte Carlo 神经元筛选。

训练仍使用固定权重的 PINN loss：

- data loss
- physics mean residual
- smoothness loss
- boundary consistency loss

该模型作为实验 Baseline，用于衡量不引入量子启发特征和动态优化策略时的渗透率场预测性能。
