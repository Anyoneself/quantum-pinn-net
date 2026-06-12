# Standard PINN Baseline

本目录对应标准 PINN 基线方法。网络实现位于 `standard_pinn.py` 中的 `StandardPINN`，训练入口 `gen-figures/run_pinn_experiment.py` 会动态加载该模型文件。

网络输入为二维坐标、五维物理条件和可学习样本 embedding，输出对应空间点的归一化 log-permeability。样本 embedding 使 Baseline 能够在每个 GSTools 渗透率场内部根据稀疏观测点预测未观测位置。训练使用固定权重的 PINN loss：

- data loss
- physics mean residual
- smoothness loss
- boundary consistency loss

该模型作为实验 Baseline，用于衡量不引入量子启发特征和动态优化策略时的渗透率场预测性能。
