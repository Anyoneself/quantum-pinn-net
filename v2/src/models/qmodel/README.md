# Quantum-PINN

本目录对应量子启发 PINN 方法。网络实现位于 `quantum_pinn.py` 中的 `QuantumFeaturePINN` 和 `QuantumLossBalancer`；训练入口 `v2/src/experiment/run_pinn_experiment.py` 会动态加载该模型文件，并在训练阶段执行 Monte Carlo neuron screening。

方法由三部分组成：

- Learnable sample embedding: 与 Baseline 一样，为每个 GSTools 渗透率场学习样本标识，用于在同一场内部预测未观测空间点。
- Quantum-inspired angle encoding: 通过可学习角度编码和 `sin/cos` 相位特征增强空间场表达。
- Dynamic loss weighting: 将各 loss 分量映射为归一化概率幅，按训练过程动态调整权重。
- Monte Carlo neuron screening: 用随机批次估计神经元效用，对低效量子特征神经元进行轻量重初始化。

该实现属于量子启发优化算法，不声称在真实量子硬件上运行。实验结果表明，在当前 GSTools 渗透率场空间留出预测任务中，该方法相较标准 PINN 具有更好的误差指标、结构相似性和综合评分。
