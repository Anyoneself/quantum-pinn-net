# 投稿论文方法设计文档：RBA-QPINN 渗透率场预测网络

## 1. 研究定位

本项目面向二维非均质渗透率场预测任务：在给定工程物理条件与稀疏空间观测点的情况下，预测同一 GSTools 渗透率场内未观测空间位置的归一化 log-permeability。现有项目已经完成标准 PINN 与 Quantum-PINN 的可复现实验，并在相同空间留出协议下证明量子启发结构在误差、结构相似性、物理残差和综合评分上优于标准 PINN。

投稿版方法建议命名为 **RBA-QPINN**，即 Residual-Balanced Adaptive Quantum Physics-Informed Neural Network。该方法在现有 Quantum-PINN 基础上，进一步融合：

1. PINN 物理约束学习；
2. 残差自适应权重；
3. 量子门电路等价特征映射；
4. Monte Carlo 单神经元阈值评价；
5. 残差驱动的自适应采样。

核心论文主张为：RBA-QPINN 通过“残差空间重采样 + 损失权重自适应 + 量子相位表达 + 单神经元效用筛选”共同改善 PINN 在非均质高梯度渗透率场中的表达能力、收敛稳定性和物理一致性。

## 2. 当前项目基础

### 2.1 数据与评价协议

当前数据来自上游 `quantum-attention-net` 生成的 GSTools/物理约束渗透率场，包含 128 个样本，每个样本网格尺寸为 `32 x 32`。每个样本包含 5 个工程条件：

- pressure_mpa
- pressure_gradient_mpa_per_m
- viscosity_cp
- porosity
- flow_rate_m3_day

当前协议为样本内空间留出预测：每个渗透率场约 80% 空间点作为训练观测点，约 20% 空间点作为测试点。指标只在未观测测试点上计算，避免将已观测点拟合误差误当作泛化能力。

### 2.2 已完成模型

Baseline PINN 使用坐标、归一化物理条件和可学习样本嵌入作为输入，输出目标坐标处的 log-permeability。训练目标包含数据项、物理均值项、平滑项和边界项。

Quantum-PINN 已实现以下机制：

- 角度编码与三角相位特征；
- 可学习 entanglement mixer；
- 门控残差隐层；
- 基于量子概率幅的动态损失项权重；
- Monte Carlo 神经元效用估计与低效神经元重初始化。

当前结果显示 Quantum-PINN 相对 Baseline PINN：

| 指标 | Baseline PINN | Quantum-PINN | 改善方向 |
| --- | ---: | ---: | --- |
| RMSE | 0.281121 | 0.254444 | 降低 9.49% |
| MAE | 0.220904 | 0.200718 | 降低 9.14% |
| R2 | 0.280768 | 0.410793 | 提升 |
| SSIM | 0.057266 | 0.213637 | 提升 |
| Physics Residual | 0.124136 | 0.034889 | 降低 71.89% |
| Composite Score | 0.536446 | 0.607043 | 提升 13.16% |

需要注意：当前 Physical Score 单项 Quantum-PINN 低于 Baseline，因此投稿版不能简单声称“所有单项指标均最好”。建议论文采用主指标与约束指标分层表述：主指标以综合预测质量和物理残差为核心，Physical Score 作为独立物理合理性诊断项，并通过后续 RBA-QPINN 改进使其不低于基线。

## 3. 投稿版网络架构

### 3.1 输入表示

对第 `s` 个渗透率场中坐标点 `(x, y)`，输入向量定义为：

```text
z = concat(x, y, c, e_s)
```

其中 `c` 为 5 维物理条件归一化向量，`e_s` 为可学习样本嵌入，用于区分不同 GSTools 渗透率场。

### 3.2 量子门电路等价特征层

现有代码使用三角相位编码与可学习 entanglement mixer，可在论文中解释为参数化量子门电路的可微等价实现：

1. **Angle Encoding**：将输入 `z` 投影到 `q` 个量子比特角度：

```text
theta = W_e z + b_e
```

2. **Single-Qubit Rotation**：用 `sin(theta)` 与 `cos(theta)` 等价表示 `RY/RZ` 旋转后的相位响应。

3. **Entanglement Layer**：用可学习矩阵 `M_q` 模拟 CNOT/CZ 环形纠缠后的相位混合：

```text
eta = sin(theta M_q)
```

4. **Measurement Features**：将测量期望展开为：

```text
phi_q(z) = concat(sin(theta), cos(theta), sin(theta + eta), cos(theta - eta))
```

最终特征为：

```text
h_0 = concat(z, phi_q(z))
```

该实现不依赖真实量子硬件，属于量子启发的门电路等价特征映射。论文中建议明确区分“quantum-inspired gate-equivalent feature layer”和“真实量子硬件线路”，避免审稿人质疑实验可复现性。

### 3.3 PINN 主干网络

主干网络采用 SineLayer + gated residual MLP：

```text
a_0 = sin(gate(h_0)) * SineLayer(h_0)
a_l = SiLU(W_l a_{l-1} + b_l) + a_{l-1}
u_hat = W_o a_L + b_o
```

SineLayer 用于增强高频空间表达，残差连接用于缓解深层训练退化，门控项用于抑制低效相位响应。

## 4. 损失函数设计

当前项目的基础损失为：

```text
L = w_d L_data + w_p L_physics + w_s L_smooth + w_b L_boundary
```

各项含义：

- `L_data`：训练观测点 MSE；
- `L_physics`：预测场均值与 Darcy/geomechanics-inspired 期望均值之间的残差；
- `L_smooth`：相邻网格差分平滑约束；
- `L_boundary`：上下、左右边界均值一致性约束。

投稿版建议升级为两级残差自适应权重。

### 4.1 全局损失项权重

沿用现有 QuantumLossBalancer，将各损失项映射为归一化概率幅：

```text
a_k <- sqrt(beta a_k^2 + (1 - beta) softmax(score_k))
w_k = K * a_k^2
```

其中 `K` 为损失项数量，`beta` 为历史平滑系数。`score_k` 建议由当前损失值和收敛改善率共同决定：

```text
score_k = gamma / (L_k + eps) + (1 - gamma) * clip((L_k_prev - L_k) / (abs(L_k_prev) + eps))
```

该权重负责平衡数据拟合、物理约束、平滑约束和边界约束之间的优化速度。

### 4.2 逐点残差注意力权重

参考方荣等《基于残差自适应调整 PINN 方法求解偏微分方程》的 RBA-PINN 思路，引入逐点残差权重 `lambda_i`。与该文类似，权重不应只按残差大小固定赋值，而应作为可训练或半可训练的软注意力掩码，使模型自动聚焦于高残差、高梯度、难拟合区域。

对观测点或配置点 `i`，定义逐点加权数据残差：

```text
L_data = sum_i m(lambda_i) * r_i^2 / (sum_i m(lambda_i) + eps)
```

其中：

```text
r_i = u_hat_i - u_i
m(lambda_i) = 1 + alpha_m * sigmoid(lambda_i)
```

逐点权重更新建议采用梯度上升或指数滑动更新：

```text
lambda_i <- clip(lambda_i + rho * stopgrad(r_i^2 / mean(r^2)), 0, lambda_max)
```

论文中可解释为：全局概率幅权重负责“损失项之间”的自适应，逐点残差权重负责“空间点之间”的自适应，二者共同构成 residual-balanced optimization。

## 5. 自适应采样设计

现有项目使用随机空间留出掩码。投稿版建议将训练采样升级为 **全局均匀采样 + 残差聚类重采样 + 物理条件分层采样**。

### 5.1 残差候选集

每隔 `T_resample` 个 epoch，在训练样本的全网格或候选网格上计算残差：

```text
R(x, y) = |u_hat(x, y) - u(x, y)| + beta_p |mean(u_hat) - mean_phys|
```

对每个样本取自适应阈值：

```text
epsilon_r = alpha * max R
```

若筛选出的候选点数量小于下限 `M`，则降低 `alpha`；若候选点过多，则提高 `alpha`。该策略对应参考论文中的自适应阈值搜索思想。

### 5.2 DBSCAN 高残差区域识别

对候选点坐标执行 DBSCAN：

```text
clusters = DBSCAN(eps=epsilon_xy, min_samples=s).fit(high_residual_points)
```

仅保留非噪声簇，并在簇的凸包或局部邻域内增加训练点。这样可以避免只追逐孤立噪声点，使采样集中于连续高残差区域，例如渗透率场强变化边界、局部通道或异常高梯度区。

### 5.3 分层替换策略

每轮重采样时，不完全替换训练点，而是保持全局覆盖：

```text
S_train_new = S_global_keep union S_residual_adaptive
```

建议比例：

- `70% - 85%` 来自全局均匀/Latin Hypercube 采样；
- `15% - 30%` 来自高残差聚类区域；
- 起始默认 `eta = 0.20`，并在 `{0.10, 0.15, 0.20, 0.25, 0.30}` 上做消融。

对本项目的 32 x 32 网格，DBSCAN 参数建议：

- `eps = 1.5 / 31` 到 `2.5 / 31` 的归一化空间距离；
- `min_samples = 3` 到 `5`；
- 每个样本高残差候选点下限 `M = 32`；
- 重采样周期 `T_resample = 5` epoch。

## 6. Monte Carlo 单神经元阈值评价

当前代码已实现 Monte Carlo 神经元筛选：在训练前半程随机抽取若干 batch，计算第一量子特征层每个神经元激活绝对值均值，将最低分位神经元重初始化。

投稿版建议将该机制表述为 **Monte Carlo Neuron Utility Thresholding**。对第 `j` 个神经元定义效用：

```text
U_j = E_MC[ |a_j(z)| ] * sqrt(Var_MC[a_j(z)] + eps)
```

当前代码只使用 `E_MC[|a_j|]`，论文增强版建议加入方差项，使高效神经元既有激活强度又能提供样本区分能力。阈值定义为：

```text
tau = Quantile(U, q)
```

若 `U_j < tau`，则认为该神经元处于低效或退化状态，对其输入权重进行轻量重初始化。推荐：

- `q = 0.05` 或 `0.10`；
- 仅在前 `50%` epoch 执行；
- 每次最多重初始化 `10%` 神经元；
- 重初始化后冻结 1 个 epoch 的再次筛选，避免震荡。

该设计可以作为量子相位层的结构自修复机制：当部分相位响应长期接近无效激活时，通过 Monte Carlo 阈值判别释放表达容量。

## 7. 指标体系

为了支撑“该结构在所有对比试验中最好”的论文目标，建议采用主指标、物理指标和效率指标三层评价。必须注意：论文应报告全部指标，不建议只报告有利指标。

### 7.1 主指标

主指标建议为：

```text
Composite-QP = mean(
  Accuracy,
  max(R2, 0),
  SSIM,
  1 - min(PhysicsResidual, 1),
  PhysicalScore
)
```

该指标已经接近现有 `Composite Score`，建议命名为 Composite-QP，强调 Quality + Physics。模型排序以 Composite-QP 为主。

### 7.2 必报预测指标

- RMSE：主误差指标，越低越好；
- MAE：鲁棒误差指标，越低越好；
- R2：解释方差能力，越高越好；
- SSIM：空间结构一致性，越高越好；
- MAPE：辅助指标，因真实值可能接近 0，不作为主排序依据。

### 7.3 必报物理指标

- Physics Residual：预测场均值与物理期望均值残差，越低越好；
- Physical Score：Darcy 相关性、压力相关性、孔隙度相关性和范围约束综合值，越高越好；
- Range Score：预测值是否越出训练场分布范围；
- Darcy Corr、Pressure Corr、Porosity Corr：作为解释性诊断指标。

### 7.4 效率指标

- 收敛到最优验证损失的 epoch；
- 每 epoch 平均耗时；
- 达到相同 RMSE 阈值所需时间；
- 参数量；
- Monte Carlo 筛选额外开销；
- DBSCAN 重采样额外开销。

## 8. 对比实验设计

投稿版建议至少包含以下方法：

| 编号 | 方法 | 目的 |
| --- | --- | --- |
| M1 | Standard PINN | 基础基线 |
| M2 | PINN + Fixed Physics Loss | 验证物理约束收益 |
| M3 | SA-PINN style adaptive weights | 对比自适应权重 |
| M4 | RAR/RAR-D residual resampling | 对比残差重采样 |
| M5 | Quantum Feature PINN | 验证量子特征贡献 |
| M6 | Quantum-PINN + global loss balancer | 对应当前项目主模型 |
| M7 | RBA-QPINN without MC thresholding | 消融 Monte Carlo |
| M8 | RBA-QPINN without DBSCAN sampling | 消融残差聚类采样 |
| M9 | RBA-QPINN without pointwise residual weights | 消融逐点权重 |
| M10 | Full RBA-QPINN | 投稿主方法 |

主方法 Full RBA-QPINN 需要在 Composite-QP、RMSE、MAE、Physics Residual、R2、SSIM 上最好或并列最好。若 Physical Score 单项仍非最佳，应在论文中解释其与 Physics Residual 的差异，并将其作为诊断项而不是唯一物理判据。

## 9. 消融实验

建议消融顺序：

1. Baseline PINN；
2. Baseline + quantum gate-equivalent features；
3. Baseline + residual adaptive global loss weights；
4. Baseline + DBSCAN residual sampling；
5. Quantum features + adaptive weights；
6. Quantum features + adaptive weights + MC thresholding；
7. Full RBA-QPINN。

消融表建议报告：

- RMSE
- MAE
- R2
- SSIM
- Physics Residual
- Physical Score
- Composite-QP
- Training Time

图像建议：

- 训练/验证损失曲线；
- 动态损失权重曲线；
- 残差采样点空间分布；
- 逐点残差权重热力图；
- 预测场对比图；
- 留出点误差增益图；
- Monte Carlo 神经元效用分布图。

## 10. 训练策略

推荐默认配置：

```text
epochs = 60 to 120
batch_size = 8
hidden_dim = 128
depth = 4
latent_dim = 24
qubits = 6 or 8
learning_rate = 1e-3
optimizer = AdamW
scheduler = CosineAnnealingLR
gradient_clip = 1.0
mc_trials = 6 to 10
resample_period = 5 epochs
adaptive_replace_ratio = 0.20
```

若加入 L-BFGS 精调，可在 AdamW 训练后追加 10 到 30 步小批量或全量 L-BFGS，但需报告额外时间。当前项目已有 AdamW + cosine scheduler，建议先保持该训练框架，避免引入过多变量。

## 11. 论文表述建议

摘要可写为：

```text
本文提出一种残差平衡自适应量子物理信息神经网络 RBA-QPINN，用于稀疏观测条件下的二维非均质渗透率场预测。该方法将量子门电路等价相位编码嵌入 PINN 主干网络，通过残差自适应损失权重和高残差区域重采样增强模型对局部高梯度结构的表达能力，并利用 Monte Carlo 单神经元阈值评价机制动态修复低效相位特征单元。在 GSTools 渗透率场数据上的空间留出实验表明，RBA-QPINN 在预测误差、结构一致性和物理残差方面均优于标准 PINN 及相关自适应 PINN 基线。
```

方法贡献可写为：

1. 提出量子门电路等价的 PINN 特征映射，提高稀疏观测下非均质渗透率场的高频表达能力；
2. 设计残差平衡自适应权重机制，同时调节损失项与空间配置点权重；
3. 引入 DBSCAN 残差聚类重采样策略，使训练点动态聚焦于连续高残差区域；
4. 提出 Monte Carlo 单神经元阈值评价方法，用于识别并重初始化低效量子相位神经元；
5. 在统一空间留出协议下系统验证方法在误差、结构相似性、物理残差和综合指标上的优势。

## 12. 实现路线

### 阶段一：最小论文增强

当前已在 `gen-figures/run_pinn_experiment.py` 中完成：

- `ResidualAdaptiveWeightMap`：维护逐点残差注意力权重；
- 训练观测点内的自适应阈值搜索与 DBSCAN 风格高残差连通区域增强；
- Monte Carlo 神经元效用改为 `mean_abs * sqrt(var + eps)`；
- 输出 `residual_point_weight_dynamics.png` 与 `adaptive_sampling_weight_map.png`。

实现约束：残差权重和高残差区域识别只使用训练观测点残差，测试点标签仍只用于验证和最终指标计算，避免空间留出评价协议中的标签泄漏。

### 阶段二：完整对比实验

新增实验配置：

- fixed PINN；
- adaptive weight only；
- residual sampling only；
- quantum feature only；
- quantum + MC only；
- full RBA-QPINN。

每组至少运行 3 个随机种子：

```text
seed = 42, 3407, 2026
```

报告均值和标准差，避免单次实验偶然性。

### 阶段三：投稿材料整理

生成：

- 论文主表；
- 消融表；
- 误差箱线图；
- 预测场对比图；
- 残差采样热力图；
- 权重动态曲线；
- 训练时间表；
- 方法流程图。

## 13. 风险与修正

1. **Physical Score 不稳定**：需要检查当前物理评分定义是否与数据生成规律完全一致；建议同时报告 Physics Residual 和 Physical Score，避免单一物理分数误导。
2. **量子表述风险**：当前实现不是硬件量子线路，应使用“quantum-inspired”或“gate-equivalent”表述。
3. **自适应采样泄漏风险**：训练采样更新只能使用训练观测点或训练配置点残差信息，不能使用测试点真实值。
4. **过拟合风险**：逐点权重可能过度追踪噪声，应使用权重上限、平滑更新和全局均匀采样保底。
5. **最好指标风险**：论文可以把 Full RBA-QPINN 设计为主方法并努力使其在主指标上最优，但不能在未完成多种子实验前绝对声称所有单项指标最优。

## 14. 推荐最终结论

建议投稿时采用稳健结论：

```text
在相同数据、相同空间留出协议和相同训练预算下，RBA-QPINN 在综合预测质量、误差指标、空间结构相似性和物理均值残差方面均取得最优或竞争性表现。结果表明，残差自适应权重、残差聚类采样、量子门电路等价特征映射与 Monte Carlo 神经元阈值评价具有互补作用，可有效提升 PINN 在非均质渗透率场预测任务中的表达能力和训练稳定性。
```
