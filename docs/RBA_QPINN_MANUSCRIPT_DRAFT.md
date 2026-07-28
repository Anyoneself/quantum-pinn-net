# 融合残差自适应权重与量子门等价特征的物理信息神经网络用于非均质渗透率场预测

## 摘要

针对物理信息神经网络（physics-informed neural network, PINN）在非均质渗透率场预测中容易出现高梯度区域拟合不足、物理约束与数据拟合难以平衡以及局部残差信息利用不充分等问题，本文提出一种融合残差自适应权重与量子门等价特征的物理信息神经网络（Residual-Balanced Adaptive Quantum Physics-Informed Neural Network, RBA-QPINN）。该方法以空间坐标、工程物理条件和可学习样本嵌入为输入，通过量子启发角度编码、三角相位响应和可学习纠缠混合构造量子门等价特征；在优化层面，引入动态损失权重、逐点残差自适应权重和 Monte Carlo 神经元效用阈值评价，使模型能够在训练过程中聚焦高残差区域并抑制低效神经元。受残差信息自适应 PINN 思路启发，本文进一步设计了基于高残差候选点、密度聚类和局部残差补偿的采样与评价流程。实验基于 128 个二维 GSTools 非均质渗透率场样本，每个样本采用 32 x 32 网格，并按照样本内空间留出协议，以约 80% 空间点作为观测点、约 20% 空间点作为测试点。当前已完成的 6 组对比结果表明，Quantum Feature PINN 在预测误差上取得最优表现，其测试点 RMSE 由 Baseline PINN 的 0.0288 降至 0.0175，MAE 由 0.0210 降至 0.0102，R2 由 0.9925 提升至 0.9972；含损失平衡、Monte Carlo 阈值和残差自适应模块的 RBA-QPINN 系列在物理残差约束方面表现出进一步改善趋势。本文结果说明，量子门等价特征能够显著增强 PINN 对非均质空间结构的表达能力，而残差自适应机制为高残差区域建模和物理一致性约束提供了可解释的优化路径。

**关键词**：物理信息神经网络；残差自适应权重；量子启发神经网络；渗透率场预测；自适应采样；消融实验

## 1 引言

非均质渗透率场的快速预测是地下流体模拟、智能油田开发和多孔介质流动分析中的重要问题。传统数值方法通常依赖精细网格离散和迭代求解，在复杂非均质结构、参数不确定性和多工况快速评估场景下面临较高计算成本。深度学习方法能够从已有样本中学习空间结构映射，但若缺少物理约束，模型可能在样本外区域产生不符合工程规律的预测。PINN 通过将控制方程、边界条件或物理先验嵌入损失函数，为数据驱动建模与物理规律约束之间提供了统一框架。

然而，在非均质渗透率场这类空间变化明显、局部梯度较高的问题中，标准 PINN 仍存在三方面局限。首先，常规 MLP 或固定频率特征对高频空间结构的表达能力有限，容易生成过度平滑的预测场。其次，数据损失、物理均值约束、平滑约束与边界约束之间的尺度差异会影响优化稳定性，固定损失权重难以适配训练阶段变化。再次，随机采样或固定掩码无法区分易拟合区域与难拟合区域，模型训练资源可能被均匀分配到物理变化较弱的区域，而高残差区域没有获得充分关注。

针对上述问题，已有研究分别从损失权重自适应和配置点自适应采样两个方向改进 PINN。方荣等提出的 RBA-PINN 通过 DBSCAN 密度聚类从高残差候选区域提取高密度点，并结合自适应阈值搜索实现配置点动态重采样；同时采用自适应注意力掩码机制同步更新各配置点的损失权重，使模型聚焦难训练点。该研究表明，样本分布优化与损失权重动态调整具有协同作用，能够缓解 PINN 在高梯度区域拟合不足的问题。本文在这一残差自适应思想基础上，面向二维非均质渗透率场预测任务，进一步引入量子门等价特征映射和 Monte Carlo 神经元效用评价，形成 RBA-QPINN 框架。

本文的主要贡献如下：

1. 提出一种面向非均质渗透率场预测的 RBA-QPINN 网络结构，将量子启发角度编码、相位测量特征、门控残差 MLP 与 PINN 物理约束结合，提高模型对空间非均质结构的表达能力。
2. 设计两级自适应优化机制：全局动态损失权重用于平衡数据项、物理项、平滑项和边界项，逐点残差权重用于强化高残差空间位置的训练贡献。
3. 引入 Monte Carlo 单神经元效用阈值评价，通过激活强度与方差估计识别低效神经元，并在训练前期进行轻量重初始化，提高量子特征层的可用表达容量。
4. 建立面向投稿论文的实验评价体系，包括强 Baseline PINN、多模块消融、误差分布、预测-真实散点、残差采样热力图、权重动态曲线和训练时间分析。

## 2 相关工作

### 2.1 物理信息神经网络

Raissi 等提出的 PINN 将偏微分方程残差、初始条件和边界条件共同纳入神经网络优化目标，使神经网络能够在有限观测数据下学习满足物理约束的解函数。PINN 的无网格特性使其在流体力学、地下水动力学、多相流模拟和反问题求解等领域具有广泛适用性。但是，PINN 在强非线性、多尺度和高梯度问题中常出现训练收敛慢、局部区域误差大和损失项梯度失衡等现象。

### 2.2 损失权重自适应与残差自适应采样

为缓解 PINN 训练中的损失不平衡问题，SA-PINN 通过可训练权重动态放大难训练区域的损失贡献；LA-PINN 则通过损失注意力机制学习不同损失项或残差点的重要性。在采样方面，RAR、RAD 和 RAR-D 等方法利用残差信息在训练过程中新增或替换配置点，以提升高误差区域的建模精度。方荣等指出，若采样策略仅关注残差幅值，容易受孤立噪声点干扰，并可能忽略高残差点的空间聚集特征。因此，RBA-PINN 将自适应阈值搜索、DBSCAN 密度聚类和逐点自适应注意力掩码结合，使采样点和损失权重同时聚焦于连续高残差区域。

本文借鉴上述思想，但任务对象从经典 PDE 基准方程扩展到二维非均质渗透率场预测。由于本文采用的是样本内空间留出预测协议，残差信息主要来自训练观测点，而非使用测试点标签；因此，逐点残差权重与后处理补偿均限制在训练观测信息范围内，以保证评价过程的独立性。

### 2.3 量子启发神经网络与特征表达

量子启发神经网络通常利用角度编码、相位响应、测量期望或纠缠结构模拟量子线路的表达形式，在经典硬件上构建非线性特征映射。与真实量子硬件线路不同，本文采用可微的量子门等价特征层，将输入变量映射为三角相位特征，并通过可学习纠缠混合矩阵模拟量子比特间的相位耦合。该结构不依赖量子硬件，具有完全可复现的经典深度学习实现，同时保留了量子相位特征在非线性表达上的启发意义。

## 3 方法

### 3.1 问题定义

设第 `s` 个样本对应的二维归一化 log-permeability 场为：

```text
u_s: Omega -> R,  Omega = [-1, 1] x [-1, 1].
```

每个样本具有工程物理条件向量：

```text
c_s = [pressure, pressure_gradient, viscosity, porosity, flow_rate].
```

训练过程中，样本内约 80% 网格点作为观测点，记为 `Omega_train`；其余约 20% 网格点作为 held-out 测试点，记为 `Omega_test`。模型目标是在给定 `(x, y, c_s)` 与样本嵌入 `e_s` 的条件下预测 `u_s(x, y)`，并仅在 `Omega_test` 上计算泛化指标。

### 3.2 输入表示与强 Baseline PINN

对坐标点 `(x, y)`，模型输入定义为：

```text
z = concat(x, y, c_s, e_s),
```

其中 `e_s` 为可学习样本嵌入，用于区分不同 GSTools 渗透率场。为避免弱基线造成不公平对比，Baseline PINN 采用固定 Fourier 坐标特征、SiLU 残差 MLP 和 LayerNorm。该 Baseline 仍保持经典 PINN 属性，即不引入量子门等价特征、残差自适应权重或 Monte Carlo 神经元筛选。

### 3.3 量子门等价特征层

RBA-QPINN 首先将输入向量投影到 `q` 个相位角：

```text
theta = W_e z + b_e.
```

随后使用三角函数模拟单量子比特旋转响应：

```text
phi_1 = sin(theta),  phi_2 = cos(theta).
```

为模拟量子线路中的纠缠相位混合，引入可学习矩阵 `M_q`：

```text
eta = sin(theta M_q).
```

最终量子门等价特征定义为：

```text
phi_q(z) = concat(sin(theta), cos(theta), sin(theta + eta), cos(theta - eta)).
```

该特征与原始输入拼接后送入门控残差主干网络：

```text
h_0 = concat(z, phi_q(z)).
```

### 3.4 PINN 损失函数

本文采用由数据项、物理项、平滑项和边界项构成的联合损失：

```text
L = w_d L_data + w_p L_physics + w_s L_smooth + w_b L_boundary.
```

其中，`L_data` 为训练观测点 MSE；`L_physics` 约束预测场均值与 Darcy/geomechanics-inspired 物理期望均值一致；`L_smooth` 为相邻网格差分平滑项；`L_boundary` 约束上下、左右边界均值一致性。该物理项并非完整 Darcy PDE 求解，而是作为工程条件与渗透率场统计特征之间的一致性约束。

### 3.5 全局动态损失权重

为避免固定权重导致部分损失项主导优化，本文采用基于概率幅的动态损失权重：

```text
a_k <- sqrt(beta a_k^2 + (1 - beta) softmax(score_k)),
w_k = K a_k^2,
```

其中 `K` 为损失项数量，`a_k` 为第 `k` 个损失项的概率幅，`w_k` 为其对应权重。`score_k` 由当前损失规模和收敛改善率共同决定：

```text
score_k = gamma / (L_k + eps) + (1 - gamma) clip((L_k_prev - L_k) / (|L_k_prev| + eps)).
```

该机制用于动态协调数据拟合、物理一致性、平滑性和边界约束之间的优化关系。

### 3.6 逐点残差自适应权重

参考 RBA-PINN 中自适应注意力掩码的思想，本文对训练观测点引入逐点残差权重。对训练点 `i`，残差为：

```text
r_i = u_hat_i - u_i.
```

加权数据损失为：

```text
L_data = sum_i lambda_i r_i^2 / (sum_i lambda_i + eps).
```

本文实现中，`lambda_i` 根据训练观测点残差的相对大小以指数滑动方式更新，并被裁剪到预设范围，避免极端权重造成训练震荡。与直接使用测试点残差不同，该权重更新只利用训练观测点信息，因此不引入测试标签泄漏。

### 3.7 残差驱动采样与局部补偿

受方荣等自适应阈值搜索与 DBSCAN 聚类采样思想启发，本文在训练过程中维护训练点残差权重图。每隔固定周期，根据训练观测点残差识别高残差候选区域，并在空间邻域内扩展残差权重，使模型更多关注连续高残差区域。该策略避免仅追逐孤立最大残差点，有助于保留非均质场的空间结构连续性。

为改善可视化和 held-out 点预测的局部细节，本文还采用基于训练观测点残差的 IDW 局部补偿。该补偿仅使用训练观测点处的 `u - u_hat`，对 held-out 点预测进行局部残差插值校正，并保持训练观测点精确值不变。评价指标仍仅在 held-out 点上计算。

### 3.8 Monte Carlo 神经元效用阈值

为识别量子特征层中的低效神经元，本文定义第 `j` 个神经元的 Monte Carlo 效用：

```text
U_j = E_MC[|a_j(z)|] sqrt(Var_MC[a_j(z)] + eps).
```

在训练前半程，对随机 batch 估计 `U_j`，并将低于分位阈值的神经元进行轻量重初始化。该机制可视为量子相位表达层的结构自修复策略，用于释放长期处于弱激活状态的表达容量。

## 4 实验设计

### 4.1 数据集与训练协议

实验使用 128 个由 GSTools/物理约束流程生成的二维非均质渗透率场，每个样本尺寸为 `32 x 32`。每个样本包含压力、压力梯度、黏度、孔隙度和流量五类工程条件。训练采用样本内空间留出预测协议，即同一渗透率场中约 80% 网格点作为训练观测点，约 20% 网格点作为测试点。所有误差指标均在 held-out 测试点上计算。

### 4.2 评价指标

本文报告以下指标：

- MSE、RMSE、MAE、MAPE：衡量 held-out 点预测误差；
- Accuracy：基于 MAE 和目标值范围定义的归一化精度；
- R2：衡量预测对 held-out 点方差的解释能力；
- SSIM：衡量预测场与目标场的结构相似性；
- Physics Residual：预测场均值与物理期望均值之间的平均绝对偏差；
- Physical Score：由 Darcy 相关性、压力相关性、孔隙度相关性和范围约束构成的综合物理合理性诊断项；
- Composite Score：综合 Accuracy、R2、SSIM、Physics Residual 和 Physical Score 的总体评分。

### 4.3 对比模型与消融设置

当前已完成训练的模型包括：

1. Baseline PINN；
2. Quantum Feature PINN；
3. Quantum + Loss Balancer；
4. Quantum + Loss + MC；
5. Quantum + Loss + RBA；
6. Full RBA-QPINN。

为进一步支撑投稿论文中的模块分析，本文已设计扩展消融矩阵，包含 Classical + Loss Balancer、Classical + RBA、Classical + Loss + RBA、Quantum + MC、Quantum + RBA 等额外组合。需要说明的是，扩展消融矩阵已写入实验脚本，但新增组合的完整训练结果仍需后续运行 `--run-ablations` 后更新。

## 5 实验结果与分析

### 5.1 总体指标对比

当前已完成的 6 组实验结果如表 1 所示。

| 方法 | RMSE | MAE | R2 | SSIM | Physics Residual | Physical Score | Composite Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline PINN | 0.0288 | 0.0210 | 0.9925 | 0.9987 | 0.1327 | 0.5818 | 0.8859 |
| Quantum Feature PINN | **0.0175** | **0.0102** | **0.9972** | **0.9995** | 0.1326 | 0.5817 | **0.8881** |
| Quantum + Loss Balancer | 0.0245 | 0.0171 | 0.9945 | 0.9991 | 0.1321 | 0.5817 | 0.8869 |
| Quantum + Loss + MC | 0.0249 | 0.0177 | 0.9943 | 0.9991 | **0.1317** | **0.5818** | 0.8869 |
| Quantum + Loss + RBA | 0.0242 | 0.0168 | 0.9947 | 0.9991 | 0.1321 | 0.5816 | 0.8870 |
| Full RBA-QPINN | 0.0248 | 0.0176 | 0.9944 | 0.9990 | 0.1318 | 0.5817 | 0.8869 |

从预测误差看，Quantum Feature PINN 取得最低 RMSE 和 MAE，说明量子门等价特征层对非均质渗透率场的空间表达具有直接贡献。相较于 Baseline PINN，Quantum Feature PINN 的 RMSE 由 0.0288 降至 0.0175，下降约 39.17%；MAE 由 0.0210 降至 0.0102，下降约 51.25%。从结构指标看，该模型的 R2 和 SSIM 也最高，表明其预测结果与目标场在整体空间结构上更一致。

从物理残差看，Quantum + Loss + MC 和 Full RBA-QPINN 的 Physics Residual 较 Baseline 有一定降低，其中 Quantum + Loss + MC 当前达到最低物理残差 0.1317。这说明动态损失权重和 Monte Carlo 神经元效用机制对物理一致性约束具有一定积极作用，但其对预测误差的贡献并非单调增加。该现象提示：在当前数据规模和训练预算下，量子特征层是主要的预测误差改进来源，而损失平衡、MC 阈值和 RBA 模块更偏向改善物理约束和训练关注区域。

### 5.2 消融实验分析

消融实验表明，模块贡献具有非线性交互特征。单独引入量子门等价特征时，模型获得最明显的预测误差下降；进一步叠加动态损失平衡或 MC 阈值后，预测误差有所回升，但物理残差得到改善。这说明多目标 PINN 优化中存在预测精度与物理一致性之间的权衡，不能简单假设模块越多预测误差越低。

本文已补充完整消融设计矩阵，以区分经典网络与量子网络在不同模块组合下的差异。扩展消融的关键问题包括：Classical + RBA 是否能在不引入量子特征时改善高残差区域；Quantum + RBA 是否能在没有动态损失平衡时独立发挥作用；Quantum + MC 是否能在单独使用时改善神经元表达效率；Full RBA-QPINN 是否能在更充分训练或不同权重配置下同时兼顾预测误差与物理残差。上述问题将在完整 11 组训练后进一步定量分析。

### 5.3 可视化结果

预测场对比图显示，Baseline PINN 能够恢复主要低频结构，但在局部高梯度区域仍存在过平滑现象。Quantum Feature PINN 的预测场与目标场更接近，尤其在 held-out 点级预测图中，其点值分布与目标点更一致。预测-真实散点图进一步显示，Quantum Feature PINN 的散点更集中于 `y = x` 对角线附近，说明其 held-out 点预测偏差更小。

残差采样热力图和权重动态曲线用于解释 RBA 模块的作用机制。高残差区域被赋予更高训练权重，模型因此能够在训练过程中持续关注难拟合区域。该现象与方荣等 RBA-PINN 中“通过样本空间和权重空间共同聚焦高损失区域”的思想一致。

## 6 讨论

当前实验结果具有两点启示。第一，量子门等价特征层是当前任务中最显著的预测性能来源。由于非均质渗透率场具有明显空间频率变化，三角相位编码和可学习纠缠混合能够提供比普通坐标输入更丰富的非线性表达。第二，残差自适应与动态权重模块的作用更偏向优化过程调节和物理一致性约束，而非在当前设置下必然带来 RMSE 单调下降。因此，论文表述应避免声称 Full RBA-QPINN 在所有单项指标上均最优，而应强调不同模块对预测误差、物理残差和训练稳定性的差异化贡献。

本文仍存在若干需要完善之处。首先，扩展 11 组消融实验尚需完成全部训练，以验证单模块和双模块组合的独立贡献。其次，当前 Physics Residual 使用场均值与物理期望均值的一致性约束，尚未构造完整 Darcy PDE 残差；后续可引入压力场或流速场联立约束，以提升物理项解释力。最后，局部残差补偿虽然仅使用训练观测点信息，但作为后处理策略，应在论文中与端到端网络预测分开报告，避免混淆模型原始表达能力与观测残差校正能力。

## 7 结论

本文提出 RBA-QPINN，用于在稀疏空间观测和工程物理条件下预测二维非均质渗透率场。该方法将 PINN 物理约束、量子门等价特征映射、动态损失权重、逐点残差自适应权重和 Monte Carlo 神经元效用评价结合起来，形成面向高残差区域和高频空间结构的联合优化框架。当前实验结果表明，Quantum Feature PINN 在 held-out 点预测误差、R2、SSIM 和综合评分上优于强 Baseline PINN，验证了量子门等价特征对非均质空间表达的有效性；RBA-QPINN 相关模块则为残差区域关注和物理一致性改善提供了可解释机制。未来工作将完成扩展消融矩阵训练，并进一步引入更严格的 Darcy 方程残差和多工况泛化测试。

## 参考文献

[1] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707. DOI: 10.1016/j.jcp.2018.10.045.

[2] Fang R, Ju R, Yang F L. 基于残差自适应调整PINN方法求解偏微分方程[J/OL]. 广西师范大学学报(自然科学版). DOI: 10.16088/j.issn.1001-6600.2026040803.

[3] McClenny L D, Braga-Neto U M. Self-adaptive physics-informed neural networks[J]. Journal of Computational Physics, 2023, 474: 111722. DOI: 10.1016/j.jcp.2022.111722.

[4] Lu L, Meng X H, Mao Z P, Karniadakis G E. DeepXDE: a deep learning library for solving differential equations[J]. SIAM Review, 2021, 63(1): 208-228. DOI: 10.1137/19M1274067.

[5] Nabian M A, Gladstone R J, Meidani H. Efficient training of physics-informed neural networks via importance sampling[J]. Computer-Aided Civil and Infrastructure Engineering, 2021, 36(8): 962-977. DOI: 10.1111/mice.12685.

[6] Wu C X, Zhu M, Tan Q Y, Kartha Y, Lu L. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2023, 403: 115671. DOI: 10.1016/j.cma.2022.115671.

[7] Wang S F, Teng Y J, Perdikaris P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J]. SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081. DOI: 10.1137/20m1318043.

## 写作依据与待确认内容

### 已使用的项目内证据

- `docs/PAPER_METHOD_DESIGN.md`
- `docs/EXTENDED_ABLATION_PLAN.md`
- `figures/metrics_summary.csv`
- `figures/ablation_design_matrix.csv`
- `gen-figures/run_pinn_experiment.py`
- `models/standard-pinn/standard_pinn.py`
- `models/quantum-pinn/quantum_pinn.py`

### 已使用的参考论文信息

- 方荣、琚然、杨凤莲《基于残差自适应调整PINN方法求解偏微分方程》中的 RBA-PINN 思路：DBSCAN 高残差区域聚类、自适应阈值搜索、逐点自适应注意力掩码、NTK 解释和消融实验设计。

### 需要后续补充或确认

- 扩展 11 组消融实验的完整训练结果；
- 目标投稿期刊或会议格式；
- 作者、单位、基金、通讯作者信息；
- 是否将 IDW 残差补偿作为主方法组成部分，或仅作为后处理/可视化校正报告；
- 是否补充完整 Darcy PDE 残差、压力场或流速场约束。

