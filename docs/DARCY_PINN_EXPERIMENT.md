# 渗透率场 Darcy-PINN 物理约束实验补充

## 1. 当前任务

当前数据为 `32 x 32` 的二维归一化 log-permeability 模拟场。模型输入为坐标、
5 个样本级工程条件和样本嵌入，输出为未观测位置的 log-permeability。每个场
约 80% 网格点用于训练，约 20% 网格点用于空间留出测试。

原实验的物理项只约束预测场均值与工程条件推导的期望均值。该项能表达全局
Darcy 趋势，但不能保证局部质量守恒。本补充实验在保留原约束的同时加入稳态
Darcy 偏微分方程残差。

## 2. 控制方程

假设流体为单相、不可压缩，忽略重力和源汇项：

\[
\mathbf q=-\frac{k}{\mu}\nabla p ,
\]

\[
\nabla\cdot\mathbf q=0 ,
\]

即

\[
\nabla\cdot\left(\frac{k}{\mu}\nabla p\right)=0 .
\]

当前黏度是样本内常数，在无量纲压力问题中只改变通量尺度，不改变压力方程，
因此辅助求解采用相对渗透率 \(k_r\)：

\[
k_r=\exp\left[
\operatorname{clip}
\left(\widehat{\ell}-\langle\widehat{\ell}\rangle,-6,6\right)
\right].
\]

减去场均值可使 PDE 项主要约束空间结构；绝对均值仍由现有工程条件物理项
约束。

## 3. 边界条件与离散

单位正方形上的辅助边界条件为

\[
p(0,y)=1,\quad p(1,y)=0,
\]

\[
\frac{\partial p}{\partial n}(x,0)=
\frac{\partial p}{\partial n}(x,1)=0.
\]

相邻单元面渗透率采用调和平均

\[
k_{i+1/2,j}=\frac{2k_{i,j}k_{i+1,j}}
{k_{i,j}+k_{i+1,j}+\varepsilon},
\]

再由有限体积形式计算面通量和散度。压力场通过 16 次可微
weighted-Jacobi 迭代近似求解，使 PDE 损失能够对预测渗透率反向传播。

## 4. PINN 总损失

\[
\mathcal L =
w_d\mathcal L_{\mathrm{data}}
+w_p\mathcal L_{\mathrm{physics}}
+w_s\mathcal L_{\mathrm{smooth}}
+w_b\mathcal L_{\mathrm{boundary}},
\]

其中

\[
\mathcal L_{\mathrm{physics}}
=\mathcal L_{\mathrm{mean}}
+\lambda_c\mathcal L_{\mathrm{continuity}}
+\lambda_f\mathcal L_{\mathrm{flux}},
\]

\[
\mathcal L_{\mathrm{continuity}}
=\frac{1}{N_c}\sum_i
\left(
\frac{\nabla\cdot\mathbf q_i}{q_{\mathrm{scale}}+\varepsilon}
\right)^2,
\]

\[
\mathcal L_{\mathrm{flux}}
=\frac{1}{N_s}\sum_s
\left(
\frac{Q_{\mathrm{in},s}-Q_{\mathrm{out},s}}
{Q_{\mathrm{scale},s}+\varepsilon}
\right)^2.
\]

默认参数为 \(w_p=0.1\)、\(\lambda_c=0.02\)、
\(\lambda_f=0.01\)。

## 5. 对比与消融实验

建议固定相同随机种子、空间掩码、模型容量和训练 epoch，至少报告：

| 编号 | 模型 | 均值物理项 | Darcy PDE | 目的 |
| --- | --- | --- | --- | --- |
| E1 | Baseline PINN | 否 | 否 | 纯数据基线 |
| E2 | Baseline PINN | 是 | 否 | 复现原物理约束 |
| E3 | Baseline PINN | 是 | 是 | 评价 PDE 对经典模型的贡献 |
| E4 | RBA-QPINN | 是 | 否 | 量子启发方法原版 |
| E5 | RBA-QPINN | 是 | 是 | 完整方法 |

当前命令行已提供 `--no-darcy-physics`，可直接完成 E2/E3 和 E4/E5 的
成对比较。若需要严格的 E1，后续还应把 `L_mean` 单独增加开关。

建议对 \(\lambda_c\in\{0.005,0.01,0.02,0.05\}\)、
\(\lambda_f\in\{0.002,0.005,0.01,0.02\}\) 做小规模敏感性分析，再锁定参数
运行完整 128 样本实验。

## 6. 报告指标

预测质量指标：

- held-out RMSE、MAE、R2、SSIM；
- 每样本 RMSE 分布与高梯度区域误差。

物理一致性指标：

- `Physics Residual`：预测均值相对工程条件期望均值的偏差；
- `Darcy Continuity Residual`：归一化散度残差的 RMS；
- `Darcy Flux Imbalance`：归一化进出口通量差的 RMS。

PDE 指标越低越好。论文中应同时报告预测指标和物理指标，避免仅以更平滑的场
换取较低 PDE 残差。

## 7. 适用边界

当前数据没有逐点压力、井源项、真实区域尺寸、绝对渗透率单位和真实边界条件，
所以本实验属于无量纲物理结构正则化。若上游后续提供压力场 \(p(x,y,t)\)、
总压缩系数 \(c_t\) 和井源项 \(s\)，应升级为瞬态控制方程

\[
\phi c_t\frac{\partial p}{\partial t}
-\nabla\cdot\left(\frac{k}{\mu}\nabla p\right)
=s(x,y,t),
\]

并增加压力数据、初始条件、井边界条件和 PDE 配置点损失。这时才能把任务表述
为更严格的渗透率反演 PINN。
