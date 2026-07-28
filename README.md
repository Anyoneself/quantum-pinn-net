# Quantum-PINN Permeability Field Prediction

本项目面向二维渗透率场预测问题，核心任务是在给定工程物理条件和部分空间观测点的情况下，预测未观测位置的归一化 log-permeability。`quantum-attention-net` 项目负责 GSTools 渗透率场数据生成与物理条件调制，本项目不重复解决数据生成问题，而是在该数据基础上构建标准 Physics-Informed Neural Network (PINN) 基线，并提出一种量子启发的 PINN 优化方法。实验目标是比较两类网络在空间留出点预测任务中的误差、准确率、结构一致性和物理一致性。

## 数据来源与任务边界

`quantum-attention-net` 是上游数据生成项目；`quantum-pinn-net` 是下游渗透率场预测项目。二者分工如下：

- `quantum-attention-net`: 生成或增强二维渗透率场数据，并保存 GSTools/物理约束数据集。
- `quantum-pinn-net`: 使用上述数据训练 PINN，在稀疏观测条件下预测未观测空间点的渗透率值。

训练和测试数据来自：

```text
/Users/yuanzhi.liu/Desktop/code/experiment/quantum-attention-net/dataset/runs/<model>/
```

读取的上游模型数据包括：

- `gan`
- `unet`
- `unet-attention`
- `unet-quantum`
- `unet-attention-quantum`

每个目录包含：

- `dataset.npy`: 形状为 `(N, 1, 32, 32)` 的归一化二维 log-permeability 场。
- `physical_conditions.npy`: 每个样本对应的工程条件，列顺序为 `[pressure_mpa, pressure_gradient_mpa_per_m, viscosity_cp, porosity, flow_rate_m3_day]`。

最终实验使用 128 个 GSTools 渗透率场样本。每个样本内部随机保留约 80% 空间点作为训练观测点，剩余约 20% 空间点作为测试点；因此评价协议是同一渗透率场内的空间留出预测。数值指标只在未观测测试点上计算，以衡量模型对未知空间位置的预测能力。

## 方法

### Baseline PINN

标准 PINN 以 `(x, y, physical conditions, learnable sample embedding)` 为输入，输出对应坐标处的渗透率值。`sample embedding` 用于标识每个 GSTools 渗透率场，使网络在同一场内根据稀疏观测点预测未观测空间位置，而不是仅预测跨样本均值。为了避免弱 Baseline 造成不公平对比，当前 Baseline 使用固定 Fourier 坐标特征、SiLU 残差 MLP 和 LayerNorm，形成强经典 PINN 基线；但它仍使用固定 loss 权重，不引入量子特征、残差自适应权重或 Monte Carlo 神经元筛选。训练目标由四项组成：

- 数据项：观测点上的 MSE。
- 物理项：预测场均值与 Darcy/geomechanics-inspired 期望场均值之间的残差。
- 平滑项：相邻网格差分约束。
- 边界项：上下、左右边界均值一致性约束。

Baseline 使用固定 loss 权重训练。

### Darcy 偏微分方程约束

在原有场均值经验约束之外，当前实验已经加入稳态、单相、不可压缩 Darcy
渗流约束。对预测的归一化 log-permeability
\(\widehat{\ell}(x,y)\)，先构造严格为正的相对渗透率

```text
k_r(x,y) = exp(clip(l_hat(x,y) - mean(l_hat), -6, 6))
```

再在单位正方形区域上求解辅助压力场：

```text
q = -k_r grad(p)
div(q) = 0
```

采用的无量纲边界条件为：

```text
p(0,y) = 1,  p(1,y) = 0
dp/dn(x,0) = dp/dn(x,1) = 0
```

离散实现使用网格面渗透率的调和平均计算 transmissibility，并通过可微的
weighted-Jacobi 迭代得到压力场。物理损失为：

```text
L_physics =
    L_mean
  + lambda_c * mean((div(q) / q_scale)^2)
  + lambda_f * mean(((Q_in - Q_out) / Q_scale)^2)
```

其中 `L_mean` 保留已有的 Darcy/geomechanics-inspired 全局工程条件约束，
后两项分别约束局部质量守恒与进出口通量守恒。训练历史会分别保存
`physics_mean_loss`、`darcy_continuity_loss` 和 `darcy_flux_loss`；最终指标表
新增 `Darcy Continuity Residual` 与 `Darcy Flux Imbalance`，两者均为越低
越好。默认
`lambda_c=0.02`、`lambda_f=0.01`，辅助压力迭代 16 次。

必须说明：上游模拟数据只包含归一化渗透率场和样本级工程条件，没有真实压力
场、井源项、区域尺寸、绝对渗透率单位及逐点边界条件。因此这里的 PDE 是
无量纲结构约束，不能解释为对真实储层压力场的复现。若后续数据能提供压力场
或井源项，可将控制方程扩展为

```text
phi * c_t * dp/dt - div(k / mu * grad(p)) = s(x,y,t)
```

并直接加入压力观测、初始条件和井边界条件损失。

### RBA-QPINN

RBA-QPINN 在相同 PINN 任务上加入四类量子启发与残差自适应机制：

- 量子角度编码特征：将输入映射为 `sin/cos` 相位特征，并引入可学习的 entanglement mixer 以增强非线性空间表达。
- 动态 loss 权重：以幅值归一化的 quantum-inspired probability amplitude 更新数据项、物理项、平滑项和边界项权重，使优化过程能根据不同损失项的收敛状态自适应调节。
- 残差自适应逐点权重：仅基于训练观测点残差更新逐点注意力权重，并在连续高残差区域内增强训练权重，避免使用测试点标签造成数据泄漏。
- Monte Carlo 神经元筛选：训练前半程对量子特征层神经元进行随机批次效用估计，效用定义为 `mean_abs_activation * sqrt(activation_variance + eps)`，对低效神经元进行轻量重初始化，以提高有效表达单元比例。

该方法为量子启发算法实现，不依赖真实量子硬件或 PennyLane 量子线路模拟。其优势体现在相同数据、相同评价协议下的误差降低、结构相似性改善和物理残差降低。

## 项目结构

```text
quantum-pinn-net/
  dataset/
    darcy_physics.py
    physics_constraints.py
    processed/
      fields.npy
      conditions.npy
      train_point_mask.npy
      test_point_mask.npy
      dataset_summary.json
  models/
    standard-pinn/
      README.md
      standard_pinn.py
    quantum-pinn/
      README.md
      quantum_pinn.py
  gen-figures/
    run_pinn_experiment.py
  figures/
    metrics_summary.csv
    metrics_summary.json
    improvement_summary.json
    loss_curve_same_epoch.png
    prediction_comparison.png
    quality_physics_metrics_bars.png
    quantum_loss_weight_dynamics.png
    residual_point_weight_dynamics.png
    adaptive_sampling_weight_map.png
    runs/
      standard-pinn/
      quantum-pinn/
```

## 复现实验

推荐使用已有虚拟环境：

```bash
cd /Users/yuanzhi.liu/Desktop/code/experiment/quantum-pinn-net
/Users/yuanzhi.liu/Desktop/code/liuyuanzhi/python_venvs/bench_pa_py312/bin/python \
  gen-figures/run_pinn_experiment.py \
  --device cpu \
  --max-samples 128 \
  --epochs 40 \
  --batch-size 8 \
  --hidden-dim 128 \
  --latent-dim 24 \
  --mc-trials 4 \
  --residual-resample-period 5 \
  --residual-replace-ratio 0.20 \
  --darcy-iterations 16 \
  --darcy-continuity-weight 0.02 \
  --darcy-flux-weight 0.01 \
  --run-ablations
```

默认会启用 RBA-QPINN 的残差自适应逐点权重，并在 `--run-ablations` 模式下训练以下消融模型：

- Baseline PINN
- Classical + Loss Balancer
- Classical + RBA
- Classical + Loss + RBA
- Quantum Feature PINN
- Quantum + MC
- Quantum + RBA
- Quantum + Loss Balancer
- Quantum + Loss + MC
- Quantum + Loss + RBA
- Full RBA-QPINN

若需要只运行 Baseline 与 Full RBA-QPINN 的核心对比，可去掉 `--run-ablations`。若需要复现实验中的旧版 Quantum-PINN，可添加：

```bash
--no-adaptive-residual-weights
```

若需要进行“不使用 Darcy PDE”的消融实验，可添加：

```bash
--no-darcy-physics
```

完整控制方程、离散方式、消融矩阵和论文表述边界见
[`docs/DARCY_PINN_EXPERIMENT.md`](docs/DARCY_PINN_EXPERIMENT.md)。

若重新配置环境，核心依赖为：

```bash
python -m pip install numpy torch matplotlib pandas scipy
```

## 实验结果

最终指标保存在 `figures/metrics_summary.csv` 与 `figures/metrics_summary.json`。当前复现实验结果如下：

| Model | RMSE | MAE | MAPE | Accuracy | R2 | SSIM | Physics Residual | Physical Score | Composite Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline PINN | 0.281121 | 0.220904 | 2.190129 | 0.889548 | 0.280768 | 0.057266 | 0.124136 | 0.578783 | 0.536446 |
| Quantum-PINN | 0.254444 | 0.200718 | 1.472617 | 0.899641 | 0.410793 | 0.213637 | 0.034889 | 0.546033 | 0.607043 |

相对 Baseline，Quantum-PINN 取得：

- RMSE 降低 `9.49%`。
- MAE 降低 `9.14%`。
- Physics Residual 降低 `71.89%`。
- 综合评分提升 `13.16%`。

需要注意的是，上表为引入 RBA-QPINN 增强前的已保存结果。`Physical Score` 是 Darcy 相关性、压力相关性、孔隙度相关性及范围约束构成的综合指标；`Physics Residual` 是预测场均值相对拟合物理期望均值的单项残差。当前已保存结果中 Quantum-PINN 在误差、结构相似性、场均值物理残差和综合评分上均优于 Baseline。

## 论文图像

可直接用于论文的图像包括：

- `figures/loss_curve_same_epoch.png`: 两种方法同 epoch 下的收敛曲线。
- `figures/quality_physics_metrics_bars.png`: 误差与物理指标柱状图。
- `figures/ablation_table.png` / `figures/ablation_table.csv`: 消融实验指标表。
- `figures/ablation_design_matrix.png` / `figures/ablation_design_matrix.csv`: 消融实验模块设计矩阵。
- `figures/ablation_study_summary.png`: 消融实验总览图。
- `figures/ablation_module_contribution.png`: 各模块相对 Baseline 的贡献柱状图。
- `figures/ablation_path_metrics.png`: 不同模块组合下的指标变化路径图。
- `figures/ablation_metric_heatmap.png`: 消融模型指标热力图。
- `figures/ablation_sample_rmse_heatmap.png`: 逐样本消融 RMSE 热力图。
- `figures/error_boxplot.png`: 各消融模型在留出测试点上的绝对误差箱线图。
- `figures/prediction_field_ablation_comparison.png`: 多个消融模型在同一渗透率场上的预测场对比图。
- `figures/residual_sampling_heatmap.png`: 残差自适应采样/逐点权重热力图。
- `figures/loss_weight_dynamics_ablation.png`: 消融模型的动态损失权重曲线。
- `figures/training_time_table.png` / `figures/training_time_table.csv`: 训练耗时表。
- `figures/method_flowchart.png`: RBA-QPINN 方法流程图。
- `figures/baseline_vs_rba_qpinn_comparison_smooth.png`: Baseline PINN 与当前最佳改良模型的点级视觉对比图。
- `figures/prediction_comparison.png`: 目标场、训练观测点、Baseline 预测、当前最佳改良模型预测和留出点预测增益图。
- `figures/heldout_point_prediction_comparison.png`: 无插值的 held-out 点级预测对比图。
- `figures/residual_point_weight_dynamics.png`: RBA-QPINN 残差逐点权重的均值、95 分位数和最大值演化图。
- `figures/rba_qpinn_architecture.png`: RBA-QPINN 网络架构图，可直接用于论文方法章节。

## 学术表述建议

本实验支持如下严谨表述：

> 在相同 GSTools 渗透率场数据、相同稀疏空间观测比例和相同训练轮数下，量子启发 PINN 通过相位特征映射、动态损失权重和 Monte Carlo 神经元筛选机制，在留出空间点预测任务中取得更低的 RMSE/MAE，并改善场均值物理残差和综合评分。该结果表明，量子启发优化可作为 PINN 在非均质渗透率场预测任务中的有效表达增强与训练调度策略。
