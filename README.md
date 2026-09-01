# Phase-Periodic Network for Permeability-Field Completion

本项目面向二维渗透率场内空间补全：在给定工程条件和部分空间观测值的情况下，预测训练阶段已见场中的未观测位置。当前论文比较固定 Fourier 坐标网络与完全由经典张量运算实现的可学习周期相位网络，并考察动态目标权重、训练点残差自适应加权和 Monte Carlo 神经元筛选。归档主实验不含局部 Darcy 方程残差，也不包含量子线路或量子硬件。

## 数据来源与任务边界

`quantum-attention-net` 是上游数据生成项目；`quantum-pinn-net` 是下游渗透率场预测项目。二者分工如下：

- `quantum-attention-net`: 生成或增强二维渗透率场数据，并保存 GSTools/物理约束数据集。
- `quantum-pinn-net`: 使用上述数据训练 PINN，在稀疏观测条件下预测未观测空间点的渗透率值。

训练和测试数据来自：

```text
<upstream-project>/dataset/runs/<model>/
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

代码可读取上述多种上游目录；当前正式归档包含 128 个 \(32\times32\) 合成场，其来源类别标识均为 `gan`。同一组场在五次独立随机化运行中共享，观测、验证与空间留出位置两两互斥。由于模型使用可学习场嵌入，评价协议属于训练阶段已见场内的空间补全，而不是未知场泛化。

## 方法

### Fourier 对照网络

Fourier 对照网络以 `(x, y, engineering conditions, learnable field embedding)` 为输入，使用固定 Fourier 坐标特征、SiLU 残差主干和 LayerNorm 输出连续场值。场嵌入用于区分训练阶段已见场。训练目标由四项组成：

- 数据项：观测点上的 MSE。
- 物理项：预测场均值与 Darcy/geomechanics-inspired 期望场均值之间的残差。
- 平滑项：相邻网格差分约束。
- 边界项：上下、左右边界均值一致性约束。

Fourier 对照网络使用固定目标权重训练。

### Darcy 偏微分方程约束

代码提供可选的稳态、单相、不可压缩 Darcy 诊断与训练约束，但当前论文归档通过 `--no-darcy-physics` 将其排除在训练目标之外。对预测的归一化 log-permeability
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

### 周期相位网络与自适应机制

周期相位网络在相同场内补全任务上使用或考察以下机制：

- 周期相位特征：将输入映射为 `sin/cos` 相位响应，并通过可学习通道混合形成谐波频率族。
- 动态目标权重：以归一化概率幅值更新数据、场均值、平滑和边缘统计目标的相对权重。
- 残差自适应逐点权重：仅基于训练观测点残差更新逐点注意力权重，并在连续高残差区域内增强训练权重，避免使用测试点标签造成数据泄漏。
- Monte Carlo 神经元筛选：训练前半程对正弦支路神经元进行随机批次效用估计，效用定义为 `mean_abs_activation * sqrt(activation_variance + eps)`，对低效神经元进行轻量重初始化。

上述模块全部采用经典张量运算。当前证据只支持完整架构方案的探索性比较，不支持量子优势、相位编码的独立因果贡献或局部物理守恒。

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

## 复现当前归档协议

五个种子须分别输出到新目录，防止覆盖已有归档。以一个种子为例：

```bash
python3 gen-figures/run_pinn_experiment.py \
  --output-dir experiments/rba_qpinn_submission_v1/seed_<seed> \
  --processed-dir experiments/rba_qpinn_submission_v1/seed_<seed>/processed \
  --fail-if-output-exists \
  --paper-ablations \
  --seed <seed> \
  --epochs 40 --batch-size 8 --hidden-dim 128 --depth 4 --latent-dim 24 \
  --learning-rate 0.001 --train-ratio 0.70 --validation-ratio 0.10 \
  --max-samples 128 --device cpu --mc-trials 4 \
  --residual-resample-period 5 --residual-replace-ratio 0.20 \
  --no-darcy-physics
```

完成 seeds 42–46 后聚合：

```bash
python3 gen-figures/aggregate_multiseed_results.py \
  --experiment-root experiments/rba_qpinn_submission_v1 \
  --seeds 42,43,44,45,46
```

归档内部键对应六组配置：Fourier 对照网络、周期相位网络、周期相位 + 动态权重、周期相位 + 动态权重 + MC、周期相位 + 动态权重 + RBA，以及周期相位 + 动态权重 + RBA + MC。Darcy PDE 实现、离散方式和后续确认性实验设计见 [`docs/DARCY_PINN_EXPERIMENT.md`](docs/DARCY_PINN_EXPERIMENT.md)。

若重新配置环境，核心依赖为：

```bash
python -m pip install -r requirements.txt
```

## 当前权威实验结果

根目录 `figures/` 保留了早期开发实验，**不应用于当前结论或投稿引用**。当前权威结果来自
[`experiments/rba_qpinn_submission_v1/`](experiments/rba_qpinn_submission_v1/) 中五次不可覆盖的独立随机化运行；五次运行共享同一组 128 个场，主结果为网络的原始输出，IDW 校正仅作为次分析。

| Model | Raw RMSE（均值±标准差，n=5） | Raw MAE | Raw R² | Raw SSIM | Weak Physics Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fourier 对照网络 | 0.1741 ± 0.0021 | 0.1317 ± 0.0014 | 0.7237 ± 0.0058 | 0.6866 ± 0.0122 | 0.1228 ± 0.0003 |
| 周期相位网络 | **0.0845 ± 0.0014** | **0.0591 ± 0.0009** | **0.9349 ± 0.0025** | **0.9384 ± 0.0028** | 0.1209 ± 0.0001 |
| 周期相位 + 动态权重 + RBA + MC | 0.2477 ± 0.0041 | 0.1966 ± 0.0034 | 0.4406 ± 0.0212 | 0.1852 ± 0.0291 | **0.0395 ± 0.0035** |

当前证据支持：周期相位网络的原始 RMSE 相对 Fourier 对照低 **51.47%**，平均配对差为 0.0896，运行级 Bootstrap 95% 区间为 `[0.0870, 0.0918]`。该结果来自共享固定数据集的五次运行，并且两种网络同时改变多个架构因素，不能解释为相位编码的独立因果效应。沿累积路径加入动态权重、残差自适应和 MC 筛选后，场均值先验偏差下降，但点值精度未同步改善。

完整六模型结果、每种子的配置与统计检验见 [`experiments/rba_qpinn_submission_v1/README.md`](experiments/rba_qpinn_submission_v1/README.md) 和 [`aggregate/`](experiments/rba_qpinn_submission_v1/aggregate/)。

## 文档与交付导航

| 文档 | 用途 |
| --- | --- |
| [`paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md`](paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md) | 当前四章学术主稿 |
| [`paper/CLAIM_EVIDENCE_LEDGER.md`](paper/CLAIM_EVIDENCE_LEDGER.md) | 主张—证据边界与投稿阻断项 |
| [`docs/PAPER_SUBMISSION_WORKFLOW.md`](docs/PAPER_SUBMISSION_WORKFLOW.md) | 无 Skill 的复算、检查与投稿流程 |
| [`docs/PROJECT_DOCUMENTATION_GUIDE.md`](docs/PROJECT_DOCUMENTATION_GUIDE.md) | 文档地图、权威性规则和术语口径 |
| [`experiments/rba_qpinn_submission_v1/README.md`](experiments/rba_qpinn_submission_v1/README.md) | 可复现实验协议与运行命令 |
| [`docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md`](docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md) | 完整研究报告／投稿预稿 |
| [`docs/QGE_PINN_GXNU_STYLE_SUBMISSION.md`](docs/QGE_PINN_GXNU_STYLE_SUBMISSION.md) | 广西师大学报风格稿件 |
| [`docs/RBA_QPINN_DELIVERY_AUDIT.md`](docs/RBA_QPINN_DELIVERY_AUDIT.md) | 证据审计、结论边界与后续整改项 |
| [`docs/RBA_QPINN_SUBMISSION_READINESS.md`](docs/RBA_QPINN_SUBMISSION_READINESS.md) | 投稿材料完成度与待补充信息 |
| [`docs/DARCY_PINN_EXPERIMENT.md`](docs/DARCY_PINN_EXPERIMENT.md) | Darcy 约束的实现与后续验证设计 |

## 历史图像

下列根目录图像来自早期开发结果，可用于内部检查或示意，**不应直接作为当前论文证据**。当前论文主图应从 `experiments/rba_qpinn_submission_v1/aggregate/` 或各 seed 目录重新生成并注明协议。

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
- `figures/baseline_vs_rba_qpinn_comparison_smooth.png`: 旧版 Fourier 对照与改良模型的点级视觉对比图。
- `figures/prediction_comparison.png`: 目标场、训练观测点、Baseline 预测、当前最佳改良模型预测和留出点预测增益图。
- `figures/heldout_point_prediction_comparison.png`: 无插值的 held-out 点级预测对比图。
- `figures/residual_point_weight_dynamics.png`: RBA-QPINN 残差逐点权重的均值、95 分位数和最大值演化图。
- `figures/rba_qpinn_architecture.png`: RBA-QPINN 网络架构图；使用前须核对其与当前正式模型及协议一致。

## 结论边界

本项目的严谨表述是：在当前 128 个训练阶段已见合成场的空间补全协议下，周期相位完整架构取得最低平均原始 RMSE。该结果不等同于量子计算优势或加速，也不证明对未见渗透率场或真实储层的泛化能力；归档训练目标不含 Darcy 方程残差，因此不能声称实现了局部 Darcy 守恒。Fourier 对照与周期相位网络同时改变多项架构设计，不能将全部差异单独归因为相位特征。详见 [`paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md`](paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md) 与 [`paper/CLAIM_EVIDENCE_LEDGER.md`](paper/CLAIM_EVIDENCE_LEDGER.md)。
