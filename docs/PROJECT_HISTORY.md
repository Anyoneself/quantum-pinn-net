# Quantum-Feature PINNs for Darcy Permeability Reconstruction

## V3：真实量子线路与经典 PINN 的受控比较

新研究入口为 [`qpinn/`](qpinn/)，默认使用 **GSTools 生成的渗透率场，只有渗透率观测**：
少量渗透率观测 + Darcy 方程 + 预设边界条件 → 渗透率场恢复。
压力网络是辅助未知场，不使用压力标签训练、筛选或选模；FVM压力参考仅用于训练后评价。
经典与量子组的压力网络及渗透率输入/输出层采用相同结构和对应初始化，各组独立训练，仅替换特征块。
量子特征来自 PennyLane 参数化量子线路的精确测量期望值，不是对普通正弦层的重新命名。

- **真正的物理约束**：训练中自动微分计算 `q=-k*grad(p)`、`div(q)=0` 和无流边界残差，左右压力边界由网络形式精确满足。
- **两项训练增强**：有界 RBA 配置点加权；多次随机子批次的配对特征消融和验证集接受的蒙特卡罗阈值筛选。
- **八组对照**：经典/量子 × 无增强/RBA/MC/RBA+MC。经典特征也接受对应筛选，避免只强化量子模型。
- **信息隔离**：拟合函数仅接收训练与验证观测；测试参考值只用于选模结束后的评价。主结果不使用插值校正。
- **结果边界**：这是无量纲合成场的逐场反演和量子线路模拟，不是现场预测、未知场零样本泛化或量子硬件加速证明。
- **数据边界**：使用GSTools重新生成可追溯的场，不复用V2已归一化、裁剪的数组；不借助完整场统计量标定或裁剪标签。Darcy一致性本身不保证渗透率唯一可辨识。

完整方法、公式、筛选规则和复现边界见 [V3 研究协议](docs/QUANTUM_PINN_V3.md)。
按用户确认大纲重写的中文稿见 [V3 论文](paper/QUANTUM_DARCY_PINN_MANUSCRIPT_V3.md)，
附 [补充材料](paper/materical/SUPPLEMENTARY_MATERIAL_V3.md)、
[文献阅读记录](paper/materical/LITERATURE_READING_V3.md) 和
[文稿核验记录](paper/materical/MANUSCRIPT_V3_QA.md)。主稿已统一为GSTools新生成场、仅渗透率监督和三项有效损失，
已嵌入中文架构图与训练流程图；重写前的三角级数与压力监督稿见[历史文本快照](paper/history/v3_before_gstools_rewrite_20261007/)。
旧 V1/V2 代码、结果及论文保留为历史记录，**不能作为 V3 量子算法的实验结论**。
下文原有的 V2 导航和说明均属于该历史路线。

### 环境与运行

建议使用独立 Python 3.12 环境。`requirements-v3.lock` 是本次核验的依赖快照；
原 `requirements.txt` 不是 V3 环境文件。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-v3.lock
# 不需要安装本项目即可从仓库根目录运行
.venv/bin/python -m qpinn run --smoke --output v3/experiments/quantum_pinn_v3_gstools_smoke_new
.venv/bin/python -m unittest discover -s tests -v

# 3个独立生成场 × 3次观测/初始化重复 × 8种方法
.venv/bin/python -m qpinn run --output v3/experiments/quantum_pinn_v3_gstools_research_new
.venv/bin/python -m qpinn verify --output v3/experiments/quantum_pinn_v3_gstools_research_new
```

输出目录必须不存在，避免覆盖已有证据。每次运行保存源代码快照、输入与输出哈希、
生成参数、观测划分、训练历史、筛选日志、最佳检查点、预测数组及配对比较。
`REPORT.md` 为结果入口；`--smoke` 只验证流程，不用于声称性能优势。
默认压力监督关闭；`--p-train 48` 显式开启联合观测协议。
`--field-generator trigonometric` 可选用历史生成器，不影响已有档案。

### GSTools 新运行记录

2026-10-07 已完成[仅渗透率监督的新72次实验](../v3/experiments/quantum_pinn_v3_gstools_konly_20261007/REPORT.md)
及[8组流程验证](../v3/experiments/quantum_pinn_v3_gstools_konly_smoke_20261007/REPORT.md)，39项测试通过。
经典与量子基础组的平均测试渗透率相对L2误差分别为0.337094和0.306759，后者低约9%；
MC单独加入量子模型后未改善该指标。新结果只适用于这批GSTools场与固定短预算，
不表示稳定的普遍增益或硬件优势。新旧档案分开保存，当前论文及补充材料已按新协议重写，未因此重新训练。

### 历史首轮运行记录

2026-10-07 已完成 [72次初步实验](../v3/experiments/quantum_pinn_v3_initial_20261007/REPORT.md)
及30项新旧测试。这批使用三角级数场和压力监督，不是当前GSTools协议。
量子基础模型的平均渗透率相对L2误差比经典基础模型低约3.1%，
但压力误差更高、模拟耗时更长；RBA+MC组合未进一步降低量子基础模型的该项误差。
这些结果不是量子优势或最终收敛结论，全部配置和负结果一并保留。

## 历史 V2：周期相位网络场内补全

本项目面向二维渗透率场内空间补全：在给定工程条件和部分空间观测值的情况下，预测训练阶段已见场中的未观测位置。当前论文比较固定 Fourier 坐标网络与完全由经典张量运算实现的可学习周期相位网络，并考察动态目标权重、训练点残差自适应加权和 Monte Carlo 神经元筛选。归档主实验不含局部 Darcy 方程残差，也不包含量子线路或量子硬件。

## V2 论文与补充实验

- [Markdown主稿](paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)：10幅图、32篇参考文献；[验收记录](paper/materical/MANUSCRIPT_V2_QA.md)说明检查范围和剩余研究边界。
- [补充实验归档](../v2/experiments/supplement_v2/README.md)：101组运行，包含预测数组、模型、逐轮记录及选型结果；本次保存没有重新训练。
- [图表与源数据](figures/manuscript_v2/)及[架构与训练流程图](figures/architecture/)：对应当前主稿。
- [文献阅读记录](paper/materical/literature-notes.md)与[写作规范](paper/materical/reference.md)：第三方论文全文及阅读用提取文本仅保留在本地，远端保存题录、来源链接与原创阅读概括。

当前实验中，周期网络原始输出相对Fourier对照的RMSE降低48.15%，校正后RMSE为0.01912，仍高于RBF的0.00247。结论限于固定数值场的场内补全，上游生成与归一化记录尚不完整。下文的早期实现、V1命令和五次运行结果保留作历史记录，不与当前三次重复的补充实验混用。已有Word/PDF为历史导出件，未同步当前Markdown修订。

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

代码可读取上述多种上游目录；V1归档包含 128 个 \(32\times32\) 数值场，其来源类别标识均为 `gan`，但完整上游生成链尚未追溯。V1在五次随机化运行中共享同一组场；补充实验复用固定数组，主比较采用三次重复。观测、验证与空间留出位置两两互斥。由于模型使用可学习场嵌入，评价协议属于训练阶段已见场内的空间补全，而不是未知场泛化。

## 方法

### Fourier 对照网络

Fourier 对照网络以 `(x, y, engineering conditions, learnable field embedding)` 为输入，使用固定 Fourier 坐标特征、SiLU 残差主干和 LayerNorm 输出连续场值。场嵌入用于区分训练阶段已见场。训练目标由四项组成：

- 数据项：观测点上的 MSE。
- 物理项：预测场均值与 Darcy/geomechanics-inspired 期望场均值之间的残差。
- 平滑项：相邻网格差分约束。
- 边界项：上下、左右边界均值一致性约束。

Fourier 对照网络使用固定目标权重训练。

### Darcy 偏微分方程约束

早期代码提供可选的稳态、单相、不可压缩 Darcy 诊断与训练约束，V1归档通过 `--no-darcy-physics` 将其排除在训练目标之外。以下说明这一早期实现；当前补充实验另以有限体积稀疏直接求解评价流动响应，不将其用于训练。对预测的归一化 log-permeability
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

## 复现历史V1协议

五个种子须分别输出到新目录，防止覆盖已有归档。以一个种子为例：

```bash
python3 gen-figures/run_pinn_experiment.py \
  --output-dir v1/experiments/rba_qpinn_submission_v1/seed_<seed> \
  --processed-dir v1/experiments/rba_qpinn_submission_v1/seed_<seed>/processed \
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
  --experiment-root v1/experiments/rba_qpinn_submission_v1 \
  --seeds 42,43,44,45,46
```

归档内部键对应六组配置：Fourier 对照网络、周期相位网络、周期相位 + 动态权重、周期相位 + 动态权重 + MC、周期相位 + 动态权重 + RBA，以及周期相位 + 动态权重 + RBA + MC。Darcy PDE 实现、离散方式和后续确认性实验设计见 [`docs/DARCY_PINN_EXPERIMENT.md`](docs/DARCY_PINN_EXPERIMENT.md)。

若重新配置环境，核心依赖为：

```bash
python -m pip install -r requirements.txt
```

## 历史V1实验结果

根目录 `figures/` 保留了早期开发实验，**不应用于当前结论或投稿引用**。下表来自
[`v1/experiments/rba_qpinn_submission_v1/`](../v1/experiments/rba_qpinn_submission_v1/) 中五次独立随机化运行；五次运行共享同一组 128 个场。这一旧协议存在完整场统计量参与先验和裁剪的问题，仅保留作历史探索性结果，不作为当前稿的独立测试证据。

| Model | Raw RMSE（均值±标准差，n=5） | Raw MAE | Raw R² | Raw SSIM | Weak Physics Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fourier 对照网络 | 0.1741 ± 0.0021 | 0.1317 ± 0.0014 | 0.7237 ± 0.0058 | 0.6866 ± 0.0122 | 0.1228 ± 0.0003 |
| 周期相位网络 | **0.0845 ± 0.0014** | **0.0591 ± 0.0009** | **0.9349 ± 0.0025** | **0.9384 ± 0.0028** | 0.1209 ± 0.0001 |
| 周期相位 + 动态权重 + RBA + MC | 0.2477 ± 0.0041 | 0.1966 ± 0.0034 | 0.4406 ± 0.0212 | 0.1852 ± 0.0291 | **0.0395 ± 0.0035** |

旧协议下，周期相位网络的原始 RMSE 相对 Fourier 对照低 **51.47%**，平均配对差为 0.0896，运行级 Bootstrap 95% 区间为 `[0.0870, 0.0918]`。该结果来自共享固定数据集的五次运行，并且两种网络同时改变多个架构因素，不能解释为相位编码的独立因果效应。沿累积路径加入动态权重、残差自适应和 MC 筛选后，场均值先验偏差下降，但点值精度未同步改善。

完整六模型结果、每种子的配置与统计检验见 [`v1/experiments/rba_qpinn_submission_v1/README.md`](../v1/experiments/rba_qpinn_submission_v1/README.md) 和 [`aggregate/`](../v1/experiments/rba_qpinn_submission_v1/aggregate/)。

## 文档与交付导航

| 文档 | 用途 |
| --- | --- |
| [`paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md`](paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md) | 当前Markdown学术主稿 |
| [`paper/materical/MANUSCRIPT_V2_QA.md`](paper/materical/MANUSCRIPT_V2_QA.md) | 当前验收结果与研究证据边界 |
| [`paper/materical/SUPPLEMENTARY_MATERIAL_V2.md`](paper/materical/SUPPLEMENTARY_MATERIAL_V2.md) | 当前实验协议、复现细节与补充分析 |
| [`docs/PAPER_SUBMISSION_WORKFLOW.md`](docs/PAPER_SUBMISSION_WORKFLOW.md) | 无 Skill 的复算、检查与投稿流程 |
| [`docs/PROJECT_DOCUMENTATION_GUIDE.md`](docs/PROJECT_DOCUMENTATION_GUIDE.md) | 文档地图、权威性规则和术语口径 |
| [`v1/experiments/rba_qpinn_submission_v1/README.md`](../v1/experiments/rba_qpinn_submission_v1/README.md) | 可复现实验协议与运行命令 |
| [`docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md`](docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md) | 完整研究报告／投稿预稿 |
| [`docs/QGE_PINN_GXNU_STYLE_SUBMISSION.md`](docs/QGE_PINN_GXNU_STYLE_SUBMISSION.md) | 广西师大学报风格稿件 |
| [`docs/RBA_QPINN_DELIVERY_AUDIT.md`](docs/RBA_QPINN_DELIVERY_AUDIT.md) | 证据审计、结论边界与后续整改项 |
| [`docs/RBA_QPINN_SUBMISSION_READINESS.md`](docs/RBA_QPINN_SUBMISSION_READINESS.md) | 投稿材料完成度与待补充信息 |
| [`docs/DARCY_PINN_EXPERIMENT.md`](docs/DARCY_PINN_EXPERIMENT.md) | Darcy 约束的实现与后续验证设计 |

## 历史图像

下列根目录图像来自早期开发结果，可用于内部检查或示意，**不应直接作为当前论文证据**。当前论文的数据图见 `figures/manuscript_v2/`，由 `v2/experiments/supplement_v2/` 的真实记录生成。

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

当前补充实验支持周期网络在有限预算下相对Fourier对照的拟合收益，但RBF插值精度仍优于网络方案。该结果不等同于量子计算优势或加速，也不证明对未见渗透率场或真实储层的泛化能力；训练目标不含Darcy方程残差，下游流动评价不等于网络实现了局部物理约束。组件贡献以受控消融为依据，不能将整体架构差异全部归因为相位特征。详见[当前主稿](paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)与[验收记录](paper/materical/MANUSCRIPT_V2_QA.md)。
