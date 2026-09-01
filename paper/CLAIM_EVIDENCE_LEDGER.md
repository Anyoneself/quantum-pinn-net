# 主张—证据台账

对应稿件：`paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md`。当前证据等级：**探索性场内空间补全研究**。本台账中的“支持”仅表示主张可由现有归档复核，不表示实验已经满足无泄漏确认性研究要求。

## 核心主张

| ID | 论文可使用的主张 | 直接证据 | 状态 | 不得扩展为 |
|---|---|---|---|---|
| C-SCOPE-01 | 任务是在训练阶段已见场上的空间留出补全 | `run_pinn_experiment.py`；各实验 `dataset_summary.json`；可学习场嵌入实现 | 支持 | 独立场测试、未知场泛化 |
| C-DATA-01 | 归档包含 128 个 32×32 归一化对数渗透率场，来源标识均指向单一 GAN 类别 | 各实验 `processed/fields.npy`、`source_ids.npy` | 支持 | 多来源数据、真实储层数据 |
| C-LEAK-01 | 场均值先验标定使用完整真实场均值，主实验不是严格无泄漏确认性实验 | `expected_normalized_mean()`；`dataset_summary.json/expected_mean_fit` | 支持且构成投稿阻断 | “空间留出标签从未参与训练信息” |
| C-METHOD-01 | 周期相位网络完全由经典张量运算实现 | `models/quantum-pinn/quantum_pinn.py` | 支持 | 量子线路、量子硬件、量子加速 |
| C-ARCH-01 | Fourier 对照与周期相位网络是两个主干和参数量均不同的完整架构方案 | 两个模型源文件；参数形状计数 | 支持 | 相位编码的单因素因果效应 |
| C-THEORY-01 | 耦合相位通道具有 Bessel 系数控制的精确 Fourier 展开及一致收敛余项界 | 稿件命题 1，式（13）—（19） | 已证明 | 无限有效带宽、性能必然提高 |
| C-THEORY-02 | 周期特征有界，并具有由投影矩阵谱范数控制的 Lipschitz 上界 | 稿件命题 2—3，式（20）—（28） | 已证明 | 与训练参数无关的统一鲁棒性 |
| C-THEORY-03 | RBA 逐点权重及其加权误差与未加权误差之比有界 | 稿件命题 4，式（39）—（42） | 已证明 | RBA 必然降低泛化误差 |
| C-THEORY-04 | 正目标权重不足以保证数据损失下降；下降取决于梯度与实际更新方向的对齐 | 稿件命题 5，式（44）—（50） | 已证明 | 动态权重必然改善优化 |
| C-THEORY-05 | 场均值先验不蕴含局部 Darcy 守恒，并在式（51）的相对渗透率映射下与常数平移解耦 | 稿件式（51）—（55）的不变性与显式反例 | 已证明 | 场均值偏差等同于 PDE 残差 |
| C-RESULT-01 | 周期相位网络原始 RMSE 为 0.0845±0.0014，Fourier 对照为 0.1741±0.0021 | `aggregate/aggregate_metrics_raw.csv` | 探索性支持 | 严格无泄漏确认性性能 |
| C-RESULT-02 | 共享固定 128 场的五次独立随机化运行具有一致配对差方向，平均差为 0.0896，运行级 Bootstrap 95% 区间为 [0.0870, 0.0918] | `aggregate/statistical_tests.json` | 探索性支持 | 数据抽样不确定性、预注册显著性检验、无偏模型选择 |
| C-RESULT-03 | 周期相位网络在 640/640 个配对场—运行组合上单场 RMSE 更低 | `aggregate/manuscript_supporting_statistics.json` | 描述性支持 | 将 640 个组合视为独立推断样本 |
| C-RESULT-04 | 自适应模块的累积加入未形成单调精度增益，20 个动态配置运行中 19 个结束状态由平滑项主导 | 同上；各配置 `run_summary.json/history` | 探索性支持 | 任一模块在其他任务中普遍无效 |
| C-RESULT-05 | 两个 RBA 配置的结束状态逐点权重均未饱和至上限 4 | `aggregate/manuscript_supporting_statistics.json/rba_weight_statistics` | 描述性支持 | 排除其他目标尺度或训练预算下的饱和 |
| C-RESULT-06 | 含 MC 的配置每次随机化运行累计发生 140 次神经元级轻量重初始化 | `aggregate/manuscript_supporting_statistics.json/mc_statistics` | 描述性支持 | MC 的独立因果效应 |
| C-REPRO-01 | 表 4、表 5 和表 7 的主指标可由真实场、预测数组与空间留出掩码独立重算，并与派生 JSON/CSV 在 (10^{-6}) 容差内一致 | `aggregate/manuscript_supporting_statistics.json/array_crosscheck` | 支持 | 原始数组已公开或普通仓库克隆即可复算 |
| C-IDW-01 | IDW 校正后六种配置的 RMSE 均下降，模型间差距被明显压缩 | `aggregate/aggregate_metrics_corrected.csv`；IDW 实现 | 仅诊断 | 无泄漏后处理优势、网络本体性能 |
| C-PARAM-01 | Fourier 对照和周期相位系列的参数量分别为 142977 与 67301 | 模型源文件；参数形状计数 | 支持 | 相位表示带来一般性计算加速 |
| C-CAUSE-01 | 相位编码具有独立因果贡献 | 当前无共享主干、等参数因子实验 | 阻断 | 摘要或结论中的机制定论 |
| C-GEN-01 | 方法可泛化到全新地质场 | 当前无按完整场划分、无稀疏观测编码器 | 阻断 | 工程外推、真实储层适用性 |
| C-PHYS-01 | 当前模型满足局部 Darcy 守恒 | 归档明确关闭 Darcy 训练项 | 阻断 | PINN 物理守恒结论 |

## 表图—证据映射

| 正文对象 | 数据或实现来源 | 复算入口 |
|---|---|---|
| 表 1 工程条件统计 | `seed_42/processed/dataset_summary.json`；各实验条件数组一致 | 数据摘要与数组校验 |
| 表 2 组件与参数量 | `models/standard-pinn/standard_pinn.py`；`models/quantum-pinn/quantum_pinn.py` | 按参数形状求和 |
| 表 3 训练参数 | 各实验 `experiment_manifest.json/config` | 五个 manifest 一致性检查 |
| 表 4 原始总体指标 | `aggregate/aggregate_metrics_raw.csv` | `aggregate_multiseed_results.py` |
| 表 5 逐场分布 | 各配置 `per_sample_metrics_raw.json` | `analyze_manuscript_evidence.py` |
| 表 6 动态权重结束状态 | 各配置 `run_summary.json/history/loss_weights` | `analyze_manuscript_evidence.py` |
| 表 7 IDW 次分析 | `aggregate/aggregate_metrics_corrected.csv` | 聚合脚本与 IDW 实现 |
| 图 1 架构 | 两个模型源文件、损失与自适应模块实现 | `figures/manuscript/phase_periodic_architecture.svg` |
| 图 2 空间重建 | 第一次归档实验的原始预测、真实场和掩码 | `generate_manuscript_spatial_figure.py`；同名 JSON 记录客观选样规则 |

## 投稿使用规则

1. 摘要、结果与结论只能使用“支持”或明确标注为“探索性支持”的主张。
2. 所有性能数值必须同时保留“训练阶段已见场”“完整架构方案比较”和“场均值先验存在信息泄漏”三项边界。
3. `blocked` 主张只能出现在局限性或未来工作中。
4. 正式确认性重跑完成后，应先更新本台账，再同步摘要、表格、图注和结论。
5. 作者、基金、利益冲突、数据许可证、公开链接和目标期刊格式不属于数值证据，但均是最终投稿前的独立门禁。
