# RBA-QPINN 投稿就绪性检查

本文件保留历史V1协议的审计和阻断记录，不作为补充实验V2的完成度说明。当前稿及新增实验的状态以[最新验收记录](../paper/materical/MANUSCRIPT_V2_QA.md)为准。

## 文档入口与历史交付

- 证据审计源稿：`docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md`
- 当前长篇论文稿：[PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md](../paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)
- 五种子不可变实验：`experiments/rba_qpinn_submission_v1/seed_42` 至 `seed_46`
- 跨种子汇总：`experiments/rba_qpinn_submission_v1/aggregate`
- 主统计图：`experiments/rba_qpinn_submission_v1/aggregate/multiseed_rmse_raw_vs_corrected.png`
- 统计检验：`experiments/rba_qpinn_submission_v1/aggregate/statistical_tests.json`

## 结论等级

当前稿件已达到**探索性场内空间补全研究**的内部交付标准，但未达到期刊确认性投稿标准。三类空间掩码在点索引上互斥，五随机种子结果和统计分析也已归档；然而，场均值先验标定仍读取完整真实场统计量，IDW 校正仍使用完整真实场极值裁剪，因此不能称为严格无泄漏独立测试。

按 2026-09-09 的自动审计结果：

- `exploratory` 轨道：通过；
- `journal` 轨道：未通过；
- 主要阻断：训练期统计泄漏、校正期范围泄漏、缺少新确认性归档、缺少外部竞争基线、作者及投稿元数据未完成。

## 已完成的探索性研究基础

### 1. 场内空间三分协议：已完成

- `train_mask`：用于梯度训练；
- `validation_mask`：用于 checkpoint 选择；
- `test_mask`：训练完成后用于点值评价。

三类掩码互斥且覆盖完整网格。该协议检验同一批训练阶段已见场中的未观测位置恢复，不检验未知场泛化；完整场均值参与先验标定的问题仍需修复。

### 2. 不可覆盖结果归档：已完成

实验保存在 `experiments/rba_qpinn_submission_v1/seed_42` 至 `seed_46`。每个目录包含：

- `experiment_manifest.json`；
- 三类空间掩码与数据摘要；
- 六个模型的权重、训练历史及预测数组；
- Raw/Corrected 汇总与逐样本指标。

运行命令启用 `--fail-if-output-exists`，不会静默覆盖已有结果。

### 3. Raw 与 IDW-Corrected 分离：报告层面已完成

- Raw：网络原始输出，作为论文主结果；
- Corrected：网络输出 + 训练点 IDW 残差校正，作为次分析。

两套结果分别保存并在论文中以两张表报告。Raw 是当前主结果，但仍受训练期完整场均值标定影响；Corrected 还额外受完整真实场极值裁剪影响，只能作为诊断性次分析。

### 4. 五随机种子与探索性统计：已完成

已完成 seeds 42–46，共 30 次模型训练。主要结论：

- Baseline Raw RMSE：0.1741±0.0021；
- Quantum Feature PINN Raw RMSE：0.0845±0.0014；
- 平均相对改善：51.47%；
- 种子级单侧 Wilcoxon：\(p=0.03125\)；
- 种子级配对 t 检验单侧：\(p=1.59\times10^{-7}\)；
- 平均 RMSE 改善 bootstrap 95% CI：[0.0870, 0.0918]。

这些统计量描述固定 128 个场上的运行随机性。由于主比较由同批候选模型事后选出、运行数仅为 5，不能替代预注册确认性检验，也不包含从地质场总体重新抽样的不确定性。

## 确认性研究与投稿门禁

### 1. 增加新样本泛化协议

现有训练/测试点来自同一批场，且使用可学习样本嵌入。建议增加按场划分：

- train fields；
- validation fields；
- unseen test fields。

若新场无法使用已学习嵌入，应移除样本 ID 嵌入，或设计由观测点编码得到的场级表示。

### 2. 物理表述与 Darcy 约束

旧结果的物理项是场均值统计先验，不是局部 Darcy PDE 残差。当前主稿已改用“物理引导”表述。若要使用“严格 PINN”作为核心卖点，应完成 `docs/DARCY_PINN_EXPERIMENT.md` 中的成对实验，并报告：

- Darcy Continuity Residual；
- Darcy Flux Imbalance；
- 预测误差与物理残差的共同变化。

## 作者需要补充

- 作者、单位、通讯作者邮箱；
- 基金项目、致谢和 CRediT 作者贡献；
- 目标期刊及其模板、篇幅和参考文献格式；
- 数据与代码的匿名公开地址；
- 确认以实验最优的 Quantum Feature PINN 作为主模型。

## 建议投稿定位

基于当前结果，最稳妥的论文定位是：

> 周期相位特征增强的物理引导空间场补全方法及残差自适应模块的精度—一致性消融研究。

不建议在现阶段使用以下主张：

- “完整 RBA-QPINN 在所有指标上达到最优”；
- “实现量子加速”；
- “满足 Darcy 局部守恒”；
- “对未知储层具有良好泛化能力”；
- “完整 RBA-QPINN 优于只使用量子门等价特征的模型”。
