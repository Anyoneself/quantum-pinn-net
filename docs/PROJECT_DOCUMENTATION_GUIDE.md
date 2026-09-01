# 项目文档指南

## 目的

本指南为项目建立统一入口，明确哪些材料可用于复现、论文写作和结论引用。它不替代实验归档或论文正文；数值结论应始终回溯到不可变实验目录中的 CSV、JSON、manifest 和预测数组。

## 文档分层与权威性

| 层级 | 材料 | 使用场景 | 权威性 |
| --- | --- | --- | --- |
| A | `experiments/rba_qpinn_submission_v1/seed_42` 至 `seed_46` | 单次运行配置、掩码、权重、Raw/Corrected 预测 | 原始归档，最高 |
| A | `experiments/rba_qpinn_submission_v1/aggregate/` | 五次随机化运行汇总、统计检验、主图 | 当前数值结论的唯一来源 |
| B | `experiments/rba_qpinn_submission_v1/README.md` | 复现实验和输出说明 | 当前协议说明 |
| B | `paper/PHASE_PERIODIC_PINN_MANUSCRIPT_V2.md` | 四章学术正文、公式、表图与参考文献 | 当前主稿 |
| B | `paper/CLAIM_EVIDENCE_LEDGER.md` | 主张—证据映射与投稿边界 | 结论门禁 |
| B | `docs/QGE_PINN_GXNU_STYLE_SUBMISSION.md` | 指定中文期刊风格版本 | 排版与投稿改写依据 |
| C | `docs/RBA_QPINN_DELIVERY_AUDIT.md` | 审计、风险与确认性实验路线 | 结论审查依据 |
| C | `docs/RBA_QPINN_SUBMISSION_READINESS.md` | 投稿行政材料的完成度 | 交付清单 |
| H | 根目录 `figures/`、早期草稿 | 开发记录与图形素材 | 历史材料，不作正式数值依据 |

## 当前实验口径

- 数据：128 个 `32×32` 归一化 log-permeability 合成场；当前归档的来源类别标识均为 `gan`。
- 任务：训练阶段已见场内部的空间补全，不是对新场或真实储层的外推。
- 划分：每次随机化运行重新生成互斥的训练、验证与空间留出掩码，具体比例保存在运行配置中。
- 重复：运行标识 42–46，共享同一组 128 个场；区间估计只刻画固定数据集上的运行随机性，不包含数据抽样不确定性。
- 主结果：Raw 网络输出在独立测试掩码上的 RMSE、MAE 与 R²；IDW-Corrected 结果属于次分析。
- 物理项：归档训练使用弱的场均值先验，且默认关闭 Darcy PDE 训练；`Physics Residual` 不可写作局部 Darcy 守恒误差。

## 已支持与不支持的结论

| 可以陈述 | 不应陈述 |
| --- | --- |
| 周期相位网络在当前归档协议下取得最低 Raw RMSE（0.0845±0.0014） | 完整自适应组合在所有指标上最优 |
| 相较 Fourier 对照，其 Raw RMSE 数值低 51.47%，五次随机化运行方向一致 | 量子优势、量子加速或使用真实量子硬件 |
| 当前累积模块显示预测精度与弱先验一致性的非单调权衡 | 已验证局部 Darcy 守恒或真实储层泛化 |
| Raw 与 IDW-Corrected 应作为两套结果分别报告 | 将架构包差异全部归因于单一相位特征 |

## 推荐阅读路径

1. 了解项目：根目录 `README.md`。
2. 复现实验：`experiments/rba_qpinn_submission_v1/README.md`，再查看各运行目录的 `experiment_manifest.json`。
3. 核对数值：`aggregate/aggregate_metrics_raw.csv` 与 `aggregate/statistical_tests.json`。
4. 撰写论文或答辩：先读 V2 主稿与 `paper/CLAIM_EVIDENCE_LEDGER.md`，再核对投稿工作流。
5. 推进确认性实验：按审计文档中的泄漏修复、等骨干消融、跨场泛化和强基线顺序实施。

## 文档维护规则

1. 新实验必须写入新的不可覆盖目录，并保存命令、依赖环境、掩码与随机化运行标识。
2. 聚合结果更新后，同步更新根 README、主稿和本指南中的数值与主张。
3. 文中首次出现结果时标注 `Raw` 或 `Corrected`、空间留出掩码和随机化运行数。
4. 历史图仅用于开发或示意；论文图须能回溯到当前归档实验和生成脚本。
5. 未重新运行并审计前，不得把候选方案、旧版结果或设计目标写为已验证结论。
