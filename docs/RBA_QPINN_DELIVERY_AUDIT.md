# RBA-QPINN 论文交付与证据审计

## 交付定位

- 主稿：`docs/RBA_QPINN_SUBMISSION_MANUSCRIPT.md`
- 语言：中文正文，含英文标题与 Abstract
- 引用格式：顺序编码制，按 GB/T 7714 要素整理
- 当前定位：完整研究报告／投稿预稿 v0
- 目标定位：一区 TOP 期刊
- 当前判断：尚未达到一区 TOP 投稿就绪标准

## 推荐题目

可学习周期相位编码网络用于非均质渗透率场内空间补全：多随机种子评估与精度—先验一致性权衡

## 已使用的本地证据

- 五随机种子实验：`experiments/rba_qpinn_submission_v1/seed_42` 至 `seed_46`
- 聚合指标：`experiments/rba_qpinn_submission_v1/aggregate/aggregate_metrics_raw.csv`
- IDW 次结果：`experiments/rba_qpinn_submission_v1/aggregate/aggregate_metrics_corrected.csv`
- 统计检验：`experiments/rba_qpinn_submission_v1/aggregate/statistical_tests.json`
- 运行环境：`experiments/rba_qpinn_submission_v1/aggregate/environment.json`
- 实验协议：`experiments/rba_qpinn_submission_v1/README.md`
- 模型源码：`models/standard-pinn/standard_pinn.py`、`models/quantum-pinn/quantum_pinn.py`
- 训练与评价实现：`gen-figures/run_pinn_experiment.py`
- 数据来源标识：`dataset/processed/source_ids.npy`

## 当前证据支持的结论

1. 当前任务是同一批已登记渗透率场内部的空间留出点补全，不是未知场泛化。
2. Quantum Feature PINN 架构包在归档五种子 Raw 结果中最优，RMSE 为 \(0.0845\pm0.0014\)。
3. 相较 Baseline PINN 的 \(0.1741\pm0.0021\)，其平均 Raw RMSE 降低 51.47%。
4. Full RBA-QPINN 的 Raw RMSE 为 \(0.2477\pm0.0041\)，不是最佳模型。
5. 动态损失、RBA 和 MC 的组合降低场均值先验偏差，却损害点值与结构精度。
6. IDW 校正后最佳方法身份不变，但其余方法排序改变；Raw 与 Corrected 必须分开报告。

## 不能使用的主张

- 完整 RBA-QPINN 全面优于其他模型；
- 量子优势、量子加速或真实量子计算；
- 当前训练满足局部 Darcy 质量守恒；
- 对未知渗透率场或真实储层具有良好泛化；
- 51.47% 的全部改善由相位特征单独造成；
- 当前结果已经达到一区 TOP 投稿标准。

## TOP 投稿阻断项

1. `expected_normalized_mean()` 使用完整真实场均值标定先验；IDW 输出又用完整真实场最小值/最大值裁剪。两处都包含验证和测试标签，必须同时修复并重跑。
2. Baseline 与 Quantum Feature PINN 同时改变编码、激活、门控、归一化和残差主干，必须做等骨干因子消融。
3. 128 个归档样本的 `source_ids` 全为 0，实际来自单一 GAN 数据目录；需忠实披露或重新分层采样。
4. 场专属 sample embedding 不支持新场；需按完整场划分并改用观测编码器。
5. 正式实验关闭 Darcy 训练；若保留 PINN/Darcy 主张，需补齐物理变量、边界和守恒实验。
6. 缺少 IDW-only、Kriging、RBF、强坐标网络和神经算子等外部基线。
7. 缺少观测率、噪声、分辨率、训练预算和主要超参数敏感性。
8. 主要推断单位只有 5 个随机种子；确认性实验建议预先指定至少 10 个种子。
9. Baseline 约 142977 参数，Quantum Feature PINN 约 67301 参数，深度定义也不同；需匹配参数量和计算预算。
10. 聚合主图仍含 “Independent-test” 旧标签，图 2 又是按测试收益后验选择的最佳案例；投稿前需重生成图表。

## 【待确认】信息

- 作者姓名、排序与 ORCID；
- 作者单位、通讯作者和邮箱；
- 基金、致谢及 CRediT 作者贡献；
- 具体目标期刊及其模板；
- 数据和代码的匿名公开地址；
- 上游 GSTools 数据生成参数及数据许可；
- 是否授权修复实验代码并重新运行无泄漏确认性实验；
- 是否继续使用 PINN 命名，或统一改为 prior-guided neural network。
- 引言所需的渗透率非均质性、连通性与储层表征领域文献。

## 本轮验证限制

当前默认 Python 环境缺少 NumPy、PyTorch 和 pytest，未能重新运行单元测试或训练。论文数值来自已归档 CSV/JSON、manifest 与源码的交叉核验；现有原始实验和图表未被覆盖。
