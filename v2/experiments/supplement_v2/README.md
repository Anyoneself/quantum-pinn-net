# 补充实验归档

本目录对应[当前Markdown论文](../../paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)。保存101组已完成运行，包括65次网络训练及36组插值参数搜索。本次提交归档已有结果，没有重新训练。

## 内容与来源

- `manifest.json`：输入和训练脚本的SHA-256、运行配置及实验环境。
- `seed_*/<scenario>/observations.npz`：观测值及训练、验证、测试掩码。
- 各方法目录的 `result.json`、`history.json`：最终评价及逐轮轨迹或插值选型记录。
- `raw.npy`、适用时的 `corrected.npy` 和 `model.pt`：预测数组与模型检查点。
- 固定输入位于 `../rba_qpinn_submission_v1/seed_42/processed/fields.npy` 和 `conditions.npy`，两者随此次归档提交，哈希与清单一致。

主比较采用三次重复，120轮预算诊断仅有一次重复。全部运行使用同一批128幅数值场，不能解释为独立数据集验证。完整协议见[实验方案](../../paper/materical/SUPPLEMENT_EXPERIMENT_PROTOCOL_V2.md)，实现细节及研究边界见[补充材料](../../paper/materical/SUPPLEMENTARY_MATERIAL_V2.md)。

## 复核与重现

在仓库根目录、具备所需科学计算依赖的环境中执行：

```bash
python gen-figures/verify_supplement_v2.py
python -B -m unittest discover -s tests -v
```

数值复核从保存预测重算指标，并核对表格、曲线和输入哈希，不进行训练。已生成图表和源数据位于 `figures/manuscript_v2/`。实验环境版本见 `manifest.json`；旧 `requirements.txt` 不是此次实验的锁定文件。

需要重新运行完整训练时，使用新目录保留本归档：

```bash
python gen-figures/run_supplement_v2.py --extended --output experiments/supplement_v2_rerun
```

第三方文献全文仅保存在本地，不在此仓库中再分发。`paper/materical/tools/verify_requirements.py` 除稿件结构外还检查本地阅读文本，运行前需依据阅读台账合法获取相应材料。该检查与数值复核不同，不应将材料缺失解释为实验未完成。
