# V3 有限宽度经验 NTK

独立比较经典与量子特征 Darcy PINN 的核、核演化和留出误差。复用 `v3/src` 模型与训练，不覆盖历史实验。

| 文件 | 作用 |
| --- | --- |
| `core.py` | 参数雅可比、输出/联合损失核、分组贡献与频率响应。 |
| `run_experiment.py` | 配对训练、中间检查点、原始数组和可追溯归档。 |
| `analyze_results.py` | 数值复核、场级统计、CSV和中文结果报告。 |
| `plot_results.py` | 从归档CSV绘制实验诊断图。 |
| `test_ntk.py` | 雅可比、损失等价性、分组加和及训练无干扰测试。 |
| `PROTOCOL.md` | 运行前固定的实验方案、判据和解释边界。 |
| `runs/` | 不可覆盖的实验数据、模型、源码快照和报告。 |

在仓库根目录运行：

```bash
PYTHONPATH=v3/src .venv/bin/python -m unittest discover -s v3/ntk -p 'test_*.py' -v
PYTHONPATH=v3/src .venv/bin/python -B v3/ntk/run_experiment.py --output v3/ntk/runs/research_20261016
PYTHONPATH=v3/src .venv/bin/python -B v3/ntk/analyze_results.py v3/ntk/runs/research_20261016
```

NTK变化量不是质量评分；本实验检验优势，不保证量子组胜出。
