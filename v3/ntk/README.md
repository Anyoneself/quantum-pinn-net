# V3 有限宽度经验 NTK

以六个新场的基础特征替换检验平均重建收益，经验核及核演化用于解释表示与训练的关系。复用`v3/src`，不覆盖历史实验。

研究归档中，量子组平均留出误差降低15.23%，六场中五场的场内平均更低。场级精确检验p=0.09375，探索性bootstrap差值区间为[-0.09615913,-0.00822420]；两者并列解释，结果为描述性支持，与三场主实验分别统计。

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

分析命令加 `--deep` 可重载首场首次重复的初始/最佳模型重新求导核对。原始数据位于各方法的 `epoch_*.npz`，汇总表和中文报告位于归档的 `analysis/`。绘图入口 `plot_results.py` 的 `--audit-tools` 指定包含 `audit_panel_alignment.py` 的绘图审计工具目录，渲染后另做PDF文字和碰撞检查。

主终点为留出渗透率误差，NTK承担事后诊断职责，成本与物理响应保留为次要评价。运行前判据和归档报告保持冻结；当前论文叙事见`v3/paper/`，写作规则见根目录`AGENTS.md`。
