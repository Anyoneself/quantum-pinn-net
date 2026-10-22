# Quantum PINN 渗透率场研究

仓库按研究版本组织：`v2/` 是经典计算实现的周期相位网络与空间补全，`v3/` 是 GSTools 合成场上的 Darcy 约束经典/量子特征 PINN。两者的模型、训练协议和指标不能直接合并比较。

远端保留代码、实验配置、统计汇总、报告和论文图件；模型权重、原始数据及全部`ppt/`目录保留本地。

当前 V3 入口：

```bash
.venv/bin/python -m pip install -e .
.venv/bin/qpinn run --smoke --output v3/experiments/new_smoke
.venv/bin/qpinn verify --output v3/experiments/new_smoke
```

不安装包时可使用 `PYTHONPATH=v3/src .venv/bin/python -m qpinn`。实验归档按版本保存在 `v1/experiments/`、`v2/experiments/` 和 `v3/experiments/`，不要覆盖既有目录。目录和文件职责见 [架构文档](docs/artecture.md)，原有详细项目说明保存在 [历史导航](docs/PROJECT_HISTORY.md)。
