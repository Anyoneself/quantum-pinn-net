# 项目架构

根目录 `pyproject.toml` 安装 V3 的 `v3/src` 包；`v2/requirements.txt` 与 `v3/requirements-v3.lock` 分别对应历史路线和 V3 依赖快照。`AGENTS.md` 为协作约定，`README.md` 为最短项目入口。`docs/PROJECT_HISTORY.md` 保存重构前的详细导航，其中旧路径仅作历史记录。

## V3：Darcy 逆问题

`qpinn` 命令调用 `v3/src/qpinn/cli.py`，由 `v3/src/experiment/runner.py` 安排场、种子和方法组合；`v3/src/dataset/cases.py` 生成场、压力参考与观测划分；`v3/src/models/baseline_model.py` 和 `qmodel.py` 分别定义经典与量子模型，共用 `darcy_pinn.py` 的渗透率、压力网络；`v3/src/metrics/physics.py` 计算通量、Darcy 残差和训练目标；`v3/src/qpinn/training.py` 负责拟合、选模和评价，`v3/src/qpinn/strategies.py` 提供 RBA 与 MC 特征筛选。

| 文件或目录 | 作用 |
| --- | --- |
| `v3/src/qpinn/__init__.py`、`__main__.py` | 包标识与 `python -m qpinn` 入口。 |
| `v3/src/qpinn/cli.py` | 参数解析和配置组装。 |
| `v3/src/qpinn/training.py`、`strategies.py` | 训练/评价和增强策略。 |
| `v3/src/dataset/cases.py` | 合成场、FVM 压力参考、观测划分；同目录 `__init__.py` 为包标识。 |
| `v3/src/models/baseline_model.py`、`qmodel.py` | V3 经典对照和量子特征模型，分别定义特征块及可直接构造的 Darcy 模型。 |
| `v3/src/models/darcy_pinn.py`、`quantum_pinn.py` | 共用网络与配置、旧导入路径的兼容入口；同目录 `__init__.py` 为包标识。 |
| `v3/src/metrics/physics.py` | 物理损失与验证评分；同目录 `__init__.py` 为包标识。 |
| `v3/src/experiment/runner.py` | 写入运行清单、模型、源代码快照与配对汇总，校验归档；同目录 `__init__.py` 为包标识。 |
| `v3/src/tests/test_qpinn_v3.py` | 数据、梯度、策略和归档协议测试。 |
| `v3/src/scripts/analyze_manuscript_v3.py`、`draw_v3_architecture.py` | 历史 V3（压力监督）图件与架构绘制。 |
| `v3/src/scripts/analyze_v3_gstools_results.py`、`audit_v3_gstools_results.py` | 当前 V3 图源分析及渲染审计。 |
| `v3/src/scripts/analyze_v3_reconstruction_budget.py` | 延长预算的独立对照。 |
| `v3/src/scripts/verify_manuscript_v3.py` | V3 文稿核验入口。 |
| `v3/README.md` | V3最短导航及当前叙事规则入口。 |
| `v3/docs/QUANTUM_PINN_V3.md` | 模型、物理问题、受控实验协议及证据分工。 |
| `v3/docs/V3_PROJECT_EXPLANATION.md`、`V3_RECONSTRUCTION_BUDGET_PROTOCOL.md` | 项目解释与预算诊断协议。 |
| `v3/ntk/core.py`、`run_experiment.py` | 有限宽度经验NTK、配对训练、中间检查点及独立实验归档。 |
| `v3/ntk/analyze_results.py`、`plot_results.py`、`test_ntk.py` | NTK数值复核、场级统计、诊断绘图与测试；`README.md`、`PROTOCOL.md`说明入口和固定方案，`runs/`保存新数据。 |
| `v3/ntk/run_supplement.py`、`SUPPLEMENT_PROTOCOL.md` | 固定四组新场复验和四种单因素条件，批量训练、逐组深度核验及24场主要统计。 |
| `v3/paper/QUANTUM_DARCY_PINN_MANUSCRIPT_V3.md`、`V3_MANUSCRIPT_OUTLINE.md` | 以平均重建收益为主线的主稿和大纲，三场主对照、预算诊断及六场补充证据分别报告。 |
| `v3/paper/materical/`、`v3/paper/history/` | 补充材料、阅读/核验记录和重写前稿件。`materical` 延续原目录拼写。 |
| `v3/paper/materical/tools/verify_v3_gstools_manuscript.py` | 稿件数值和图件来源检查。 |
| `v3/paper/figures/manuscript_v3_gstools/` | 当前 GSTools 图、图源与渲染检查。 |
| `v3/paper/figures/manuscript_v3_budget5000/`、`manuscript_v3/` | 延长预算图和首轮历史图。 |
| `v3/ppt/quantum_darcy_pinn_v3_deck/` | V3 汇报大纲、讲稿、逐页图和配置；2026年10月19日文字已更新，原图片和生成配置尚未同步。 |

## V2：场内空间补全

| 文件或目录 | 作用 |
| --- | --- |
| `v2/src/models/baseline_model/standard_pinn.py`、`qmodel/quantum_pinn.py` | V2 实际使用的 Fourier 对照与经典周期相位网络；各目录 `README.md` 说明模型。 |
| `v2/src/dataset/physics_constraints.py`、`darcy_physics.py` | 工程条件约束与历史 Darcy 诊断；`processed/` 为固定输入场、条件和观测掩码。 |
| `v2/src/experiment/run_pinn_experiment.py`、`run_supplement_v2.py` | 原始运行与 V2 补充实验训练入口。 |
| `v2/src/scripts/aggregate_multiseed_results.py`、`analyze_manuscript_evidence.py` | 历史多种子汇总与证据分析。 |
| `v2/src/scripts/analyze_supplement_v2.py`、`verify_supplement_v2.py`、`audit_manuscript_v2_figures.py` | 补充实验图源、数值检查与图件审核。 |
| `v2/src/scripts/generate_extended_paper_figures.py`、`generate_manuscript_rmse_distribution.py`、`generate_manuscript_spatial_figure.py` | 指标、RMSE 分布与空间场绘图。 |
| `v2/src/scripts/generate_quantum_pinn_architecture.py`、`generate_training_workflow.py` | V2 网络结构及训练流程图。 |
| `v2/src/scripts/build_chinese_core_docx.py`、`build_manuscript_v2_docx.py`、`build_submission_docx.py` | 各版本 Word 排版导出。 |
| `v2/src/scripts/render_chinese_core_equations.cjs`、`render_manuscript_v2_math.cjs` | MathJax 公式资源。 |
| `v2/src/scripts/validate_manuscript.py` | 旧稿静态检查。 |
| `v2/src/tests/test_darcy_physics.py`、`test_experiment_protocol.py`、`test_manuscript_curves.py`、`test_supplement_v2.py` | 物理、实验划分、图源和补充协议测试。 |
| `v2/src/metrics/README.md` | 指标目前由实验和分析脚本计算，说明其位置而不复制实现。 |
| `v2/paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md` | V2 主稿；无版本后缀的同名文件是早期稿。 |
| `v2/paper/materical/` | 补充材料、来源题录、写作记录；`tools/` 是文献抓取、引用整理和需求核验脚本。 |
| `v2/paper/figures/architecture/`、`manuscript_v2/`、`manuscript/` | V2 结构图、主稿结果图和早期稿图。 |
| `v2/paper/output/` | 历史 DOCX/PDF 导出。 |
| `v2/ppt/defense_ppt_rba_qpinn/`、`quantum_pinn_research_deck/` | 两套 V2 汇报资料及素材。 |

## 实验归档与 V1

实验结果分别保留在各版本的 `v1/experiments/`、`v2/experiments/` 和 `v3/experiments/`。仅移动归档目录，不重写已有 `manifest.json`、哈希和运行来源字段。

Git 远端发布轻量版本：保留代码、冻结配置与来源清单、逐次和场级统计表、报告及论文图件。模型权重、NumPy/HDF5原始数组、逐步训练日志、NTK原始检查点和高维诊断表、各版本完整`ppt/`目录及PPT导出由根目录`.gitignore`排除，完整文件仍保留本地。历史清单及核验报告不追溯修改；其中的原始文件哈希与`passed`结果对应当时的完整本地归档，远端轻量副本不能单独执行全量原始数据核验。移除跟踪不改写旧提交历史。

| 文件或目录 | 作用 |
| --- | --- |
| `v1/experiments/rba_qpinn_submission_v1/` | V1 多种子训练、统计汇总和固定输入；目录内 `README.md` 保留原运行说明。 |
| `v2/experiments/supplement_v2/` | V2 空间补全补充实验，含清单、观测划分、预测和各方法指标；目录内 `README.md` 保留原运行说明。 |
| `v3/experiments/quantum_pinn_v3_gstools_konly_20261007/` | V3 当前 GSTools 主实验归档。 |
| `v3/experiments/quantum_pinn_v3_gstools_konly_smoke_20261007/` | V3 主协议的流程检查。 |
| `v3/experiments/quantum_pinn_v3_initial_20261007/` | V3 旧压力监督协议的历史结果。 |
| `v3/experiments/quantum_pinn_v3_reconstruction_budget_20261007/` | V3 重建预算延长诊断，独立于主实验。 |

V3 归档内的 `source/` 是当次源码及环境快照，`manifest.json` 是配置和哈希，`REPORT.md`/`summary.json` 是报告和汇总。`field_*/reference.json|npz` 存参考场，`seed_*/observations.json|npz` 存划分，方法目录下的 `model.pt`、`predictions.npz`、`history.json`、`screening.json`、`result.json` 分别存权重、预测、训练、筛选和单次指标。V1 早期根目录图表及 `runs/` 中的模型/预测放在 `v1/experiment/figures/`，不作为 V2/V3 正式论文图件。

论文图件保留被正文引用的图及对应图源和核验记录，包括不利于方法的对照。`.png`、`.svg`、`.pdf` 是图像导出，`*_source.csv` 和来源/审计 JSON 记录数值与排版检查。`ppt/` 的 `outline.md`、`speech.md`、`deck_spec.json`、`slide_jobs.json`、`slide_run_state.json` 分别是大纲、讲稿、规格、任务与状态；`prompts/`、`origin_image/`、`drafts/` 为逐页制作素材。

`tmp/` 为本地渲染和试验临时文件。历史报告中的旧相对路径及原脚本哈希代表当时的冻结版本；迁移后的脚本和稿件需要分别重新校验，不能用旧的 `passed` 字样代替新布局的核验。
