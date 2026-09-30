# 文献核验与成果阅读记录

本记录对应[当前主稿](../PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)。共32篇去重研究论文，均已核对题录、阅读与引用判断有关的原文内容，并在主稿中实际引用。公开作者稿与正式发表版本按同一研究计数，不重复累计。阅读范围是下列实际章节、结果和图表说明，不表示逐字阅读了每篇全部附录。

核验来源包括Crossref、PMLR、NeurIPS、CVF、arXiv，以及已保存的两篇中文正式PDF。Crossref没有返回的中文题录，以原文首页核对。仅能访问摘要的旧候选文献不计入本次32篇；检索结果长度不作为全文判据。

文献原文与提取文本只用于阅读核对；以下成果说明由本次阅读概括。网络获取没有新增成批PDF文件。来源版本差异和适用边界逐条说明。

仓库发布说明：第三方PDF与原文提取文本不随Git提交；下列本地提取文本链接仅在保留阅读材料的工作区可用。远端保留原创阅读记录、题录核验信息和来源URL。新检出仓库需合法获取对应原文，才能运行要求本地阅读文件齐全的检查；阅读记录本身不替代原文证据。

## [1] geocalibration

SIENA M, RIVA M. Impact of geostatistical reconstruction approaches on model calibration for flow in highly heterogeneous aquifers[J]. Stochastic Environmental Research and Risk Assessment, 2020, 34(10): 1591-1606. DOI: 10.1007/s00477-020-01865-2.

- 阅读来源：https://link.springer.com/content/pdf/10.1007/s00477-020-01865-2.pdf；[本地提取文本](literature_sources/geocalibration.txt)。
- 核验记录：[来源1](literature_sources/geocalibration.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节研究区与地质统计重建、第3节连通性和标定结果、第4节结论；表1及图3—4的说明。
- 研究成果：对Bologna含水层分别用SISIM和T-PROGS生成岩相实现，再标定水力参数。两类重建的砾石连通性不同，对应的标定导水率和水头预测也不同。
- 本文引用用途：引言中说明仅比较场值或均值不足以判断流动响应。
- 边界与版本说明：研究对象是有地质和水头资料的含水层；不能将其现场证据移作本文固定数值场的验证。
- 状态：题录已核对；相关原文已读；已引用。

## [2] gstools

MÜLLER S, SCHÜLER L, ZECH A, et al. GSTools v1.3: A toolbox for geostatistical modelling in Python[J]. Geoscientific Model Development, 2022, 15(7): 3161-3182. DOI: 10.5194/gmd-15-3161-2022.

- 阅读来源：https://gmd.copernicus.org/articles/15/3161/2022/；[本地提取文本](literature_sources/gstools.txt)。
- 核验记录：[来源1](literature_sources/gstools.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：随机场生成和条件模拟说明、第4节示例、第5节结论；图15、图30说明。
- 研究成果：GSTools整合协方差建模、克里金、随机场生成和条件模拟。条件场示例中，观测附近的实现偏离参考场更小，显示空间相关建模与观测条件的联系。
- 本文引用用途：引言中交代地质统计空间恢复的技术路线。
- 边界与版本说明：本文没有运行克里金或GSTools，不将工具库论文误写为已实现的比较方法。
- 状态：题录已核对；相关原文已读；已引用。

## [3] deepsdf

PARK J J, FLORENCE P, STRAUB J, et al. DeepSDF: Learning continuous signed distance functions for shape representation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 165-174. DOI: 10.1109/CVPR.2019.00025.

- 阅读来源：https://arxiv.org/html/1901.05103；[本地提取文本](literature_sources/deepsdf.txt)。
- 核验记录：[来源1](literature_sources/deepsdf.json)；[来源2](literature_sources/deepsdf_candidates.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第6节形状表示、补全和潜在空间插值结果，第7节结论；表4和补全过程说明。
- 研究成果：共享解码器以位置和形状编码表示连续有符号距离；部分形状的补全需要依据观测优化潜在编码。作者同时报告了编码优化带来的推理成本。
- 本文引用用途：引言中解释对象嵌入与连续查询，并区分场内拟合和新对象直接预测。
- 边界与版本说明：原文处理三维形状，本文处理渗透率；引用的是表示与适配方式，不转用原文的精度优势。
- 状态：题录已核对；相关原文已读；已引用。

## [4] spectral

RAHAMAN N, BARATIN A, ARPIT D, et al. On the spectral bias of neural networks[C]//Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019, 97: 5301-5310.

- 阅读来源：https://arxiv.org/html/1806.08734；[本地提取文本](literature_sources/spectral.txt)。
- 核验记录：[来源1](literature_sources/spectral.json)；[来源2](literature_sources/spectral_proceedings.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第1节问题定义、合成实验1和对应讨论、第6节结论。
- 研究成果：对深层ReLU网络进行Fourier分析及合成拟合实验，观察到低频优先的学习行为；流形几何还影响高频函数的表达难度。
- 本文引用用途：引言中限定频谱偏置的适用条件，说明有限预算细节拟合的动机。
- 边界与版本说明：不能将ReLU网络的分析无条件推广为本文门控残差网络的理论证明。
- 状态：题录已核对；相关原文已读；已引用。

## [5] fourier

TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C]//Advances in Neural Information Processing Systems. 2020, 33: 7537-7547.

- 阅读来源：https://arxiv.org/html/2006.10739；[本地提取文本](literature_sources/fourier.txt)。
- 核验记录：[来源1](literature_sources/fourier.json)；[来源2](literature_sources/fourier_proceedings.json)；[来源3](literature_sources/fourier_bibtex.txt)。
- 访问日期：2026-09-30。
- 实际阅读范围：第6.1节编码对照、第6.2节直接/间接回归及留出评价、第7节结论。
- 研究成果：比较基本周期映射、位置编码和Gaussian随机Fourier特征；在图像、形状和成像任务中，调整频率尺度能改善坐标MLP拟合。频率尺度由留出信号调节。
- 本文引用用途：引言说明输入编码路线，讨论频率尺度和比较前提。
- 边界与版本说明：本文固定Fourier对照不等于复现原文最优超参数搜索，不能声称超过原始Fourier特征方法的全部配置。
- 状态：题录已核对；相关原文已读；已引用。

## [6] siren

SITZMANN V, MARTEL J N P, BERGMAN A W, et al. Implicit neural representations with periodic activation functions[C]//Advances in Neural Information Processing Systems. 2020, 33: 7462-7473.

- 阅读来源：https://arxiv.org/html/2006.09661；[本地提取文本](literature_sources/siren.txt)。
- 核验记录：[来源1](literature_sources/siren.json)；[来源2](literature_sources/siren_proceedings.json)；[来源3](literature_sources/siren_bibtex.txt)。
- 访问日期：2026-09-30。
- 实际阅读范围：第4节Poisson、Eikonal等导数监督任务，第5节讨论与结论。
- 研究成果：周期激活配合专门初始化可表达信号及其导数，并用于边值问题和隐式几何表示。原文对整个周期网络的激活统计作了分析。
- 本文引用用途：引言和方法中说明正弦激活来源，并声明当前网络不是完整SIREN配置。
- 边界与版本说明：只在首层使用正弦且初始化不同，不能继承原文全部性质或把本文对照称为SIREN。
- 状态：题录已核对；相关原文已读；已引用。

## [7] nerf

MILDENHALL B, SRINIVASAN P P, TANCIK M, et al. NeRF: Representing scenes as neural radiance fields for view synthesis[C]//Computer Vision – ECCV 2020. 2020: 405-421. DOI: 10.1007/978-3-030-58452-8_24.

- 阅读来源：https://arxiv.org/html/2003.08934；[本地提取文本](literature_sources/nerf.txt)。
- 核验记录：[来源1](literature_sources/nerf.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第6节合成/真实场景结果、第6.4节消融设置和讨论、第7节结论。
- 研究成果：逐场景优化位置与方向查询的连续辐射场，通过体渲染拟合图像，并分析质量、训练时间与存储之间的取舍。
- 本文引用用途：引言中说明位置编码及逐对象连续表示的已有应用。
- 边界与版本说明：视图合成并非渗透率重建，只借鉴表示概念，不引入渲染模块或迁移精度结论。
- 状态：题录已核对；相关原文已读；已引用。

## [8] bacon

LINDELL D B, VAN VEEN D, PARK J J, et al. BACON: Band-limited coordinate networks for multiscale scene representation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 16252-16262.

- 阅读来源：https://arxiv.org/html/2112.04645；[本地提取文本](literature_sources/bacon.txt)。
- 核验记录：[来源1](literature_sources/bacon.json)；[来源2](literature_sources/bacon_proceedings.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第3节频带结构、第4.1节图像拟合与编码对照、第5节结论和局限。
- 研究成果：基于乘法滤波结构控制中间输出的频带，支持多尺度信号表示；图像拟合、神经渲染和形状拟合检验了其尺度行为。
- 本文引用用途：引言中区分频带控制与一般周期特征。
- 边界与版本说明：本文没有BACON式频谱界限，不能声称对未观测位置已有带限保证。
- 状态：题录已核对；相关原文已读；已引用。

## [9] eigenvector

WANG S, WANG H, PERDIKARIS P. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 384: 113938. DOI: 10.1016/j.cma.2021.113938.

- 阅读来源：https://arxiv.org/html/2012.10047；[本地提取文本](literature_sources/eigenvector.txt)。
- 核验记录：[来源1](literature_sources/eigenvector.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第3节NTK分析、第4节多尺度Poisson等实验、第5节讨论。
- 研究成果：Fourier特征映射影响NTK特征向量频率；多尺度编码改善相应PDE算例，但频率数目和尺度需要与目标变化匹配。
- 本文引用用途：引言中说明频率设置是比较前提，不能仅凭编码名称判断表达效果。
- 边界与版本说明：固定频率对照不代表所有Fourier网络最优配置，本文未作NTK谱验证。
- 状态：题录已核对；相关原文已读；已引用。

## [10] pinn

RAISSI M, PERDIKARIS P, KARNIADAKIS G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707. DOI: 10.1016/j.jcp.2018.10.045.

- 阅读来源：https://arxiv.org/html/1711.10561；[本地提取文本](literature_sources/pinn.txt)。
- 核验记录：[来源1](literature_sources/pinn.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：对应早期作者稿Part I的第1节信息条件、连续时间PINN定义、Allen–Cahn算例及第4节总结；正式题录由Crossref核对。
- 研究成果：以神经网络近似解函数，将观测和微分方程残差共同纳入优化；Allen–Cahn算例展示了数据和时间离散结构的结合。作者指出它不应被简单视为成熟数值方法的全面替代。
- 本文引用用途：引言中给出PINN的方程约束含义。
- 边界与版本说明：阅读源是2017年Part I作者稿，题录引用2019年合并发表论文；此处仅引用二者共同的正问题框架，不声称逐页核对了正式合并版本或阅读了Part II的全部反演结果。
- 状态：题录已核对；相关原文已读；已引用。

## [11] dgm

SIRIGNANO J, SPILIOPOULOS K. DGM: A deep learning algorithm for solving partial differential equations[J]. Journal of Computational Physics, 2018, 375: 1339-1364. DOI: 10.1016/j.jcp.2018.08.029.

- 阅读来源：https://arxiv.org/html/1708.07469；[本地提取文本](literature_sources/dgm.txt)。
- 核验记录：[来源1](literature_sources/dgm.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：随机采样求导估计的成本讨论、第4节高维自由边界实验、第8节结论。
- 研究成果：用随机时空样本训练神经网络以满足微分算子和初边值条件，在高维自由边界和HJB问题上验证；同时指出高振荡解可能更困难。
- 本文引用用途：引言中介绍基于微分算子残差的神经求解路线。
- 边界与版本说明：文中收敛论证与数值案例有明确方程范围，不构成本研究统计先验的理论依据。
- 状态：题录已核对；相关原文已读；已引用。

## [12] ritz

E W, YU B. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems[J]. Communications in Mathematics and Statistics, 2018, 6(1): 1-12. DOI: 10.1007/s40304-018-0127-z.

- 阅读来源：https://arxiv.org/html/1710.00211；[本地提取文本](literature_sources/ritz.txt)。
- 核验记录：[来源1](literature_sources/ritz.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第3节Poisson及特征值数值结果、第4节讨论。
- 研究成果：以深度网络表示试探函数并最小化变分能量；数值示例展示可行性，也报告非凸优化、边界条件处理和高维特征值误差方面的限制。
- 本文引用用途：引言区分强形式残差与变分能量约束。
- 边界与版本说明：本文没有构造Darcy能量训练项，不能将经验平滑损失称为Deep Ritz。
- 状态：题录已核对；相关原文已读；已引用。

## [13] review

CUOMO S, DI COLA V S, GIAMPAOLO F, et al. Scientific machine learning through physics-informed neural networks: Where we are and what's next[J]. Journal of Scientific Computing, 2022, 92(3): 88. DOI: 10.1007/s10915-022-01939-z.

- 阅读来源：https://arxiv.org/html/2201.05624；[本地提取文本](literature_sources/review.txt)。
- 核验记录：[来源1](literature_sources/review.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第5节挑战及第6节综述结论，关于网络、损失构建、软/硬约束和方程配置点的归纳。
- 研究成果：系统整理PINN的结构、训练和约束变体，区分方程约束、变分形式、软边界与硬边界处理，并总结多尺度和优化问题。
- 本文引用用途：引言为不同物理信息注入形式提供综述背景。
- 边界与版本说明：综述用于分类，具体方法效果优先引用相应原始研究。
- 状态：题录已核对；相关原文已读；已引用。

## [14] fdpinn

宋家豪, 曹文博, 张伟伟. FD-PINN: 频域物理信息神经网络[J]. 力学学报, 2023, 55(5): 1195-1205. DOI: 10.6052/0459-1879-23-169.

- 阅读来源：tmp/pdfs/chinese_core_references/fd_pinn_extracted.txt；[本地提取文本](../../tmp/pdfs/chinese_core_references/fd_pinn_extracted.txt)。
- 核验记录：[来源1](literature_sources/fdpinn.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：本地正式PDF第1195页题录；方法及第2.1节热传导对比、表1、图4—5说明，第3节结论。
- 研究成果：对周期方向作离散Fourier变换，将PDE转为低维频域方程组。热传导等算例中减少输入样本和训练成本，并比较求解误差与频率截断影响。
- 本文引用用途：引言区分对控制方程作频域变换与本文的输入相位特征。
- 边界与版本说明：不把FD-PINN的降维、效率和精度数字当作本文的结果。中文核心期刊范文的组织特征仅用于写作规范。
- 状态：题录已核对；相关原文已读；已引用。

## [15] gradient

WANG S, TENG Y, PERDIKARIS P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J]. SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081. DOI: 10.1137/20M1318043.

- 阅读来源：https://arxiv.org/html/2001.04536；[本地提取文本](literature_sources/gradient.txt)。
- 核验记录：[来源1](literature_sources/gradient.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节框架、第3节四种配置的比较设计、腔流图18说明、第4节总结。
- 研究成果：分析PINN的刚性梯度流及复合损失梯度失衡，以梯度统计调节训练权重，并比较专用网络结构。
- 本文引用用途：引言和讨论中解释损失数值大小不等于梯度贡献。
- 边界与版本说明：本文倒数损失规则不读取梯度统计，不能把它当作该文梯度平衡算法。
- 状态：题录已核对；相关原文已读；已引用。

## [16] ntk

WANG S, YU X, PERDIKARIS P. When and why PINNs fail to train: A neural tangent kernel perspective[J]. Journal of Computational Physics, 2022, 449: 110768. DOI: 10.1016/j.jcp.2021.110768.

- 阅读来源：https://arxiv.org/html/2007.14527；[本地提取文本](literature_sources/ntk.txt)。
- 核验记录：[来源1](literature_sources/ntk.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第7节数值实验的设置与Poisson检验，第8节Discussion。
- 研究成果：在相应宽网络假设下研究PINN的NTK，指出复合损失分量收敛速率不同；动态缩放可调整平均速率，但不能消除所有频谱偏置。
- 本文引用用途：引言区分损失尺度、收敛速度和表示能力。
- 边界与版本说明：理论假设不等同于本文有限宽残差网络的实际条件。
- 状态：题录已核对；相关原文已读；已引用。

## [17] failure

KRISHNAPRIYAN A S, GHOLAMI A, ZHE S, et al. Characterizing possible failure modes in physics-informed neural networks[C]//Advances in Neural Information Processing Systems. 2021, 34: 26548-26560.

- 阅读来源：https://arxiv.org/html/2109.01050；[本地提取文本](literature_sources/failure.txt)。
- 核验记录：[来源1](literature_sources/failure.json)；[来源2](literature_sources/failure_bibtex.txt)。
- 访问日期：2026-09-30。
- 实际阅读范围：实验设置、反应扩散seq2seq结果图6说明、第6节结论。
- 研究成果：在对流、反应和扩散问题中，标准软约束PINN随问题难度增加可失效；损失地形与优化困难是原因之一，课程或分步训练能改善部分算例。
- 本文引用用途：引言说明增加约束并不保证解的准确性。
- 边界与版本说明：不能由这些特定PDE失败推广为所有PINN无效，也不把其失败机制等同于本文倒数权重问题。
- 状态：题录已核对；相关原文已读；已引用。

## [18] sapinn

MCCLENNY L D, BRAGA-NETO U M. Self-adaptive physics-informed neural networks[J]. Journal of Computational Physics, 2023, 474: 111722. DOI: 10.1016/j.jcp.2022.111722.

- 阅读来源：https://arxiv.org/html/2009.04544；[本地提取文本](literature_sources/sapinn.txt)。
- 核验记录：[来源1](literature_sources/sapinn.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：Numerical Examples部分的比较指标、NTK结果图17说明、第7节结论。
- 研究成果：给训练位置配置可训练权重，以软注意力和惩罚优化解释其作用；多个PDE算例及NTK分析显示这种逐点机制与整项缩放不同。
- 本文引用用途：引言中说明逐点可训练权重的已有研究。
- 边界与版本说明：原文权重通过优化更新；本文残差递推权重和全局倒数评分均不是原算法。阅读为作者公开稿，与2023正式题录合并计一篇。
- 状态：题录已核对；相关原文已读；已引用。

## [19] rba

ANAGNOSTOPOULOS S J, TOSCANO J D, STERGIOPULOS N, et al. Residual-based attention in physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116805. DOI: 10.1016/j.cma.2024.116805.

- 阅读来源：https://arxiv.org/pdf/2307.00379；[本地提取文本](literature_sources/rba.txt)。
- 核验记录：[来源1](literature_sources/rba.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节PINN及残差权重方法、第3.1节Allen–Cahn结果和表1说明；公开PDF。
- 研究成果：根据局部残差递推调整训练点贡献；Allen–Cahn基准将RBA施加于方程残差项，并与其他PINN训练配置比较。
- 本文引用用途：引言中说明残差驱动注意力，限定本文适配后的逐点加权含义。
- 边界与版本说明：原文方程残差权重与本文数据残差权重的对象不同，不能直接移用性能结论。
- 状态：题录已核对；相关原文已读；已引用。

## [20] annulus

徐宝昌, 张学智, 王雅欣, 等. 用于两相流环空压力预测的自适应物理信息神经网络模型[J]. 石油学报, 2023, 44(3): 545-555. DOI: 10.7623/syxb202303012.

- 阅读来源：tmp/pdfs/chinese_core_references/syxb_extracted.txt；[本地提取文本](../../tmp/pdfs/chinese_core_references/syxb_extracted.txt)。
- 核验记录：[来源1](literature_sources/annulus.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：正式PDF第545页题录，第2.3—2.4节、第3节压力预测、表3及图7—10说明，第4节结论。
- 研究成果：将两相流方程、初边值条件及数据拟合结合，用可训练权重关注困难位置。原文为避免噪声重复拟合而不对样本损失加权，并在不同气侵阶段评价压力预测。
- 本文引用用途：引言中支撑自适应权重需与观测质量相匹配的判断。
- 边界与版本说明：有限差分生成数据与实际井筒验证不能混同；不把该文压力预测结果当成本研究的应用收益。
- 状态：题录已核对；相关原文已读；已引用。

## [21] sampling

WU C, ZHU M, TAN Q, et al. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2023, 403: 115671. DOI: 10.1016/j.cma.2022.115671.

- 阅读来源：https://arxiv.org/html/2207.10289；[本地提取文本](literature_sources/sampling.txt)。
- 核验记录：[来源1](literature_sources/sampling.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节方法、第3节六个正反问题比较设置、第4节结论及建议。
- 研究成果：比较多种非自适应和残差自适应采样，并提出RAD和RAR-D；高残差区域点分布的调整在其算例中改善精度与采样效率。
- 本文引用用途：引言中区分改变配置点分布与固定观测点加权。
- 边界与版本说明：本文没有通过这些方法新增方程配置点，不把已有标签的加权称为主动采样。
- 状态：题录已核对；相关原文已读；已引用。

## [22] deepxde

LU L, MENG X, MAO Z, et al. DeepXDE: A deep learning library for solving differential equations[J]. SIAM Review, 2021, 63(1): 208-228. DOI: 10.1137/19M1274067.

- 阅读来源：https://arxiv.org/html/1907.04502；[本地提取文本](literature_sources/deepxde.txt)。
- 核验记录：[来源1](literature_sources/deepxde.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第1节方法背景、Volterra积分微分方程算例及图10说明、第5节总结。
- 研究成果：DeepXDE将自动微分、几何与约束接口组织为PINN求解工具，并提出残差自适应细化；原文也指出其正问题求解速度与传统有限元之间的差距。
- 本文引用用途：引言中作为残差采样/细化路线的原始来源。
- 边界与版本说明：本文并未使用DeepXDE执行实验；此引用不能代替本地实现记录。
- 状态：题录已核对；相关原文已读；已引用。

## [23] guide

WANG S, SANKARAN S, WANG H, et al. An expert's guide to training physics-informed neural networks[EB/OL]. arXiv:2308.08468, 2023[2026-09-30]. https://arxiv.org/abs/2308.08468.

- 阅读来源：https://arxiv.org/html/2308.08468；[本地提取文本](literature_sources/guide.txt)。
- 核验记录：[来源1](literature_sources/guide.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第7节Results中的消融与最优配置区分、圆柱绕流评价说明，第8节结论。
- 研究成果：整合无量纲化、Fourier特征、权重平衡和因果训练，并区分共同超参数消融与经搜索得到的最优配置。
- 本文引用用途：引言中支持强基线、明确比较条件和组件消融的设计原则。
- 边界与版本说明：引用arXiv公开稿并标明电子文献；不把综合训练指南作为本文尚未采用机制的效果证据。
- 状态：题录已核对；相关原文已读；已引用。

## [24] scipy

VIRTANEN P, GOMMERS R, OLIPHANT T E, et al. SciPy 1.0: Fundamental algorithms for scientific computing in Python[J]. Nature Methods, 2020, 17(3): 261-272. DOI: 10.1038/s41592-019-0686-2.

- 阅读来源：https://arxiv.org/html/1907.10121；[本地提取文本](literature_sources/scipy.txt)。
- 核验记录：[来源1](literature_sources/scipy.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：插值、稀疏线性代数和spatial模块介绍，Sparse matrices、cKDTree及Polynomial interpolators部分。
- 研究成果：介绍SciPy的插值和稀疏矩阵计算体系，并报告稀疏格式处理、近邻结构及多项式插值接口方面的改进。
- 本文引用用途：方法中注明数值计算所用的SciPy软件论文。
- 边界与版本说明：SciPy 1.0论文不是当前RBFInterpolator接口的逐项说明；具体版本与核参数另在补充材料记录，不据软件论文证明基线最优。
- 状态：题录已核对；相关原文已读；已引用。

## [25] activation

JAGTAP A D, KAWAGUCHI K, KARNIADAKIS G E. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks[J]. Journal of Computational Physics, 2020, 404: 109136. DOI: 10.1016/j.jcp.2019.109136.

- 阅读来源：https://arxiv.org/html/1906.01170；[本地提取文本](literature_sources/activation.txt)。
- 核验记录：[来源1](literature_sources/activation.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节可训练激活尺度定义、第3节函数和PDE正反问题设置、第4节结论。
- 研究成果：在激活中引入可训练尺度，比较平滑/不连续函数以及Burgers、Klein–Gordon和Helmholtz算例，观察到收敛与误差改善。
- 本文引用用途：方法中说明激活尺度会影响优化，并限定本文固定频率首层的比较。
- 边界与版本说明：未在本项目运行自适应激活，不将该文收益写为本地实验结果。
- 状态：题录已核对；相关原文已读；已引用。

## [26] localactivation

JAGTAP A D, KAWAGUCHI K, KARNIADAKIS G E. Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks[J]. Proceedings of the Royal Society A, 2020, 476(2239): 20200334. DOI: 10.1098/rspa.2020.0334.

- 阅读来源：https://arxiv.org/html/1909.12228；[本地提取文本](literature_sources/localactivation.txt)。
- 核验记录：[来源1](literature_sources/localactivation.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：条件数与收敛图2说明、第4节结论及理论适用条件。
- 研究成果：提出逐层和逐神经元的局部激活尺度，并以斜率恢复项促进训练；函数逼近、反问题和标准学习任务检验了其作用。
- 本文引用用途：方法中补充激活参数化的已有进展，说明当前首层比较的范围。
- 边界与版本说明：未复现该算法，不把其条件数或收敛保证赋予本文网络。
- 状态：题录已核对；相关原文已读；已引用。

## [27] deeponet

LU L, JIN P, PANG G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators[J]. Nature Machine Intelligence, 2021, 3(3): 218-229. DOI: 10.1038/s42256-021-00302-5.

- 阅读来源：https://arxiv.org/html/1910.03193；[本地提取文本](literature_sources/deeponet.txt)。
- 核验记录：[来源1](literature_sources/deeponet.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2.1节输入函数与位置的分支编码、数据规模收敛图10说明、第5节结论。
- 研究成果：DeepONet分别编码输入函数观测与查询位置，再合并输出；研究输入函数复杂性、传感器和训练样本数对算子泛化的影响。
- 本文引用用途：讨论中区分已登记场补全与跨输入函数的算子学习。
- 边界与版本说明：本文场标识嵌入没有实现DeepONet式输入函数编码，不能将场内留点解释为算子泛化。
- 状态：题录已核对；相关原文已读；已引用。

## [28] fno

LI Z, KOVACHKI N, AZIZZADENESHELI K, et al. Fourier neural operator for parametric partial differential equations[C]//International Conference on Learning Representations. 2021. https://arxiv.org/abs/2010.08895.

- 阅读来源：https://arxiv.org/html/2010.08895；[本地提取文本](literature_sources/fno.txt)。
- 核验记录：[来源1](literature_sources/fno.json)；[来源2](literature_sources/fno_first_page.txt)。
- 访问日期：2026-09-30。
- 实际阅读范围：第5节Burgers、Darcy和Navier–Stokes实验设置、频谱与非周期边界说明，第6节讨论。
- 研究成果：以Fourier空间积分算子学习参数函数到解函数的映射，分别保留训练和测试实例；非线性层参与恢复高频，数据生成与训练有独立成本。
- 本文引用用途：讨论中说明与算子学习的任务和信息条件差别。
- 边界与版本说明：FNO不是本文固定Fourier输入编码MLP；本文也未训练FNO，不能直接比较两个任务的误差。
- 状态：题录已核对；相关原文已读；已引用。

## [29] gpinn

YU J, LU L, MENG X, et al. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems[J]. Computer Methods in Applied Mechanics and Engineering, 2022, 393: 114823. DOI: 10.1016/j.cma.2022.114823.

- 阅读来源：https://arxiv.org/html/2111.02801；[本地提取文本](literature_sources/gpinn.txt)。
- 核验记录：[来源1](literature_sources/gpinn.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第2节残差梯度增强方法、第3节正反问题设置、第4节结论及成本讨论。
- 研究成果：在损失中加入方程残差导数，改善解、导数及反演参数精度；结合RAR处理陡梯度，但高阶自动微分增加计算成本。
- 本文引用用途：讨论中说明更强约束需要正确的局部方程与成本比较。
- 边界与版本说明：本文不使用残差导数损失，也不把统计平滑项等同于gPINN。
- 状态：题录已核对；相关原文已读；已引用。

## [30] hard

SUKUMAR N, SRIVASTAVA A. Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2022, 389: 114333. DOI: 10.1016/j.cma.2021.114333.

- 阅读来源：https://arxiv.org/pdf/2104.08426；[本地提取文本](literature_sources/hard.txt)。
- 核验记录：[来源1](literature_sources/hard.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第1节构造思路、第9节边值问题比较设计、第12节结论；正式题录与公开PDF首页核对。
- 研究成果：用距离函数和跨有限插值构造满足边界条件的试探解，减少边界损失竞争；强调总损失大小并不能替代解误差评价。
- 本文引用用途：讨论中区分已知边界的硬约束和本文边缘均值正则项。
- 边界与版本说明：已修正候选阶段误配的另一篇hPINN题录。本文没有使用距离函数边界构造。
- 状态：题录已核对；相关原文已读；已引用。

## [31] wight

WIGHT C L, ZHAO J. Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks[J]. Communications in Computational Physics, 2021, 29(3): 930-954. DOI: 10.4208/cicp.oa-2020-0086.

- 阅读来源：https://arxiv.org/html/2007.04542；[本地提取文本](literature_sources/wight.txt)。
- 核验记录：[来源1](literature_sources/wight.json)；[来源2](literature_sources/wight_candidates.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第3节Allen–Cahn、Cahn–Hilliard数值设置，第4节时间/空间自适应总结。
- 研究成果：将空间残差采样与时间域分段策略用于相场方程；时间分段可降低单次学习难度，但前段误差可能向后段传播。
- 本文引用用途：讨论中说明自适应策略的作用依赖于问题结构。
- 边界与版本说明：静态渗透率补全没有时间推进；不能把时间域效果移作本研究自适应模块的正面证据。
- 状态：题录已核对；相关原文已读；已引用。

## [32] causal

WANG S, SANKARAN S, PERDIKARIS P. Respecting causality for training physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116813. DOI: 10.1016/j.cma.2024.116813.

- 阅读来源：https://arxiv.org/html/2203.07404；[本地提取文本](literature_sources/causal.txt)。
- 核验记录：[来源1](literature_sources/causal.json)；[来源2](literature_sources/causal_candidates.json)。
- 访问日期：2026-09-30。
- 实际阅读范围：第5节混沌动力系统和流动实验设置、第6节Discussion；公开稿题名含is all you need，正式发表题名已核对。
- 研究成果：针对时间依赖PINN在初值尚未拟合时优先减小后期残差的问题，按早期状态拟合程度调整时间残差权重，改善所研究动力系统的训练。
- 本文引用用途：讨论中强调训练权重需遵循任务的信息依赖。
- 边界与版本说明：本文为静态补全，不存在相同时间因果顺序；只讨论适用条件，不声称采用了因果训练。
- 状态：题录已核对；相关原文已读；已引用。

## 未计入的候选

- Dealing with spatial heterogeneity：本次出版商页面只有订阅预览。保留检索痕迹，不把摘要和参考文献列表视为全文，已从主稿参考文献移除。
- An adaptive inverse-distance weighting spatial interpolation technique：题录可核验，但本次没有得到可读正文，已从主稿参考文献移除。IDW作为本文明确给出的距离加权计算方法说明，不据此虚构原文阅读。
- 初始hard候选误将另一篇逆向设计论文的DOI与距离函数论文配对，已按实际PDF首页及Crossref改正为Sukumar和Srivastava的研究，错误配对不进入主稿。
