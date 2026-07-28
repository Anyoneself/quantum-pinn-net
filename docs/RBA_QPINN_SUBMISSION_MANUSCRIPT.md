# 可学习周期相位编码网络用于非均质渗透率场内空间补全：多随机种子评估与精度—先验一致性权衡

**Learnable Periodic Phase-Encoded Networks for Within-Field Spatial Completion of Heterogeneous Permeability Fields: Multi-Seed Evaluation and Accuracy–Prior-Consistency Trade-offs**

> 作者：【待确认】  
> 单位：【待确认】  
> 通讯作者及邮箱：【待确认】  
> 基金项目：【待确认】

## 摘要

非均质渗透率场的快速补全是地下流体模拟与储层表征中的基础问题。本文研究一个边界明确的场内空间补全任务：给定同一批二维场中的稀疏观测、样本级工程条件和场标识，预测这些已登记场内的未观测位置；本文不评价对全新渗透率场的外推。针对坐标网络可能偏向低频结构的问题，构建了由角度投影、正弦—余弦响应、可学习相位混合、正弦首层和门控残差主干组成的周期相位架构，并考察动态损失平衡、训练点残差自适应加权和 Monte Carlo 神经元效用筛选的累积加入效果。归档实验使用实际来自单一 GAN 数据目录的 128 个 \(32\times32\) 归一化对数渗透率场；对每个随机种子生成约 70%、10% 和 20% 的训练、验证与空间留出测试掩码，采用 5 个随机种子重复六组架构比较，并分别报告网络 Raw 输出和 IDW-Corrected 输出。归档结果中，Quantum Feature PINN 架构包取得最低 Raw RMSE：\(0.0845\pm0.0014\)，相较 Baseline PINN 的 \(0.1741\pm0.0021\) 平均降低 51.47%；MAE 由 \(0.1317\pm0.0014\) 降至 \(0.0591\pm0.0009\)，\(R^2\) 由 \(0.7237\pm0.0058\) 提升至 \(0.9349\pm0.0025\)。五种子配对 Wilcoxon 单侧 \(p=0.03125\)，平均 RMSE 改善的 seed-level bootstrap 95% 置信区间为 [0.0870, 0.0918]。IDW 校正后，最佳模型 RMSE 为 \(0.01915\pm0.00018\)，相较校正基线降低 37.37%。沿当前累积消融路径加入动态模块后，场均值先验偏差降低而 Raw 点值精度大幅恶化，呈现非单调结果。方法审计同时发现：归档实验以完整真实场均值拟合弱先验标定参数；IDW 输出又使用完整真实场范围裁剪；基线与周期相位模型还同时改变了编码、激活、门控与主干结构。因此，本文将全部数值结果界定为审计前探索性证据，仅支持“当前架构包在已知场内补全中表现更好”和“沿当前累积路径完整模型非最优”，不能据此证明相位特征或自适应模块的独立因果贡献、严格无泄漏泛化或 Darcy 局部守恒。

**关键词**：渗透率场；场内空间补全；周期相位编码；坐标网络；残差自适应；效度审计

## Abstract

Sparse completion of heterogeneous permeability fields is important for subsurface-flow modelling and reservoir characterization. This study considers within-field completion: observations and held-out locations belong to the same registered fields, and no claim is made about generalization to unseen fields. We construct a learnable periodic architecture combining angle projection, sinusoidal responses, trainable phase mixing, a sine first layer, and a gated residual backbone. Dynamic loss balancing, training-point residual weighting, and Monte Carlo neuron-utility screening are added cumulatively. The archived experiment contains 128 normalized \(32\times32\) log-permeability fields from one GAN source directory. For each of five random seeds, spatial points were split into approximately 70% training, 10% validation, and 20% held-out testing locations. Six variants were evaluated, and raw outputs were separated from IDW residual-corrected outputs. The Quantum Feature PINN architecture bundle achieved the best archived raw RMSE, \(0.0845\pm0.0014\), versus \(0.1741\pm0.0021\) for the baseline, a mean reduction of 51.47%. MAE decreased from \(0.1317\pm0.0014\) to \(0.0591\pm0.0009\), and \(R^2\) increased from \(0.7237\pm0.0058\) to \(0.9349\pm0.0025\). The seed-level paired one-sided Wilcoxon test yielded \(p=0.03125\), and the seed-level bootstrap 95% confidence interval for mean RMSE improvement was [0.0870, 0.0918]. IDW correction reduced the best RMSE to \(0.01915\pm0.00018\). Along the current cumulative ablation path, adding adaptive modules reduced a field-mean prior discrepancy but substantially degraded raw predictive accuracy. A post hoc audit identified three threats: full ground-truth field means were used to calibrate the weak prior; IDW-corrected outputs were clipped using the full ground-truth field range; and the baseline and periodic models differ in multiple architectural components. The archived results are therefore treated as exploratory evidence for an architecture-bundle advantage and non-monotonic cumulative behaviour, not as proof of isolated component effects, leakage-free generalization, quantum acceleration, or local Darcy conservation.

**Keywords**: permeability field; within-field spatial completion; periodic phase encoding; quantum-inspired network; residual adaptation; validity audit

## 1 引言

渗透率控制多孔介质中的流体运移，是地下水预测、油气藏模拟和二氧化碳地质封存等任务中的关键参数。实际地层通常具有显著的空间非均质性，局部高、低渗区域及其连通结构会直接影响流动通道。传统数值模拟依赖明确的控制方程、边界条件和网格离散；当需要对大量候选场或工况反复求解时，其计算成本较高。数据驱动模型能够从已有样本中学习空间映射，但若训练目标只包含点值误差，未观测区域可能出现过度平滑或不符合工程统计关系的预测。

【需补充真实引用：非均质渗透率场的储层表征意义、连通结构对流动响应的影响，以及稀疏观测场重建综述。当前参考文献库尚无能够直接覆盖上述领域背景的来源。】

物理信息神经网络（physics-informed neural network, PINN）通过在目标函数中加入控制方程、边界条件或其他物理先验，为数据与机理的联合建模提供了统一形式[1]。然而，多尺度解、尖锐过渡以及复合损失梯度失衡会使 PINN 难以训练[2]。已有研究分别从逐点自适应权重[3]、残差驱动采样[4]和损失梯度平衡[2]等方面缓解上述问题。另一方面，坐标 MLP 对高频分量的学习速度较慢，周期特征映射能够改变其有效核的频率分布，从而增强低维空间信号的表达能力[5]。量子机器学习中的角度编码也可解释为由数据编码门决定频率谱的函数族[6]，为构造经典可微的相位特征提供了启发。

本文研究的问题不是在已知压力场和边界条件下直接求解 Darcy 方程，而是在给定稀疏空间观测、样本级工程条件和样本标识的条件下补全归一化对数渗透率场。由于现有数据缺少逐点压力、井源项、真实尺度和绝对渗透率单位，本文将工程条件与渗透率场均值的统计关系作为弱物理先验。因此，本文采用“物理引导”而非“严格 PDE 求解”的表述，并将控制方程残差扩展留作后续工作。

围绕上述边界，本文提出三个研究问题。其一，在相同场内空间三分协议下，量子启发周期相位架构包能否比带固定 Fourier 坐标特征的经典基线取得更低的留出点误差？其二，在周期相位架构上继续叠加动态损失、逐点残差加权和神经元筛选，是否能够同时改善点值精度与先验一致性，还是会造成目标竞争？其三，统一的 IDW 训练残差校正会在多大程度上改变模型排序，以及 Raw 网络能力与后处理收益应如何区分？三个问题分别对应表示能力、优化机制和系统后处理，避免将所有改进混合成单一“综合性能”叙事。

本文的主要贡献为：

1. 构建一种经典硬件可执行的量子启发周期相位架构，通过角度编码、周期响应、可学习相位混合、正弦首层和门控残差主干表达二维非均质场，并明确该架构不涉及真实量子线路与量子加速；
2. 在统一实验框架下组合动态损失平衡、训练点残差自适应权重和 Monte Carlo 神经元效用筛选，并通过六组消融实验辨析各模块的作用；
3. 建立训练—验证—空间留出测试三分协议，在 5 个随机种子上分别归档 Raw 与 IDW-Corrected 结果，并通过种子级效应量、配对检验和 bootstrap 置信区间评价稳定性；
4. 报告一个具有方法学价值的负结果：沿当前累积消融路径，完整模块配置并非最优；在场均值先验偏差下降的同时，预测精度大幅恶化；
5. 对归档实验开展代码—数据—结果一致性审计，识别完整场均值标定泄漏、架构归因混杂、单一数据来源、场内而非场外泛化以及指标计算域不一致等效度威胁，并给出确认性重跑路线。

## 2 相关工作

### 2.1 物理信息学习与复合损失优化

Raissi 等[1]将偏微分方程残差、初边值条件和观测数据统一到神经网络损失中，形成了 PINN 的基本范式。复合损失中不同项的梯度尺度可能相差较大，导致部分约束在训练中被忽略。Wang 等[2]从梯度流角度分析了这一问题，并通过动态调节不同损失项的学习强度改善训练。McClenny 和 Braga-Neto[3]则为不同训练点引入可学习权重，使模型在极小化网络参数的同时提高难训练点的权重。

本文的动态损失机制作用于数据、均值物理先验、平滑和边界四类全局目标；逐点残差权重仅作用于已观测训练点。两者分别对应“目标之间”和“空间位置之间”的优化资源分配。

### 2.2 残差驱动自适应

残差点的空间分布会显著影响 PINN 的精度。Wu 等[4]系统比较了均匀与残差自适应采样，并提出 RAD 和 RAR-D 策略。Anagnostopoulos 等[7]则提出基于累积残差的有界逐点注意力，使优化器持续关注高残差区域。本文借鉴高残差区域聚焦思想，但由于任务使用固定网格和空间留出观测，具体实现不是增加 PDE 配置点，而是更新训练掩码内的逐点数据权重，并在局部邻域扩散权重。因而本文将其称为残差自适应加权，而不将其等同于连续域 PDE 配置点重采样。

### 2.3 周期坐标特征与量子启发编码

坐标 MLP 存在偏向低频函数的频谱偏置。Rahaman 等[8]从 Fourier 分析角度指出，深层网络对不同频率分量具有不同学习速度，通常优先学习全局平滑、低频的函数成分。Tancik 等[5]进一步表明，将低维坐标映射为一组正弦和余弦特征能够改变网络有效核的频率性质，从而改善高频低维信号的拟合。Sitzmann 等[9]提出采用周期激活函数的隐式神经表示，说明适当初始化的正弦网络可以表达细节丰富的连续信号及其导数。上述研究共同提示：对于本项目所见的局部变化和多尺度空间纹理，编码频谱与激活函数可能影响网络优先学习哪些结构；这一机制仍需受控消融验证。

量子模型的数据编码研究表明，参数化量子线路的输出可写成输入变量的部分 Fourier 级数，可访问频率由数据编码门决定[6]。这为相位型特征设计提供了数学启发，但“受量子编码启发”与“使用量子计算”是两个不同命题。本文不调用真实量子处理器，不模拟含噪量子电路，也不比较量子与经典计算复杂度，而是将角度投影、相位耦合和周期响应转换为经典张量运算。因此，文中的“量子启发”仅描述表示形式的来源，不能解释为量子优势、量子并行性或量子加速。

### 2.4 稀疏空间场补全与神经算子

稀疏空间观测到连续场的恢复可由地质统计插值、坐标隐式表示或算子学习等不同范式完成。逆距离加权根据距离衰减对邻域观测加权，具有实现简单、计算快速和易于解释的特点，但权重指数、邻域大小及空间非平稳性会影响结果[10]。Kriging 则利用变差函数或协方差模型表达空间相关性；GSTools 为随机场生成、变差函数估计和 Kriging 提供了统一的软件环境[11]。这些方法构成空间补全任务不可忽视的传统参照，因为当观测密度较高且空间相关性强时，局部插值本身即可取得较小误差。

神经算子试图学习函数空间之间的映射。DeepONet 通过 branch net 编码输入函数观测、通过 trunk net 编码输出坐标，从而学习非线性算子[12]；Fourier Neural Operator 在 Fourier 空间参数化核，可在固定训练分辨率之外讨论分辨率相关的映射[13]。与这些旨在泛化到新输入函数的算子模型不同，本文模型为每个已登记渗透率场维护可学习样本嵌入。因而其测试问题是“已知场中未观测位置的补全”，不是“输入一个新场的稀疏观测并输出完整新场”。这种区别决定了评价协议和结论外延：场内测试可检验坐标插值能力，却不能替代按完整场划分的归纳泛化实验。

### 2.5 物理信息、物理引导与统计先验

严格的 PDE 型 PINN 通常把控制方程残差、初始条件和边界条件直接加入目标函数[1]。更广义的 physics-informed machine learning 还包括结构约束、守恒先验、对称性、低阶机理或统计关系等多种知识注入形式[14]。二者都可利用领域知识，但可验证主张不同：只有在变量、尺度、方程、边界条件及离散或自动微分残差均有明确物理含义时，才适合讨论局部控制方程满足程度；若损失只约束场均值与工程条件的经验关系，则应称为“统计先验一致性”或“物理引导”，而非 Darcy 方程求解。

本项目代码虽然包含无量纲稳态 Darcy 辅助求解与连续性诊断，但正式五种子命令使用 `--no-darcy-physics`。因此归档模型训练时的“physics”项仅为工程条件—场均值统计先验，连续性和通量指标为事后诊断。本文据此把核心任务界定为先验引导的空间补全，并把真正的局部 Darcy 约束留给未来具有压力场、源汇项、真实尺度与边界条件的数据。

### 2.6 本文定位与可证伪命题

基于以上工作，本文不预设复杂模块一定优于简单模块，而提出三个可由归档数据检验的描述性问题：第一，周期相位架构包相较强 Fourier 基线是否在五个随机种子上均降低 Raw 测试 RMSE；第二，沿当前累积路径加入动态 Loss、RBA 和 MC 后，误差是否单调下降；第三，IDW-Corrected 后最佳方法身份与其余方法排序如何变化。结果显示前两问分别为“是”和“否”；第三问中最佳方法仍为 Quantum Feature PINN，但其余方法的排序发生变化。由于基线和周期相位模型同时改变多个结构组件，第一项观察不能进一步收缩为“相位特征单独造成全部改善”。严格因果归因需要在相同主干、相同参数量和相同初始化预算下，只替换一个组件的正交消融。

## 3 问题定义与方法

### 3.1 空间留出补全任务

设第 \(s\) 个样本的归一化对数渗透率场为

\[
u_s:\Omega\rightarrow\mathbb{R},\qquad
\Omega=[-1,1]\times[-1,1].
\]

每个样本对应工程条件

\[
\mathbf c_s=[P_s,\nabla P_s,\mu_s,\phi_s,Q_s],
\]

分别表示压力、压力梯度、黏度、孔隙度和流量。对每个场生成两两互斥且覆盖完整网格的掩码 \(\mathcal M_s^{\mathrm{tr}}\)、\(\mathcal M_s^{\mathrm{val}}\) 与 \(\mathcal M_s^{\mathrm{te}}\)。模型只在 \(\mathcal M_s^{\mathrm{tr}}\) 上计算梯度训练数据损失，使用 \(\mathcal M_s^{\mathrm{val}}\) 选择最低验证损失 checkpoint，并在全部训练完成后在 \(\mathcal M_s^{\mathrm{te}}\) 上计算点值误差。这里的“留出”只针对空间位置：三类位置属于相同的 128 个场，共享样本级工程条件和已经通过训练点优化的场嵌入。因此测试位置与训练位置在索引上互斥，但测试样本不是独立新场。为避免把点级留出误写成样本级外推，下文统一称“样本内空间留出测试”。

网络输入为

\[
\mathbf z_{s,i}=[x_i,y_i,\widetilde{\mathbf c}_s,\mathbf e_s],
\]

其中 \(\widetilde{\mathbf c}_s\) 是标准化工程条件，\(\mathbf e_s\in\mathbb R^{24}\) 是可学习样本嵌入。样本嵌入使模型能够在同一训练集中区分不同随机场，但也意味着当前协议评价的是已知样本内部的空间插值能力，而不是对全新样本的外推能力。

### 3.2 数据来源与可追溯性审计

上游加载器按 `gan`、`unet`、`unet-attention`、`unet-quantum` 和 `unet-attention-quantum` 的顺序读取字段与工程条件，先拼接所有可用数组，再截取前 `max_samples` 个样本。正式归档配置令 `max_samples=128`。对保存的 `source_ids.npy` 进行核验后，128 个来源标识均为 0，这意味着五种子投稿实验实际全部使用第一个 `gan` 目录中的样本，而不是五类来源的混合样本。故本文只把数据描述为“来自上游 GAN 目录、由 GSTools 流程形成的 128 个归一化场”，不声称多来源训练。

每个场为 \(32\times32\) 的单通道 `float32` 归一化对数渗透率，配套工程条件依次为压力、压力梯度、黏度、孔隙度和流量。仓库保存了处理后数组、掩码和运行配置，但归档 manifest 的 `source_root` 是个人计算机绝对路径，且当前仓库没有完整记录上游 GSTools 的协方差模型、相关长度、方差、随机场种子和生成脚本哈希。因此，现有文件足以复核模型在已保存数组上的计算，却不足以由零开始完全重建原始随机场数据。正式开放数据时需要补充匿名仓库、上游生成配置和数据哈希。

### 3.3 基线网络

基线模型采用频率集合 \(\{1,2,4,8\}\) 的固定坐标 Fourier 特征、四层残差 MLP、SiLU 激活和 LayerNorm，隐藏维度为 128。基线不使用可学习相位混合、动态损失、残差自适应或神经元筛选。它比纯坐标 MLP 更强，但并不是“仅移除量子特征、其他结构完全相同”的因子对照：量子启发模型还使用正弦首层、门控乘法、不同残差主干，并取消了基线中的 LayerNorm。因此，Baseline 与 Quantum Feature PINN 的比较属于两个架构包的比较，不能把全部性能差异唯一归因于相位编码。

### 3.4 量子启发周期相位架构

首先将输入投影为 \(q\) 个相位角：

\[
\boldsymbol\theta=W_e\mathbf z+\mathbf b_e.
\]

单量子比特旋转后的测量响应由周期函数等价表示：

\[
\boldsymbol\phi_1=\sin(\boldsymbol\theta),\qquad
\boldsymbol\phi_2=\cos(\boldsymbol\theta).
\]

为描述不同相位通道之间的耦合，引入可学习矩阵 \(M_q\)：

\[
\boldsymbol\eta=\sin(\boldsymbol\theta M_q).
\]

最终特征为

\[
\boldsymbol\phi_q(\mathbf z)=
\left[
\sin(\boldsymbol\theta),
\cos(\boldsymbol\theta),
\sin(\boldsymbol\theta+\boldsymbol\eta),
\cos(\boldsymbol\theta-\boldsymbol\eta)
\right].
\]

\(\boldsymbol\phi_q\) 与原始输入拼接后，首先进入频率系数 \(\omega=18\) 的正弦线性层。并行门控分支由线性层和 Sigmoid 构成，代码将门控输出再经正弦函数后与首层输出逐元素相乘；后续隐层采用 Linear—SiLU—残差相加，最终输出一个标量。默认相位通道数 \(q=6\)，因此周期特征维数为 24。上述运算全部由经典自动微分框架实现。

### 3.5 先验引导复合损失

总损失写为

\[
\mathcal L=w_d\mathcal L_{\mathrm{data}}
+w_p\mathcal L_{\mathrm{mean}}
+w_s\mathcal L_{\mathrm{smooth}}
+w_b\mathcal L_{\mathrm{boundary}}.
\]

数据项为训练掩码内的加权均方误差：

\[
\mathcal L_{\mathrm{data}}=
\frac{\sum_{s,i\in\mathcal M_s^{\mathrm{tr}}}
\lambda_{s,i}(\hat u_{s,i}-u_{s,i})^2}
{\sum_{s,i\in\mathcal M_s^{\mathrm{tr}}}\lambda_{s,i}+\varepsilon}.
\]

工程条件首先通过一个无量纲指标映射为期望趋势，再由线性标定得到期望场均值 \(m_s^{\mathrm{prior}}\)。归档代码中的工程指标综合了流量、黏度、压力梯度、孔隙度和压力相对参考值的对数或线性项。均值先验项为

\[
\mathcal L_{\mathrm{mean}}=
\frac{1}{S}\sum_s
\left(\frac{1}{|\Omega|}\sum_i\hat u_{s,i}-m_s^{\mathrm{prior}}\right)^2.
\]

平滑项惩罚相邻网格预测差分，边界项约束左右边界均值以及上下边界均值的一致性。固定权重版本使用
\(w_d=1.0\)、\(w_p=0.1\)、\(w_s=0.002\)、\(w_b=0.001\)。

需要强调的是，\(\mathcal L_{\mathrm{mean}}\) 是由样本级工程条件构造的统计先验，并非局部 Darcy 守恒残差。代码中的边界项只要求相对两侧的预测均值接近，也不是由实际储层边界条件推导的 Dirichlet 或 Neumann 约束。正式归档实验把 `darcy_physics` 设为 `false`，训练目标不含局部连续性或通量平衡项。

审计还发现，归档函数 `expected_normalized_mean()` 使用每个样本的完整真实场均值拟合趋势斜率与截距。完整场均值包含验证点和测试点标签，拟合结果随后进入训练损失，构成跨掩码的信息使用。虽然它只以样本级均值而非逐点标签进入训练，但仍破坏“测试信息从未参与模型构建”的严格条件。因此，本节忠实记录原实验实现，同时把受该项影响的所有归档结果标为探索性结果。确认性实验必须用训练掩码内观测估计标定量，或在独立训练场上固定标定参数，并重新训练全部模型。

### 3.6 动态损失平衡

对四个损失项维护概率幅 \(a_k\)，并更新为

\[
a_k\leftarrow
\sqrt{\beta a_k^2+(1-\beta)\operatorname{softmax}(g_k)},\qquad
w_k=K a_k^2,
\]

其中 \(K=4\)，\(g_k\) 综合当前损失尺度与相对改善率。该设计使各项权重随训练状态变化，同时通过平方归一形式保持非负。实现中动态权重不是乘在固定权重之上，而是替代固定权重；初始四项权重均为 1。权重更新使用每个 epoch 最后一个 mini-batch 的损失，而不是全 epoch 平均损失，这会增加批次顺序对调度轨迹的影响，也是后文解释性能恶化时需要考虑的实现因素。

### 3.7 逐点残差自适应

在训练掩码内计算残差

\[
r_{s,i}=\left|\hat u_{s,i}-u_{s,i}\right|.
\]

依据归一化残差以指数滑动方式更新 \(\lambda_{s,i}\)，并裁剪至 \([1,4]\)。每隔 5 个 epoch，从超过自适应阈值的训练点中识别高残差候选区域；对满足密度条件的局部邻域扩散权重，以降低孤立异常点对训练的影响。残差权重更新本身不访问留出点标签。尽管配置中保留 `residual_replace_ratio` 名称，当前代码没有新增、删除或替换坐标点，其作用是对区域内训练点增加权重。因此准确术语是“残差自适应空间加权”，而不是“残差重采样”。

### 3.8 Monte Carlo 神经元效用

对相位特征层第 \(j\) 个神经元定义效用

\[
U_j=\mathbb E_{\mathrm{MC}}\!\left[|a_j(\mathbf z)|\right]
\sqrt{\operatorname{Var}_{\mathrm{MC}}[a_j(\mathbf z)]+\varepsilon}.
\]

训练前半程使用 4 次随机批采样估计 \(U_j\)，以效用分布的 5% 分位为阈值，对低效用正弦首层神经元进行轻量重初始化。该步骤旨在减少长期弱激活通道，不改变网络宽度。但实现没有同步清除 AdamW 对被重置参数保存的一阶与二阶动量，重置后的参数与旧优化状态可能不一致，因而 MC 模块既可能带来表达刷新，也可能造成优化扰动。

### 3.9 训练观测残差校正

实验脚本对所有模型统一应用逆距离加权（inverse distance weighting, IDW）残差校正[10]。对测试位置 \(\mathbf x\)，利用同一场内 \(K=8\) 个最近训练观测点的残差估计

\[
\Delta(\mathbf x)=
\frac{\sum_{j\in\mathcal N_K(\mathbf x)}
d(\mathbf x,\mathbf x_j)^{-p}
\left[u(\mathbf x_j)-\hat u(\mathbf x_j)\right]}
{\sum_{j\in\mathcal N_K(\mathbf x)}d(\mathbf x,\mathbf x_j)^{-p}+\varepsilon},
\]

并令 \(\hat u^{\,c}(\mathbf x)=\hat u(\mathbf x)+0.95\Delta(\mathbf x)\)，其中 \(p=2\)。邻域残差计算只读取训练观测点，但代码最后使用完整真实场的最小值与最大值裁剪校正输出，该范围包含验证和测试标签。因此归档 Corrected 结果不是严格只使用训练信息的无泄漏后处理，只能作为探索性次结果。确认性版本应改用训练观测范围、预先固定的归一化范围或不裁剪，并重新生成表 4。本文仍将 Raw 输出作为模型比较主结果，以避免把后处理和模型本体混为一谈。

## 4 实验设置

### 4.1 数据与划分

根据上游项目说明，归档实验数据来自 `gan` 数据目录，其随机场生成流程使用 GSTools 地质统计工具[11]；但当前仓库缺少完整上游生成链和哈希，因此该来源说明仍需原始记录确认。数据包含 128 个 \(32\times32\) 二维归一化对数渗透率场，共 131072 个网格点。采用随机种子 42–46 重复实验。以 seed 42 为例，训练、验证和空间留出测试掩码分别包含 91928、13070 和 26074 个点，占 70.14%、9.97% 和 19.89%。不同种子重新生成点级掩码和训练随机状态，但保持相同的 128 个场、划分概率与模型配置。因此五个种子是算法随机性与空间掩码重复，不是五个独立数据集。五个工程条件的统计范围见表 1。

**表 1 工程条件统计**

| 条件 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 压力/MPa | 30.1751 | 4.9193 | 19.7594 | 41.8335 |
| 压力梯度/(MPa·m\(^{-1}\)) | 0.0120 | 0.0017 | 0.0065 | 0.0166 |
| 黏度/cP | 1.0130 | 0.1450 | 0.6792 | 1.3856 |
| 孔隙度 | 0.2210 | 0.0178 | 0.1727 | 0.2689 |
| 流量/(m\(^3\)·d\(^{-1}\)) | 119.1079 | 18.5368 | 69.0806 | 167.9973 |

### 4.2 训练配置与对比模型

所有模型采用 AdamW 优化器，初始学习率为 \(10^{-3}\)，权重衰减为 \(10^{-5}\)，并使用余弦退火学习率。批大小为 8，训练 40 个 epoch，梯度范数裁剪阈值为 1.0。量子启发模型与基线均采用四层、隐藏维度 128 的主干网络。每个种子的运行目录均不可覆盖，并保存配置 manifest、三类掩码、模型权重、Raw/Corrected 预测与逐样本误差。六个对比模型为：

1. Baseline PINN；
2. Quantum Feature PINN；
3. Quantum + Loss Balancer；
4. Quantum + Loss + MC；
5. Quantum + Loss + RBA（RBA-QPINN w/o MC）；
6. Full RBA-QPINN。

**表 2 六组模型的组件关系**

| 方法 | 经典 Fourier 基线 | 周期相位架构包 | 动态 Loss | RBA 空间加权 | MC 筛选 |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline PINN | √ | — | — | — | — |
| Quantum Feature PINN | — | √ | — | — | — |
| Quantum + Loss Balancer | — | √ | √ | — | — |
| Quantum + Loss + MC | — | √ | √ | — | √ |
| Quantum + Loss + RBA | — | √ | √ | √ | — |
| Full RBA-QPINN | — | √ | √ | √ | √ |

表 2 反映当前消融是一条累积路径，而不是完整正交设计。缺少 Classical + Loss、Classical + RBA、Quantum + MC-only 和 Quantum + RBA-only 等组合。因此，现有实验可以回答“沿当前累积路径发生了什么”，但不能完整估计所有模块的主效应与交互效应。

运行环境由聚合文件记录为 Python 3.12.13、PyTorch 2.13.0、NumPy 2.3.5、SciPy 1.18.0 和 Matplotlib 3.11.1，平台为 macOS 15.7.7 arm64，配置使用 CPU。以 seed 42 为例，六模型单次训练耗时约 15.90–23.58 s，其中 Quantum Feature PINN 为 16.32 s，Baseline PINN 为 22.22 s。由于当前只保存逐种子耗时表，本文不对平均加速比作统计推断，也不把 CPU 小规模训练耗时外推到更高分辨率或 GPU 场景。

### 4.3 指标

主指标包括空间留出测试点 MSE、RMSE、MAE 和 \(R^2\)。其中 RMSE、MAE 与 \(R^2\) 只读取测试掩码。项目中的 SSIM 实现则对每幅完整 \(32\times32\) 场使用全局均值、方差和协方差计算，再跨样本平均；它不是标准滑窗 SSIM，也不是纯测试点指标，本文据此将其称为“全局 SSIM 型结构指标”。Physics Residual 定义为完整预测场均值与工程条件期望均值之差的平均绝对值，同样不是局部 Darcy 残差。Accuracy 被实现为 \(1-\mathrm{MAE}/\mathrm{range}\)，Physical Score 和 Composite Score 也是项目自定义诊断量，缺乏跨研究统一含义，故不作为主要证据。

各核心指标报告 5 个随机种子的均值±样本标准差。聚合脚本在同一批结果中先按平均 RMSE 选择最佳非基线模型，再对其与基线做检验，存在探索性选择效应。因此本文把效应量与置信区间置于 \(p\) 值之前。RMSE 使用种子级配对 t 检验、单侧 Wilcoxon 符号秩检验和以种子为重采样单位的 20000 次 bootstrap 95% 置信区间分析；逐样本 Wilcoxon 仅作辅助分析，不将同一种子内的 128 个场或跨种子重复观测视为主要独立重复。对 \(n=5\) 且五个差值同号的情形，精确单侧 Wilcoxon 的最小可达 \(p\) 值即为 0.03125，故显著性结论需结合样本量谨慎解释。

## 5 实验结果

### 5.1 总体性能

**表 3 六组模型的样本内空间留出 Raw 结果（均值±标准差，\(n=5\)）**

| 方法 | RMSE↓ | MAE↓ | \(R^2\)↑ | SSIM↑ | Physics Residual↓ |
|---|---:|---:|---:|---:|---:|
| Baseline PINN | 0.1741±0.0021 | 0.1317±0.0014 | 0.7237±0.0058 | 0.6866±0.0122 | 0.1228±0.0003 |
| Quantum Feature PINN | **0.0845±0.0014** | **0.0591±0.0009** | **0.9349±0.0025** | **0.9384±0.0028** | 0.1209±0.0001 |
| Quantum + Loss Balancer | 0.2309±0.0040 | 0.1809±0.0034 | 0.5141±0.0168 | 0.3296±0.0109 | 0.0600±0.0016 |
| Quantum + Loss + MC | 0.2553±0.0048 | 0.2020±0.0035 | 0.4055±0.0253 | 0.1546±0.0133 | **0.0339±0.0030** |
| Quantum + Loss + RBA | 0.2284±0.0015 | 0.1794±0.0010 | 0.5245±0.0033 | 0.3483±0.0217 | 0.0627±0.0036 |
| Full RBA-QPINN | 0.2477±0.0041 | 0.1966±0.0034 | 0.4406±0.0212 | 0.1852±0.0291 | 0.0395±0.0035 |

Quantum Feature PINN 架构包在四个主要预测指标上均最优。相较 Baseline PINN，其平均 RMSE 下降 51.47%，MAE 下降 55.12%，\(R^2\) 提升 0.2112。五个种子的 RMSE 改善均为正，依次为 0.09174、0.08808、0.09250、0.09072 和 0.08507；平均绝对改善为 0.08962，seed-level bootstrap 95% 置信区间为 [0.08701, 0.09184]。种子级配对 t 检验的单侧 \(p=1.59\times10^{-7}\)，单侧 Wilcoxon 检验 \(p=0.03125\)。上述结果说明架构包优势在五次随机重复中方向一致，但 \(n=5\) 较小，且最佳模型由同一批结果选出。它支持“当前周期相位架构包比当前 Fourier 基线取得更低场内留出误差”的描述，不足以单独证明相位特征的因果作用，更不支持量子计算加速或完整 RBA-QPINN 全面优越。

![多随机种子 Raw 与 IDW 校正 RMSE](../experiments/rba_qpinn_submission_v1/aggregate/multiseed_rmse_raw_vs_corrected.png)

**图 1 六组模型在 Raw 与 IDW-Corrected 设置下的 RMSE（5 个随机种子的均值±标准差）。**

图 1 为归档聚合图，图内纵轴仍沿用 “Independent-test RMSE” 的旧标签；准确含义应为 “Within-field spatial-holdout RMSE”。该图需在正式投稿前重生成，避免把点级留出误解为独立新场测试。

### 5.2 模块贡献与精度—一致性权衡

动态损失平衡加入后，Raw RMSE 从 Quantum Feature PINN 的 0.0845 回升至 0.2309；沿当前累积路径继续加入 MC 或 RBA 后，RMSE 为 0.2284–0.2553。Full RBA-QPINN 的 RMSE 为 0.2477，高于基线的 0.1741。由此可见，当前数据和训练预算下，累积加入模块的结果明显非单调，不能用“模块越多性能越好”解释结果。由于缺少 MC-only、RBA-only 和完整正交组合，这一现象不能被直接归因于某个模块的主效应或严格的统计交互。

Physics Residual 的变化方向不同。Quantum + Loss + MC 得到最低 Raw 值 \(0.0339\pm0.0030\)，相较基线下降约 72.43%；Full RBA-QPINN 为 \(0.0395\pm0.0035\)。与此同时，这两个模型的 RMSE 明显恶化，说明动态损失与神经元筛选将优化重点转向了场均值先验，形成预测精度与弱物理一致性的强权衡，而不是同时改善两个目标。该指标仍不是局部 Darcy 守恒残差。

**表 4 六组模型的 IDW-Corrected 空间留出结果（均值±标准差，\(n=5\)）**

| 方法 | RMSE↓ | MAE↓ | \(R^2\)↑ | SSIM↑ | Physics Residual↓ |
|---|---:|---:|---:|---:|---:|
| Baseline PINN | 0.03057±0.00091 | 0.02221±0.00066 | 0.99148±0.00048 | 0.99766±0.00018 | 0.13266±0.00008 |
| Quantum Feature PINN | **0.01915±0.00018** | **0.01131±0.00011** | **0.99666±0.00007** | **0.99906±0.00007** | 0.13259±0.00004 |
| Quantum + Loss Balancer | 0.02603±0.00047 | 0.01800±0.00033 | 0.99382±0.00021 | 0.99838±0.00009 | 0.13172±0.00003 |
| Quantum + Loss + MC | 0.02676±0.00021 | 0.01887±0.00017 | 0.99347±0.00011 | 0.99829±0.00005 | **0.13121±0.00006** |
| Quantum + Loss + RBA | 0.02590±0.00034 | 0.01792±0.00017 | 0.99389±0.00014 | 0.99846±0.00007 | 0.13174±0.00006 |
| Full RBA-QPINN | 0.02641±0.00039 | 0.01857±0.00025 | 0.99364±0.00019 | 0.99828±0.00012 | 0.13128±0.00007 |

IDW 校正大幅降低了所有模型的误差，最佳方法身份仍为 Quantum Feature PINN，但其余方法排序发生改变：Raw 排序中 Baseline 位列第二，而 Corrected 排序中 Baseline 的 RMSE 最高。Quantum Feature PINN 相较校正基线的平均 RMSE 下降 37.37%，种子级单侧 Wilcoxon 检验 \(p=0.03125\)。Raw 与 Corrected 的相对改善幅度分别为 51.47% 和 37.37%，但 Corrected 数值还受到完整真实场范围裁剪的信息泄漏影响，不能作为确认性系统性能。另一方面，校正后所有方法的 RMSE 都降到 0.019–0.031 区间，显示在 70% 规则网格观测密度下，局部残差插值本身十分强。这也暴露了当前基线集合的不足：确认性研究应修复裁剪并加入 IDW-only、Kriging、RBF 等纯空间方法，以区分“网络预测+残差补丁”相对传统插值的净收益。

值得注意的是，IDW 后处理改善了点值误差，却没有改善场均值先验偏差：六模型 Corrected Physics Residual 均约为 0.131，而 Raw 中动态模型最低约为 0.034。该方向差异进一步说明，后处理主要利用局部观测残差修正点值，不保证维持原训练目标中的场均值统计关系。因而不能用“Corrected 系统全面改善”概括表 4。

### 5.3 误差分布与空间结构

预测—真实散点图中，Quantum Feature PINN 的样本更集中于 \(y=x\) 附近；绝对误差分布和累计分布也表明其大误差比例较低。重建场对比显示，基线能够恢复主要低频区域，但在局部变化较快的位置更易产生平滑偏差。周期相位特征为坐标网络提供了更丰富的频率基，从而可能缩短高频结构的学习路径。这一解释与 Fourier 特征缓解坐标 MLP 频谱偏置的结论一致[5]。

![空间留出测试点预测](../experiments/rba_qpinn_submission_v1/seed_42/heldout_point_prediction_comparison.png)

**图 2 seed 42 下 Target、Baseline PINN 与 Quantum Feature PINN 在样本内空间留出测试点上的预测。脚本按 Quantum Feature PINN 相对 Baseline 的测试 MAE 改善选择收益最大的 3 个样本，因此该图是后验最佳案例展示，不代表预先指定的典型样本。**

### 5.4 RBA 机制的可解释性与局限

残差权重图用于展示模型对训练掩码内难拟合区域的关注变化。密度扩散使连续高残差区域获得较高权重，避免只追逐单个异常点。然而，五个随机种子的结果均未证明这种关注变化能够转化为更低的测试 RMSE；Quantum + Loss + RBA 仅略优于 Quantum + Loss Balancer，而 Full RBA-QPINN 明显弱于 Quantum Feature PINN。因此，残差热图只能作为机制诊断。现有图片内部标题仍写作 “Residual-adaptive sampling”，与实际仅做 weighting 的代码不一致，故本版正文不嵌入该图；正式排版前需重生成并统一术语。

### 5.5 有效性边界

本文结果受多项边界约束。第一，当前采用同一样本内的空间三分，训练、验证和测试点共享样本级嵌入及工程条件，不能据此推断模型对全新渗透率场的泛化能力。第二，物理项约束场均值而非局部质量守恒，不能将当前模型描述为严格求解 Darcy PDE 的 PINN。第三，完整真实场均值参与先验标定，使测试信息以样本级统计量的形式进入训练；因此三掩码虽在点索引上互斥，归档实验仍不满足严格无泄漏条件。第四，基线与量子启发模型的编码、激活、门控、归一化和残差主干同时变化，不能将 51.47% 改善全部归因于某一个“量子特征”组件。第五，128 个场全部来自同一个上游来源，且只有单一分辨率和观测比例。

这些问题并不改变已保存数组中“哪一个归档模型误差最低”的事实，却会改变该事实可以支持的理论解释和外部推广范围。因此本文将表 3、表 4 称为审计前探索性结果：它们可用于生成下一轮假设、识别失效模块和设计确认性实验，但不应直接作为一区 TOP 稿件的最终证据。

### 5.6 图表证据层级

跨种子聚合图 `multiseed_rmse_raw_vs_corrected.png` 直接读取五种子聚合结果，是当前主图。seed 42 的 `heldout_point_prediction_comparison.png` 在未插值的空间留出点上比较六个模型，可用于展示点值分布；`prediction_field_ablation_comparison.png` 只展示单个代表样本，应作为定性案例而非总体统计；`residual_sampling_heatmap.png` 与 `residual_point_weight_dynamics.png` 只能说明残差权重机制确实运行，不能证明该机制改善泛化。

根目录 `figures/` 中部分雷达图、热力图和扩展图来自较早的单次实验或 IDW 校正数组，其数值体系与五种子 Raw 主结果不同。因此正式排版时不得把根目录旧图标注为“五种子结果”，也不得把旧表中接近校正尺度的 RMSE 与表 3 的 Raw RMSE 混用。正文优先引用 `experiments/rba_qpinn_submission_v1/aggregate/` 和对应 seed 目录下的不可覆盖产物。

## 6 讨论

### 6.1 周期相位架构为何可能改善场内补全

在本项目保存的归一化场中，可观察到大尺度趋势、局部高低值斑块和不同空间尺度的过渡结构。直接以坐标输入常规 MLP 时，优化过程往往先拟合平滑低频成分，再逐步逼近局部变化[8]。固定 Fourier 特征通过预设频率扩展输入基，但其频率集合与方向由设计者固定。本文周期相位架构将完整输入投影为可学习角度，并通过相位混合产生 \(\sin(\theta+\eta)\) 与 \(\cos(\theta-\eta)\) 等响应；从函数表示角度看，这相当于允许网络在训练中调整一组非线性周期基。正弦首层还进一步改变了隐层的频谱偏好。这些机制为 Quantum Feature PINN 架构包在留出点上降低大误差提供了一种合理解释。

然而，合理解释不等于因果证明。当前 Baseline 使用固定坐标 Fourier 特征、LayerNorm 和 4 个双线性 SiLU 残差块；Quantum Feature PINN 使用全输入角度投影、SineLayer、门控分支和 3 个单线性残差隐层。按源码参数形状计算，在 128 个场、24 维嵌入、128 隐藏维和当前输入维数下，Baseline 约含 142977 个可训练参数，Quantum Feature PINN 约含 67301 个，两者预算并不匹配。参数初始化与方法运行顺序也未做成严格配对。因而实验比较的是“两个完整架构方案”，不是只替换相位层的受控实验。未来应建立共同主干，按以下顺序逐项加入：固定 Fourier、可学习角度、相位混合、Sine 激活和门控。只有当其余条件固定时，才能量化每一项的边际贡献及交互作用。

### 6.2 动态损失为何降低先验偏差却损害点值精度

固定权重 Baseline 与 Quantum Feature PINN 令数据项权重为 1，而均值、平滑和边界项权重分别为 0.1、0.002 和 0.001。动态平衡启用后，四项初始权重均为 1，并由损失倒数和相对改善率更新。这个变化不仅是“自适应”，还显著提高了训练早期平滑、边界和均值项相对数据项的作用。若场均值先验比局部点值更容易被压低，优化器可能优先找到全局均值接近目标、但局部纹理被平滑的解。表 3 中 Physics Residual 下降而 RMSE、MAE 和全局 SSIM 型指标恶化，与这一机制一致。

更新频率也可能放大该问题。代码使用最后一个 mini-batch 的损失更新全局权重，最后批次的样本组成会影响下一轮调度；同一个 DataLoader 随机生成器又在不同方法间持续推进，方法之间并未获得完全匹配的 batch 顺序。动态调度因此混合了真实优化状态与批次噪声。更稳妥的实现应使用 epoch 平均损失或梯度范数，记录权重更新前后的各项梯度，并对权重范围、平滑系数和更新周期做敏感性分析。

### 6.3 RBA 和 MC 累积路径的非单调结果

RBA 仅根据训练掩码残差调整逐点权重，从信息访问上没有直接读取测试点标签。它把优化资源集中到训练中难拟合的区域，但“训练残差较大”不必然等于“对测试位置最有价值”。在规则网格、70% 观测率和场专属嵌入的设置下，高残差训练点可能位于极值或当前主干难以拟合的局部区域。提高这些点的权重会减少网络对整体结构的优化预算，尤其当动态 Loss 已经改变全局目标比例时，二者可能产生竞争。结果中 Quantum + Loss + RBA 略优于 Quantum + Loss Balancer，却仍明显弱于不使用动态模块的 Quantum Feature PINN，说明沿当前累积路径，RBA 没有抵消此前的全局目标偏移。由于缺少 RBA-only 对照，这不能被解释为 RBA 的独立负效应。

MC 模块在训练前半程反复重置低效用首层神经元。其效用指标结合激活绝对均值与方差，能够识别长期低响应通道；但神经元被重置后，AdamW 中与原参数对应的动量状态没有清除。网络参数和优化器状态之间的不连续，可能造成短期损失波动，并破坏已经形成的相位组合。Quantum + Loss + MC 的 Raw RMSE 为六个模型中最高，Full RBA-QPINN 也没有恢复精度。这一结果并不证明 MC 筛选在所有任务中无效，也不能在缺少 MC-only 对照时确定其主效应；它只说明包含当前动态 Loss、效用定义、重置频率、动量处理和 40 epoch 预算的累积配置表现不佳。

### 6.4 负结果的研究意义

复合科学机器学习方法常把多个看似合理的模块叠加，并以完整模型作为创新主体。但每个模块改变的可能是不同目标：周期表示改变坐标函数表达，动态 Loss 改变目标间资源分配，RBA 改变空间点间资源分配，MC 改变可用神经元集合。若这些模块共享有限训练预算，收益就不必可加。本文归档结果清楚展示了沿当前累积路径的非单调性：最简单的周期相位架构包具有最低误差，而完整配置在弱先验偏差上更好、在点值与结构精度上更差。

因此，Full RBA-QPINN 非最优不是需要隐藏的“失败实验”，而是对模块化方法设计的重要提醒。评价复合模型时，应同时报告单模块、组合模块与完整模型，预先指定主指标，并区分预测精度、物理或先验一致性、计算代价等不同目标。若不存在一个模型在所有目标上占优，更适合报告 Pareto 权衡，而不是通过任意加权的 Composite Score 把多目标压缩成单一排名。

### 6.5 Raw 与 IDW-Corrected 的工程含义

Raw 输出反映网络在给定坐标、工程条件和场嵌入后的直接预测能力；IDW-Corrected 则把训练观测点上的网络残差作为新信息，通过局部距离权重传播到未观测位置。二者回答不同问题：Raw 适合比较表示与训练方法，Corrected 原本用于评估“网络+观测后处理”的系统性能。当前 IDW 邻域残差本身没有读取测试标签，但最终裁剪使用了完整真实场范围，因此归档 Corrected 结果仍有信息泄漏。修复裁剪后，仍可研究“网络全局趋势+局部观测残差插值”的混合方案。

一种更具可解释性的系统设计是让网络学习全局趋势和非局部结构，让地质统计或局部插值恢复观测附近的高频残差。要证明该系统优于传统方案，还需加入 IDW-only、Kriging、RBF、Gaussian process 或局部多项式等基线，并在不同观测密度与空间非平稳程度下比较。否则，只能说明 IDW 对所有已训练网络都有帮助，不能说明神经网络在校正系统中不可替代。

### 6.6 工程适用范围

在当前协议中，模型适合的假想场景是：同一批已经登记的二维场有大量已知观测，系统需要补全剩余规则网格位置，并允许为每个场保存专属嵌入。这种设置可对应离线数据修复、缺测网格填补或重复访问同一场的数字化建模。它不对应新储层首次进入系统、只有少量观测就立即输出完整场的情形，因为新场没有已经训练好的样本 ID 嵌入。

此外，目标变量是归一化对数渗透率，数据缺少真实区域尺寸、绝对渗透率单位、压力场、井源项和边界条件。即使点值误差较低，也不能直接推断流量预测、突破时间或储层决策误差会同步降低。工程应用前需把归一化量恢复到物理单位，用下游流动模拟评价功能性误差，并在真实或高保真数据上验证。

## 7 有效性威胁与 TOP 投稿整改路线

### 7.1 内部有效性

最严重的内部有效性问题是两处信息泄漏。第一，归档代码先从完整真实场计算每个样本均值，再拟合工程指标与真实均值的线性关系；该关系随后进入所有模型的训练损失。由于完整均值含验证和测试位置，测试信息间接影响模型参数。第二，IDW 校正虽然用训练点残差插值，却用完整真实场的最小值和最大值裁剪输出。修复方案包括：只用训练掩码观测或独立训练场拟合均值先验；同时改用训练观测范围、预设归一化范围或不裁剪的 IDW。修复后必须重新生成全部 Raw 与 Corrected 五种子结果，不能只在论文中修改表述。

第二个问题是架构归因混杂。确认性实验应在统一参数预算下建立正交因子设计，分别控制坐标编码、相位混合、激活函数、门控、LayerNorm 与残差块。各模型应使用匹配初始化或成对随机种子，并为每个方法重建相同顺序的 DataLoader。若比较多个候选模型后再选择最佳者做检验，需要预先定义主比较或进行多重比较调整。

第三个问题是训练预算。40 epoch 足以得到当前排序，但不能保证每种动态方法都达到自己的收敛区域。应报告训练—验证曲线、最佳 epoch、早停行为和更长预算敏感性；动态 Loss 应使用 epoch 聚合量，MC 重置后应清理对应优化器状态。只有在调参预算对各方法公平时，才能把性能差异解释为模块差异。

### 7.2 外部有效性

归档的 128 个场全部来自单一 GAN 来源，网格固定为 \(32\times32\)，观测率约为 70%，且五个种子重复使用同一数据。外部有效性实验应改变完整场而非仅改变空间掩码：按场划分训练、验证和测试，保证测试场从未获得可学习 ID 嵌入；进一步改变随机场协方差、相关长度、各向异性、观测率和噪声水平。可使用由稀疏观测集合编码得到的场级表示，代替不可用于新场的 ID embedding。

真实储层数据还会带来不规则采样、测量噪声、尺度差异、非平稳性和数据缺失机制偏差。合成 GSTools 场可用于受控方法研究，却不能单独证明实际储层适用性。TOP 级投稿至少需要公开基准、独立高保真模拟或真实数据之一，并说明数据许可、预处理和不确定性。

### 7.3 构念有效性

本文的 Physics Residual 是场均值偏差，不是 \(\nabla\cdot[(k/\mu)\nabla p]\) 的局部残差；边界项是相对边界均值一致性，不是实际边界条件；全局 SSIM 型指标也不是标准滑窗 SSIM。若术语不加限定，读者可能把这些代理量误解为严格物理量或通用图像指标。确认性论文应把 RMSE、MAE 和 \(R^2\) 作为主指标，使用标准实现复算结构指标，并把自定义指标放入补充材料。

“量子启发”同样属于构念边界。当前方法全部在经典 CPU 上执行，没有量子线路深度、量子比特噪声、采样测量或经典—量子复杂度比较。可讨论量子数据编码的频谱启发[6]，但不得使用“量子优势”“量子加速”或“量子计算实验”等措辞。

### 7.4 可复现性与报告完整性

五个种子目录启用了不可覆盖保护，并保存 manifest、掩码、模型权重、Raw/Corrected 预测、逐样本结果和聚合统计，这是当前项目的重要优点。聚合环境也记录了 Python、平台与主要库版本。但 `requirements.txt` 未锁定版本，训练脚本未启用 PyTorch 完全确定性算法，manifest 未记录 Git commit、上游数据哈希和生成配置，外部数据路径不可移植。本次审计环境缺少 NumPy、PyTorch 和 pytest，因而未重新执行训练或测试；本文定量表由归档 JSON/CSV 交叉读取，而非本轮重新计算。

正式复现包应增加锁定依赖、数据校验和、Git commit、统一入口命令、标准输出日志与 CPU/GPU 确定性说明。应把根目录旧实验与五种子正式实验分离，避免图表生成脚本默认读取旧数组。图注需写明 seed、Raw/Corrected、指标计算域和数据版本。

### 7.5 TOP 投稿前的最小确认性实验

按照对结论影响的优先级，投稿前至少需要完成：

1. 同时修复完整场均值标定泄漏与 IDW 全场范围裁剪泄漏，并在相同种子与配置下重跑六模型；
2. 增加按完整场划分的 unseen-field 测试，移除或替换场 ID embedding；
3. 在共同主干上完成编码、Sine、门控、相位混合的等骨干因子消融；
4. 加入 IDW-only、Kriging、RBF、强 Fourier 坐标网络及至少一种适合稀疏场重建的深度基线；
5. 评估不同观测比例、噪声、网格分辨率、训练轮数和主要超参数；
6. 增加至少 10 个预先指定随机种子，预先规定主指标和主比较；
7. 汇总参数量、训练时间、推理时间和内存代价；
8. 若保留 PINN/Darcy 主张，补充真实压力、源汇、边界与尺度信息并启用局部守恒训练；否则统一使用 prior-guided 或 physics-guided 表述。

在完成这些整改之前，当前稿件适合作为完整研究报告、内部预稿或确认性实验方案，不宜标记为“一区 TOP 投稿就绪”。

## 8 结论

本文围绕 128 个 \(32\times32\) 归一化对数渗透率场的样本内空间补全，比较了经典 Fourier 基线、周期相位架构以及动态损失、残差空间加权和 MC 神经元筛选的累积组合。五随机种子的归档探索性结果回答了三个研究问题。第一，Quantum Feature PINN 架构包在当前场内留出协议中取得最低 Raw 误差，其 RMSE 和 MAE 相较 Baseline PINN 分别降低 51.47% 和 55.12%，五个种子的改善方向一致。第二，沿当前累积路径加入复杂模块并未带来单调增益；Full RBA-QPINN 的 Raw RMSE 为 0.2477，高于基线的 0.1741，场均值先验偏差降低的同时点值和结构精度恶化。第三，IDW 校正后的最佳方法身份仍为 Quantum Feature PINN，但其余排序改变；校正数值又受完整真实场范围裁剪泄漏影响，必须与 Raw 主结果分开并在修复后重算。

代码与数据审计限定了上述结论的证据强度。现有比较只能支持“当前完整架构包之间的描述性差异”，不能把全部改善归因于相位特征；样本 ID 嵌入把任务限制为已知场内补全；正式实验关闭 Darcy 训练；完整真实场均值参与弱先验标定，IDW 又读取完整真实场范围，使归档结果不满足严格无泄漏要求。因而本文没有声称量子加速、未知场泛化或局部 Darcy 守恒，也没有把 Full RBA-QPINN 包装为最优方法。

下一阶段的首要任务不是继续扩充文字，而是完成无泄漏重跑、等骨干因子消融、未知场评估与传统空间插值基线。若未来获得压力场、源汇项、真实尺度和边界条件，可进一步引入
\(\nabla\cdot[(k/\mu)\nabla p]=0\)
的局部守恒残差，并以流动响应而非仅点值误差评价工程价值。完成这些确认性验证后，才能判断量子启发周期表示是否在更一般的非均质场重建问题中保持优势，并据此形成符合一区 TOP 期刊证据标准的投稿版本。

## 数据与代码可用性

实验代码、处理后数据、空间掩码、模型权重、逐样本结果和图表生成脚本保存在本项目仓库中。五个种子的不可变运行目录均包含配置 manifest 与完整预测数组。正式投稿时应按目标期刊的数据开放政策提供匿名仓库链接。

## 利益冲突声明

作者声明不存在可能影响本文工作的已知利益冲突。

## 作者贡献

【待确认】按 CRediT taxonomy 补充。

## 参考文献

[1] RAISSI M, PERDIKARIS P, KARNIADAKIS G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. *Journal of Computational Physics*, 2019, 378: 686–707. DOI: 10.1016/j.jcp.2018.10.045.

[2] WANG S, TENG Y, PERDIKARIS P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J]. *SIAM Journal on Scientific Computing*, 2021, 43(5): A3055–A3081. DOI: 10.1137/20M1318043.

[3] MCCLENNY L D, BRAGA-NETO U M. Self-adaptive physics-informed neural networks[J]. *Journal of Computational Physics*, 2023, 474: 111722. DOI: 10.1016/j.jcp.2022.111722.

[4] WU C, ZHU M, TAN Q, et al. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J]. *Computer Methods in Applied Mechanics and Engineering*, 2023, 403: 115671. DOI: 10.1016/j.cma.2022.115671.

[5] TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C/OL]//*Advances in Neural Information Processing Systems*. Red Hook, NY: Curran Associates, Inc., 2020, 33: 7537–7547[2026-07-28]. https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html.

[6] SCHULD M, SWEKE R, MEYER J J. Effect of data encoding on the expressive power of variational quantum-machine-learning models[J]. *Physical Review A*, 2021, 103(3): 032430. DOI: 10.1103/PhysRevA.103.032430.

[7] ANAGNOSTOPOULOS S J, TOSCANO J D, STERGIOPULOS N, et al. Residual-based attention in physics-informed neural networks[J]. *Computer Methods in Applied Mechanics and Engineering*, 2024, 421: 116805. DOI: 10.1016/j.cma.2024.116805.

[8] RAHAMAN N, BARATIN A, ARPIT D, et al. On the spectral bias of neural networks[C]//CHAUDHURI K, SALAKHUTDINOV R, eds. *Proceedings of the 36th International Conference on Machine Learning*. Long Beach, CA: PMLR, 2019, 97: 5301–5310.

[9] SITZMANN V, MARTEL J N P, BERGMAN A W, et al. Implicit neural representations with periodic activation functions[C/OL]//*Advances in Neural Information Processing Systems*. Red Hook, NY: Curran Associates, Inc., 2020, 33: 7462–7473[2026-07-28]. https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html.

[10] LU G Y, WONG D W. An adaptive inverse-distance weighting spatial interpolation technique[J]. *Computers & Geosciences*, 2008, 34(9): 1044–1055. DOI: 10.1016/j.cageo.2007.07.010.

[11] MÜLLER S, SCHÜLER L, ZECH A, et al. GSTools v1.3: A toolbox for geostatistical modelling in Python[J]. *Geoscientific Model Development*, 2022, 15: 3161–3182. DOI: 10.5194/gmd-15-3161-2022.

[12] LU L, JIN P, PANG G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators[J]. *Nature Machine Intelligence*, 2021, 3: 218–229. DOI: 10.1038/s42256-021-00302-5.

[13] LI Z, KOVACHKI N, AZIZZADENESHELI K, et al. Fourier neural operator for parametric partial differential equations[C/OL]//*International Conference on Learning Representations*. [S.l.]: OpenReview, 2021[2026-07-28]. https://openreview.net/forum?id=c8P9NQVtmnO.

[14] KARNIADAKIS G E, KEVREKIDIS I G, LU L, et al. Physics-informed machine learning[J]. *Nature Reviews Physics*, 2021, 3: 422–440. DOI: 10.1038/s42254-021-00314-5.

---

## 稿件证据说明（投稿排版时删除）

- 表 1 来自 `experiments/rba_qpinn_submission_v1/seed_42/processed/dataset_summary.json`。
- 表 3、表 4 与统计检验来自 `experiments/rba_qpinn_submission_v1/aggregate/`。
- 每个种子目录保存独立的配置、掩码、Raw/Corrected 预测、模型权重、逐样本指标和汇总指标，运行时启用不可覆盖检查。
- 训练参数：seeds 42–46、40 epochs、batch size 8、hidden dimension 128、depth 4、latent dimension 24、learning rate \(10^{-3}\)。
- Raw 指标是论文主结果；IDW-Corrected 指标仅作为统一后处理后的系统性能。
