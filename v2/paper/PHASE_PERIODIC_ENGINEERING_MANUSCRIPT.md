# 周期相位网络的渗透率场内补全与自适应加权分析

**Within-Field Permeability Completion Using a Phase-Periodic Network: Representation and Adaptive Weighting**

## 摘要

针对部分网格位置已知条件下的非均质渗透率场补全，构建由可学习相位投影、周期响应和门控残差主干组成的坐标网络，将样本级工程条件与场标识共同用于预测未观测位置，并考察动态损失平衡、训练点残差加权和神经元效用筛选的组合效果。在128个32×32归一化对数渗透率场上，以约70%、10%和20%的比例划分训练、验证及空间留出位置，采用5个随机种子比较6种配置。归档结果表明，固定损失权重的周期相位网络取得最低原始输出误差，RMSE为0.0845±0.0014，较Fourier对照网络降低51.47%；完整自适应配置的RMSE升至0.2477±0.0041，场均值先验偏差则由对照网络的0.1228降至0.0395。结果显示，局部恢复精度与整体先验一致性并不必然同步改善，附加机制的收益取决于目标分配和训练配置。由于完整场均值参与先验标定，且架构对照同时改变多个结构因素，本文结果属于场内补全的探索性证据，不构成无泄漏泛化或单一模块因果作用的证明。

**关键词**：渗透率场；场内补全；周期相位网络；自适应加权；统计先验

## Abstract

A coordinate network is developed for completing heterogeneous permeability fields from observations at a subset of grid locations. Learnable phase projections, periodic responses, and a gated residual backbone are combined with field-level engineering conditions and learned field identifiers. Dynamic loss balancing, training-point residual weighting, and Monte Carlo neuron-utility screening are evaluated through six configurations. The archived dataset contains 128 normalized log-permeability fields on 32×32 grids. Approximately 70%, 10%, and 20% of the locations are assigned to training, validation, and spatial holdout sets, respectively, across five random seeds. The phase-periodic network with fixed loss weights achieves the lowest raw RMSE, 0.0845±0.0014, representing a 51.47% reduction relative to the Fourier comparator. The fully adaptive configuration increases RMSE to 0.2477±0.0041 while reducing the field-mean prior discrepancy from 0.1228 to 0.0395. These observations indicate that local reconstruction accuracy and agreement with a global prior need not improve together. However, full-field target means were used to calibrate the prior, and the architecture comparison changes several components simultaneously. The results are therefore exploratory evidence for within-field completion, rather than evidence of leakage-free generalization or isolated component effects.

**Keywords**: permeability field; within-field completion; phase-periodic network; adaptive weighting; statistical prior

## 0 引言

渗透率的空间非均质性是地下流动建模需要处理的基本问题。局部参数变化及空间相关结构影响模型对地层的描述，仅用一个平均值难以代替完整场信息[1]。当已有网格中部分位置缺测时，首先需要解决的是如何利用已知位置恢复同一场内的未知值。这一任务与根据压力响应反演渗透率、预测全新储层参数并不相同：它不直接求解流动方程，但补全结果决定了后续模型能够获得怎样的参数场。因此，评价补全方法既应关注总体误差，也应检查局部偏差和适用的信息条件。

空间补全可采用距离插值、地质统计方法或连续坐标表示。逆距离加权根据邻近观测及距离衰减构造估计，参数与邻域选择会影响插值结果[2]；地质统计方法则借助协方差或变差函数表达空间相关性，相关工具已将随机场、变差函数估计和Kriging纳入统一计算流程[3]。坐标网络提供了另一种表示方式：以位置为输入，由共享参数描述场的连续变化。对多个已登记场，还可引入场级条件与可学习标识，使同一网络表达不同场的空间分布。此时需要检验的不是网络能否记住已知点，而是这种表示能否降低未观测位置的恢复误差。

坐标网络的表达和训练行为与输入特征、激活函数有关。Rahaman等[4]指出，神经网络在相应研究条件下呈现先学习低频成分的频谱偏置；Tancik等[5]通过Fourier特征映射改善低维信号的高频拟合；Sitzmann等[6]采用周期激活表达连续信号及其导数。这些工作说明周期表示值得考察，但不能据此推断任意周期结构都优于常规网络。对于特定补全任务，仍需结合空间误差和对照设置判断其作用。

另一类思路是在数据拟合之外加入领域约束。物理信息神经网络将方程、初边值条件和观测误差共同纳入求解目标[7]。宋家豪等[8]通过周期维度上的傅里叶变换构造频域方程，展示了从数学问题本身改变求解形式的路径。与之不同，坐标的周期特征并未变换控制方程；若数据只支持场级经验关系，其约束也只能解释为统计先验。研究中应把函数表示、物理方程和经验正则化分开，避免用同一名称覆盖不同的信息来源。

多项约束还带来优化资源如何分配的问题。梯度尺度失衡[9]、逐点可训练权重[10]以及残差驱动注意力[11]分别对应不同的训练困难。徐宝昌等[12]将自适应权重用于两相流环空压力预测，使工程对象、方程约束和困难区域的训练联系起来。不过，改变目标权重并不自动意味着所有评价指标同时改善，尤其当一个目标约束全局均值、另一个目标要求恢复局部差异时，更需要直接检验两者的关系。

本文围绕渗透率场内补全研究两个相互关联的问题：周期相位架构与固定Fourier特征对照的误差有何差异；在周期相位架构上增加目标平衡、空间加权和神经元筛选后，局部精度与先验一致性如何变化。采用同一批场上的五种子实验，将网络直接输出与观测残差后处理分开分析，并结合逐场误差及训练记录讨论原因。研究重点是具体配置下的表示效果和目标取舍，不预设完整模块组合一定最优。

## 1 工程问题与数学描述

### 1.1 场内补全任务

设第 \(s\) 个场为 \(u_s:\Omega\rightarrow\mathbb R\)，其中 \(\Omega=[-1,1]^2\)，网格大小为 \(H\times W=32\times32\)。\(u_s\) 表示归一化对数渗透率，不具有绝对渗透率单位。每个场的网格位置划分为互斥的训练集 \(\mathcal T_s\)、验证集 \(\mathcal V_s\) 和测试集 \(\mathcal E_s\)，三者覆盖全部网格。模型利用训练位置上的场值优化参数，在验证位置上选择检查点，最后评价测试位置的误差。

各场配有条件向量 \(\mathbf c_s=[P_s,G_s,\mu_s,\phi_s,Q_s]\)，分别表示压力、压力梯度、黏度、孔隙度和流量。将条件按已登记场的均值与标准差标准化，得到 \(\widetilde{\mathbf c}_s\)，并为各场设置24维可学习嵌入 \(\mathbf e_s\)。位置 \(i\) 处的输入和输出写为

$$
\mathbf z_{s,i}=[x_i,y_i,\widetilde{\mathbf c}_s,\mathbf e_s]^\mathsf T
\in\mathbb R^{31},\qquad
\widehat u_{s,i}=f_\Theta(\mathbf z_{s,i}).
\tag{1}
$$

场嵌入由同一场的训练观测优化，因而这里的未知对象是位置，而不是整幅场。该设置对应已登记网格的缺测补全；新场进入时尚无已训练嵌入，不能直接沿用本实验结果评价其预测能力。

从信息组织看，一个位置的预测同时依赖共享网络、该场的条件和嵌入；同场已知点通过训练过程影响这些参数，而不是在每次查询时直接作为输入集合。这使网络能够重复查询同一场的不同位置，但也将建模成本与场的登记、训练联系起来。测试掩码只用于从保留的参考场计算误差，不代表额外可用的现场观测。后文将这种“位置留出、对象已知”的关系作为解释全部结果的前提。

### 1.2 工程条件与场均值先验

为考察场级信息对补全的作用，以工程条件构造经验指标

$$
I_s=\ln\frac{Q_s}{Q_0}
+\ln\frac{\mu_s}{\mu_0}
-\ln\frac{G_s}{G_0}
+2.8\frac{\phi_s-\phi_0}{\phi_0}
-\kappa(P_s-P_0),
\qquad m_s=aI_s+b.
\tag{2}
$$

其中参考量 \(Q_0=120\ \mathrm{m^3/d}\)、\(\mu_0=1\ \mathrm{cP}\)、\(G_0=0.012\ \mathrm{MPa/m}\)、\(\phi_0=0.22\)、\(P_0=30\ \mathrm{MPa}\)，\(\kappa=0.015\ \mathrm{MPa^{-1}}\)。\(m_s\) 为归一化场均值的拟合目标。以上系数是计算中采用的经验设置，不是由当前数据独立识别的物理常数。

该先验描述场级统计关系，不涉及压力分布、源汇项或局部通量。现有数据也不足以建立具有实际尺度和边界条件的局部Darcy约束，因此本文将其用于讨论先验引导的补全，而非控制方程求解。归档中 \(a,b\) 的估计使用了完整场均值，其信息使用范围及影响在第3.1节说明。

场均值与局部场值是两种不同信息：多幅空间分布可以具有相同均值，单独满足均值条件不能确定缺测位置。因此，先验在本任务中只提供额外的整体约束，仍需由观测数据和表示结构确定空间变化。它是否有助于补全，应由留出误差检验，而非由先验自身的拟合程度判断。

## 2 周期相位网络与自适应训练

### 2.1 周期相位表示

将输入投影为 \(q=6\) 个相位通道，并通过可学习混合矩阵构造耦合响应：

$$
\boldsymbol\theta=W_e\mathbf z+\mathbf b_e,\qquad
\boldsymbol\eta=\sin(M^\mathsf T\boldsymbol\theta).
\tag{3}
$$

其中 \(W_e\in\mathbb R^{6\times31}\)，\(M\in\mathbb R^{6\times6}\)。周期特征与原始输入拼接为

$$
\boldsymbol\psi(\mathbf z)=
[\sin\boldsymbol\theta,\cos\boldsymbol\theta,
\sin(\boldsymbol\theta+\boldsymbol\eta),
\cos(\boldsymbol\theta-\boldsymbol\eta)]^\mathsf T,\qquad
\boldsymbol\xi=[\mathbf z,\boldsymbol\psi]^\mathsf T\in\mathbb R^{55}.
\tag{4}
$$

固定Fourier特征预先规定坐标频率；式(3)则使坐标、条件与场嵌入共同参与相位投影。混合响应提供非线性周期组合，但这种表示并不要求目标场在实际边界上周期连续。全部计算采用经典张量运算，不涉及量子线路或量子硬件。

首层同时设置正弦分支和门控分支：

$$
\mathbf h_0=
\sin\!\left[\sigma(W_g\boldsymbol\xi+\mathbf b_g)\right]
\odot
\sin\!\left[\omega(W_0\boldsymbol\xi+\mathbf b_0)\right],
\qquad \omega=18.
\tag{5}
$$

\(\sigma\) 为Sigmoid函数，\(\odot\) 表示逐元素乘法。门控输出再次经过正弦函数，与正弦首层响应相乘；该形式按实际实现保留。后续3个隐藏层采用

$$
\mathbf h_\ell=\mathbf h_{\ell-1}
+\operatorname{SiLU}(W_\ell\mathbf h_{\ell-1}+\mathbf b_\ell),
\quad \ell=1,2,3;\qquad
\widehat u=\mathbf w_o^\mathsf T\mathbf h_3+b_o.
\tag{6}
$$

隐藏宽度为128。相位投影负责形成输入相关的周期响应，残差主干在此基础上调整场值映射，两者共同构成本文比较的架构。

保留原始输入的目的，是使后续层既能利用周期响应，也能直接访问位置与场级信息，不必把所有变化都编码为周期分量。由于相位投影同时接收条件和嵌入，相同坐标在不同场中可以形成不同响应。不过，嵌入同样能够区分场，当前实验尚未分离工程条件与场标识各自的作用，不能将条件输入的存在直接解释为网络已经学得对应的物理依赖。

### 2.2 数据拟合与先验正则化

对包含 \(B\) 个场的批次，记训练位置的非负权重为 \(\lambda_{s,i}\)。数据项为

$$
\mathcal L_d=
\frac{\sum_{s=1}^{B}\sum_{i\in\mathcal T_s}
\lambda_{s,i}(\widehat u_{s,i}-u_{s,i})^2}
{\sum_{s=1}^{B}\sum_{i\in\mathcal T_s}\lambda_{s,i}+\varepsilon},
\qquad \varepsilon=10^{-8}.
\tag{7}
$$

不启用空间加权时，\(\lambda_{s,i}=1\)。场均值项为

$$
\overline{\widehat u}_s=\frac1{HW}\sum_{i=1}^{HW}\widehat u_{s,i},
\qquad
\mathcal L_m=\frac1B\sum_{s=1}^{B}
(\overline{\widehat u}_s-m_s)^2.
\tag{8}
$$

另以相邻网格差分构造平滑项，以相对边界均值差构造边缘正则项：

$$
\begin{aligned}
\mathcal L_s&=\langle(\Delta_x\widehat u)^2\rangle
+\langle(\Delta_y\widehat u)^2\rangle,\\
\mathcal L_b&=\frac1B\sum_{s=1}^{B}
\left[
(\overline{\widehat u}_{s,L}-\overline{\widehat u}_{s,R})^2
+(\overline{\widehat u}_{s,T}-\overline{\widehat u}_{s,B})^2
\right].
\end{aligned}
\tag{9}
$$

尖括号分别表示在批次及相应邻接位置上取平均；差分未除以物理网格间距。下标 \(L,R,T,B\) 表示左、右、上、下边界，横线为边界内平均。因此 \(\mathcal L_b\) 只要求相对边缘的平均预测接近，不是实际压力或流量边界条件。

总目标为

$$
\mathcal L=w_d\mathcal L_d+w_m\mathcal L_m
+w_s\mathcal L_s+w_b\mathcal L_b.
\tag{10}
$$

固定权重配置取 \((w_d,w_m,w_s,w_b)=(1,0.1,0.002,0.001)\)。该设置以点值拟合为主，对均值与空间平滑施加较弱正则化。

四项损失分别作用于观测位置、完整预测场和网格邻接或边缘关系。数据项提供局部数值依据，平滑项限制相邻位置变化，均值项与边缘项则约束更低维的统计量。它们的计算数量、变化尺度和最小值位置未必一致，因而数值较小不等于对补全更重要。使用多个正则项的目的不是使每一项尽可能接近零，而是在保留观测约束的同时考察可接受的空间恢复。

### 2.3 动态目标权重

动态配置以四项损失的尺度和相对变化更新权重。设 \(L_{k,t}\) 为第 \(t\) 轮最后一个训练批次的第 \(k\) 项损失，评分为

$$
g_{k,t}=\frac1{L_{k,t}+\varepsilon}
+\operatorname{clip}\left(
\frac{L_{k,t-1}-L_{k,t}}{|L_{k,t-1}|+\varepsilon},
-0.5,0.5\right).
\tag{11}
$$

首轮仅使用倒数项。以 \(a_{k,0}=1/2\) 初始化幅值，再按

$$
a_{k,t}=\sqrt{0.7a_{k,t-1}^2+
0.3\,\operatorname{softmax}(\mathbf g_t)_k},
\qquad w_{k,t+1}=4a_{k,t}^2
\tag{12}
$$

更新，并作幅值归一化。权重非负且总和为4，用于下一轮梯度训练。它替代固定权重，而非乘在固定权重之上，因此四项初始权重均为1。倒数评分会放大数值较小的损失，其与局部拟合目标是否协调，需要结合训练记录检验。

该规则没有直接计算各目标的梯度范数，因此不等价于梯度平衡。其输入也是末批损失而非整轮平均，批次组成会影响评分；记录权重时还需区分本轮实际使用值和轮末为下一轮生成的值。本文在结果分析中按这一时间关系读取训练记录，避免用训练结束后的更新解释此前选出的模型。

### 2.4 残差加权与神经元筛选

空间加权只根据训练位置的绝对残差 \(r_{s,i}=|\widehat u_{s,i}-u_{s,i}|\) 更新。记 \(\bar r_s\) 为该场训练残差均值，则

$$
\lambda_{s,i}^{t+1}=
\operatorname{clip}_{[1,4]}
\left[
0.7\lambda_{s,i}^{t}
+0.3\left(1+0.2\frac{r_{s,i}}{\bar r_s+\varepsilon}
+\delta_{s,i}\right)
\right].
\tag{13}
$$

每5轮对高残差训练点作密度分组，邻域半径为2个网格、最少点数为4；每个区域的包围框向外扩展1格，并对覆盖的训练位置增加0.2的区域增量 \(\delta_{s,i}\)。候选阈值从最大训练残差的0.8倍起，每次乘0.8下调，最多检查8次，尽量取得至少32个候选点。该过程不新增或替换采样位置，改变的是已有训练点的贡献。

神经元筛选以正弦首层的激活 \(v_j\) 估计效用。每次从打乱的训练加载器取4个批次，计算

$$
U_j=\frac14\sum_{\tau=1}^{4}
\left[
\overline{|v_j|}_{\,\tau}
\sqrt{\operatorname{Var}_{\tau}(v_j)+\varepsilon}
\right].
\tag{14}
$$

训练前20轮中，每轮将低于效用5%分位阈值的首层神经元权重按标准差0.12的零均值正态分布重置，偏置置零，不改变宽度。这一Monte Carlo批采样筛选简称MC。当前实现不清除被重置参数的优化器动量，因此需要同时考虑通道更新和优化扰动，而不能只按激活效用解释结果。

### 2.5 观测残差后处理

网络直接输出记为Raw。为区分表示能力与局部观测利用，再对各配置统一施加逆距离加权残差校正。对未观测位置 \(i\)，取同一场内8个最近训练位置：

$$
\Delta_{s,i}=
\frac{\sum_{j\in\mathcal N_8(i)}
(d_{ij}^{2}+10^{-6})^{-1}(u_{s,j}-\widehat u_{s,j})}
{\sum_{j\in\mathcal N_8(i)}(d_{ij}^{2}+10^{-6})^{-1}+10^{-8}},
\qquad
\widehat u^{\,c}_{s,i}=\widehat u_{s,i}+0.95\Delta_{s,i}.
\tag{15}
$$

\(d_{ij}\) 为网格索引间的欧氏距离，训练位置直接保留观测值。归档程序随后使用全部真实场的全局最小值和最大值裁剪输出，该范围包含留出信息。因此校正结果只作为后处理行为的探索性比较，不作为无泄漏系统性能。

## 3 实验设计与评价方法

### 3.1 数据及信息使用范围

实验采用128个32×32场，共131072个位置。来源标识均对应同一个上游 `gan` 目录；目录名称不足以证明具体生成机制，当前归档也未完整保留上游协方差参数和生成过程。本文据此使用“归档数值场”的表述，不将其作为现场实测数据或多来源验证。

五项工程条件的取值范围分别为：压力19.7594-41.8335 MPa、压力梯度0.00645-0.01658 MPa/m、黏度0.6792-1.3856 cP、孔隙度0.1727-0.2689、流量69.0806-167.9973 m³/d。空间掩码由随机数按0.7、0.1和0.2的概率生成；种子42对应训练91928点、验证13070点、测试26074点。训练观测较密，不应将这一设置等同于低观测率场景。

式(2)中的线性标定以各场完整真实均值为响应量，得到 \(a=-0.0910008\)、\(b=0.0001717\)。由于真实均值包含验证与测试位置，留出信息经先验目标进入训练；各模型共用该标定并不能消除其影响。本文如实报告这一既有协议，所有结果均作探索性解释。严格的观测补全验证需要改用训练观测或独立训练场确定标定参数，并重新训练，不能仅通过改变评价措辞修复。

### 3.2 比较配置与训练流程

Fourier对照网络采用频率集合 \(\{1,2,4,8\}\) 的固定坐标特征、SiLU激活、LayerNorm和4个各含两层线性变换的残差块。周期相位网络采用第2.1节结构，两者共享条件与场嵌入输入，但不共享完整主干。配置关系见表1，参数量包含128个场的可学习嵌入。

**表1 比较配置及组件**

| 代号 | 结构 | 动态目标权重 | 残差空间加权 | MC筛选 | 参数量 |
| --- | --- | --- | --- | --- | ---: |
| F | 固定Fourier对照 | 无 | 无 | 无 | 142977 |
| P | 周期相位网络 | 无 | 无 | 无 | 67301 |
| P-L | 周期相位网络 | 有 | 无 | 无 | 67301 |
| P-L-M | 周期相位网络 | 有 | 无 | 有 | 67301 |
| P-L-R | 周期相位网络 | 有 | 有 | 无 | 67301 |
| P-L-R-M | 周期相位网络 | 有 | 有 | 有 | 67301 |

两类结构同时改变编码、激活、归一化和门控，因此F与P是完整架构比较。其余配置用于观察当前累积路径，不构成完整正交消融，不能据此独立估计所有模块的主效应。

**表2 统一训练设置**

| 项目 | 设置 |
| --- | --- |
| 随机种子 | 42、43、44、45、46 |
| 批大小与训练轮数 | 每批8个场，40轮 |
| 隐藏宽度与场嵌入 | 128与24 |
| 优化器 | AdamW，权重衰减 \(10^{-5}\) |
| 学习率 | 初值 \(10^{-3}\)，40轮余弦退火 |
| 梯度范数裁剪 | 1.0 |
| 验证选择 | 验证数据项与三项正则组成的固定权重总损失最低 |
| 计算设备 | CPU |

每轮完成梯度更新后，按配置更新动态权重、执行MC筛选和残差权重计算，再评价验证损失并保存检查点。五次运行共享同一批场，但重新生成掩码和随机状态；同一种子内各方法共用持续推进的随机加载器，批次顺序并非严格配对。归档另保存了环境、权重及预测数组；聚合时记录的平台为macOS arm64，该记录不等同于完整的训练依赖锁定文件。

### 3.3 指标与统计口径

以原始输出的空间留出RMSE为主要指标，同时报告MAE和 \(R^2\)。令 \(\mathcal E=\bigcup_s\{(s,i):i\in\mathcal E_s\}\)，其位置总数为 \(N_E\)，则

$$
\begin{aligned}
\mathrm{RMSE}&=\sqrt{\frac1{N_E}\sum_{\mathcal E}(\widehat u-u)^2},\\
\mathrm{MAE}&=\frac1{N_E}\sum_{\mathcal E}|\widehat u-u|,\\
R^2&=1-\frac{\sum_{\mathcal E}(\widehat u-u)^2}
{\sum_{\mathcal E}(u-\bar u_{\mathcal E})^2}.
\end{aligned}
\tag{16}
$$

另定义场均值先验偏差

$$
D_m=\frac1S\sum_{s=1}^{S}
|\overline{\widehat u}_s-m_s|,\qquad S=128.
\tag{17}
$$

\(D_m\) 在完整预测场上计算，与测试点指标的计算域不同，也不度量局部守恒。表中给出5个种子的均值±样本标准差；逐场分布仅作描述。种子间复用数据、场内位置存在依赖，均不能扩大为独立样本数。比较模型由现有结果选择，相关区间只描述这批探索性重复，不作预注册确认性检验解释。

## 4 结果分析与工程讨论

### 4.1 原始输出与逐场差异

表3中P的RMSE为0.0845，低于F的0.1741；MAE由0.1317降至0.0591，\(R^2\) 由0.7237升至0.9349。按未舍入的均值计算，RMSE和MAE分别降低51.47%和55.12%。由于三个指标均来自未观测测试位置，这一变化反映场内补全差异，而不是仅有训练拟合改善。

**表3 原始输出的探索性比较，5个种子的均值±样本标准差**

| 配置 | RMSE | MAE | \(R^2\) | \(D_m\) |
| --- | ---: | ---: | ---: | ---: |
| F | 0.1741±0.0021 | 0.1317±0.0014 | 0.7237±0.0058 | 0.1228±0.0003 |
| P | **0.0845±0.0014** | **0.0591±0.0009** | **0.9349±0.0025** | 0.1209±0.0001 |
| P-L | 0.2309±0.0040 | 0.1809±0.0034 | 0.5141±0.0168 | 0.0600±0.0016 |
| P-L-M | 0.2553±0.0048 | 0.2020±0.0035 | 0.4055±0.0253 | **0.0339±0.0030** |
| P-L-R | 0.2284±0.0015 | 0.1794±0.0010 | 0.5245±0.0033 | 0.0627±0.0036 |
| P-L-R-M | 0.2477±0.0041 | 0.1966±0.0034 | 0.4406±0.0212 | 0.0395±0.0035 |

注：RMSE、MAE和 \(D_m\) 越低越好，\(R^2\) 越高越好；\(D_m\) 是对经验目标的偏差，不是对真实场均值的误差。全部配置受第3.1节先验标定的信息使用限制。

五次运行中P相对F的RMSE改善方向一致，平均绝对差为0.08962。以种子为单位重采样得到的探索性bootstrap 95%区间为[0.08701，0.09184]；该区间仅反映当前数据上的随机划分与训练变化，不能覆盖数据来源变化和先验标定偏差。

为检查总体改善是否只由少数场贡献，图1纳入全部128个场，先对每个场的五次RMSE取平均，再按F的误差排序。所有场的P误差均较低。进一步查看640个场与种子的配对记录，差异方向同样一致，单场单次RMSE的中位数由0.1519降至0.0611。这里的逐场平均与式(16)先汇总平方误差再开方的口径不同，二者不应混用。

![图1 全部场的五次平均空间留出RMSE](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v2/paper/figures/manuscript/field_average_rmse_distribution.png)

**图1 全部128个场的原始输出误差。** 每个场取5次运行RMSE的算术平均，按Fourier对照误差排序，未删除任何场；阴影表示同场两种架构的误差差值，不是置信区间。

### 4.2 空间恢复与表示方式

图2给出种子42中的三个案例。周期相位网络较好地保留了所示场的高低值变化，Fourier对照在部分极值附近出现较大偏差。场80的底部高值区中，两者差异尤为直观；但这一局部观察只能说明所示案例中的场值恢复，不能直接证明连通结构或下游流动响应得到改善。

![图2 原始输出的空间补全及测试位置误差](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v2/paper/figures/manuscript/raw_spatial_comparison.png)

**图2 种子42的空间补全案例。** 根据P相对F的逐场测试RMSE改善，后验选取接近25%、50%、75%分位数的场35、39和80，不代表预先指定的典型样本。预测列为完整原始输出，未把训练位置替换为真实值；误差列只显示测试位置，空白处不表示零误差。误差色标上限取所选案例误差的98%分位数，超过上限的值显示为饱和颜色。

从表示形式看，可学习相位能够改变输入投影方向，正弦首层和残差变换则进一步调整函数表达，这为结果提供了一种解释路径。然而F本身已含Fourier特征，P又同时改变多个结构因素，故不能将改善简单解释为“使用周期函数”或“相位混合单独起效”。对于工程补全，现有证据支持优先保留该架构作为后续对照候选；对具体频率机制的判断仍需要等主干实验及频谱诊断。

### 4.3 自适应组合为何没有持续改善

加入动态目标权重后，P-L的RMSE升至0.2309，而 \(D_m\) 降至0.0600。继续加入MC得到最低的 \(D_m=0.0339\)，却同时得到最高的RMSE。完整配置的RMSE也高于F。这说明在本组配置中，接近场均值先验并未转化为更准确的局部恢复；若只以先验指标评价模型，会遗漏补全任务最直接的误差。

这一现象与目标权重的变化相一致。固定配置将平滑项权重设为0.002，动态配置则从1开始，且倒数评分容易偏向数值已经较小的损失。P-L五次运行在进入第40轮时，平滑权重约为4，其余各项约为 \(9.10\times10^{-7}\)，表明后期训练已几乎由平滑目标支配。与此同时，P-L五次均在第6轮取得最低验证总损失。因此末轮权重只能说明训练后期的偏移，不能当作表3所用检查点的权重，更不能单凭这一记录完成误差成因的证明。

P-L-R相对P-L的RMSE由0.2309降至0.2284，改变量较小，且未恢复P的水平。式(13)改变的是数据项内部的位置权重；当全局数据项权重被压低时，空间重新分配未必能弥补目标之间的失衡。MC又在训练前半程重置部分通道，同时保留旧优化器动量，可能引入额外扰动。这些都是应进一步检验的解释，而不是由累积比较直接识别的独立效应。

从任务设计看，局部观测、场均值先验和平滑要求应分别评价。需要先确认先验与数据是否相容，再判断权重如何分配；仅使总损失下降，或使某个正则项更小，都不足以判断缺测位置是否恢复得更好。

### 4.4 局部残差校正的收益与代价

表4列出统一后处理后的结果。P仍取得最低RMSE，但F从原始输出的第二位变为校正输出中误差最高的配置。校正说明网络训练残差在空间上仍可被邻近观测利用，同时也表明网络排序不一定等同于完整后处理系统的排序。

**表4 含全数据范围裁剪的残差校正结果，仅作探索性比较**

| 配置 | 校正RMSE | 校正MAE | 校正 \(D_m\) |
| --- | ---: | ---: | ---: |
| F | 0.03057±0.00091 | 0.02221±0.00066 | 0.13266±0.00008 |
| P | **0.01915±0.00018** | **0.01131±0.00011** | 0.13259±0.00004 |
| P-L | 0.02603±0.00047 | 0.01800±0.00033 | 0.13172±0.00003 |
| P-L-M | 0.02676±0.00021 | 0.01887±0.00017 | 0.13121±0.00006 |
| P-L-R | 0.02590±0.00034 | 0.01792±0.00017 | 0.13174±0.00006 |
| P-L-R-M | 0.02641±0.00039 | 0.01857±0.00025 | 0.13128±0.00007 |

P相对校正后F的RMSE降低37.37%，但六种配置的 \(D_m\) 均回到约0.13。后处理通过局部观测修正场值，并不维持原有先验目标，二者变化方向再次说明评价目标不同。由于输出裁剪读取了全部真实场范围，表4不能用于证明可部署系统的无泄漏精度；还需修复裁剪，并加入纯IDW、Kriging等空间基线，才能判断网络对混合方案的净收益。

计算代价也应分开看。以种子42为例，F和P的训练阶段分别耗时22.22 s和16.32 s，P-L-R与完整配置分别为23.48 s和23.58 s。计时包含训练循环内的验证和自适应操作，不包含最终预测、IDW后处理及上游数据准备。由于模型参数量不同、未形成完整的重复性能基准，这些记录仅用于描述当前小规模CPU运行，不能据此宣称工程实时性或通用加速。

### 4.5 对工程使用范围的判断

当前结果最接近的任务是已登记、观测较密的规则网格缺测补全。五次随机划分说明同一数据上的变化情况，却没有覆盖观测稀疏化、成片缺测、测量噪声或新场输入。工程条件作为场级输入，也不能代替真实空间压力与边界信息。因而较低的归一化场值误差尚不足以推出压力预测、流量估计或生产决策的改善。

进一步判断方法价值，应先消除完整场均值标定和输出裁剪中的留出信息，再在一致主干和可比预算下检验表示及权重机制。在这一基础上，改变观测数量与分布、引入实际物理尺度，并用下游流动响应评价补全误差的后果。上述验证决定了结论能够扩展到哪里，不应由现有图像的平滑程度或单一平均指标替代。

## 5 结论

本文针对已登记渗透率场的空间留出补全，构建周期相位坐标网络，并比较动态目标权重、残差空间加权与神经元筛选的组合。基于既有数据可得以下认识。

1. 在128个32×32场、约70%训练观测和40轮预算下，固定权重周期相位网络的原始RMSE为0.0845±0.0014，较Fourier对照降低51.47%，五次运行改善方向一致。该结果属于架构整体比较，不能单独归因于相位编码。
2. 自适应机制的累积加入没有带来单调收益。完整配置的RMSE为0.2477±0.0041，而均值先验偏差降至0.0395，说明整体统计约束与局部恢复质量需要分别评价。
3. 局部残差校正改变了部分模型的排序，也削弱了与原训练先验的一致性。评价工程补全方案时，应区分网络本体、后处理收益和实际信息需求。

完整场均值标定与全数据范围裁剪限制了当前证据强度。本文结论为场内补全的探索性发现；确认其方法贡献和工程适用性，仍需无泄漏重跑、受控结构比较及不同观测条件下的验证。

## 数据与代码说明

运行配置、处理后数组、空间掩码、模型权重、预测结果及分析脚本保存在项目实验归档中，可用于复核已保存数据上的计算。完整上游生成链尚不齐全，当前归档不等同于能够从原始数据生成开始执行的完整复现包。

## 参考文献

[1] DE MARSILY G, DELAY F, GONÇALVÈS J, et al. Dealing with spatial heterogeneity[J]. Hydrogeology Journal, 2005, 13(1): 161-183. DOI: 10.1007/s10040-004-0432-3.

[2] LU G Y, WONG D W. An adaptive inverse-distance weighting spatial interpolation technique[J]. Computers & Geosciences, 2008, 34(9): 1044-1055. DOI: 10.1016/j.cageo.2007.07.010.

[3] MÜLLER S, SCHÜLER L, ZECH A, et al. GSTools v1.3: A toolbox for geostatistical modelling in Python[J]. Geoscientific Model Development, 2022, 15(7): 3161-3182. DOI: 10.5194/gmd-15-3161-2022.

[4] RAHAMAN N, BARATIN A, ARPIT D, et al. On the spectral bias of neural networks[C]//Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019, 97: 5301-5310.

[5] TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C]//Advances in Neural Information Processing Systems. 2020, 33: 7537-7547.

[6] SITZMANN V, MARTEL J N P, BERGMAN A W, et al. Implicit neural representations with periodic activation functions[C]//Advances in Neural Information Processing Systems. 2020, 33: 7462-7473.

[7] RAISSI M, PERDIKARIS P, KARNIADAKIS G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707. DOI: 10.1016/j.jcp.2018.10.045.

[8] 宋家豪, 曹文博, 张伟伟. FD-PINN: 频域物理信息神经网络[J]. 力学学报, 2023, 55(5): 1195-1205. DOI: 10.6052/0459-1879-23-169.

[9] WANG S, TENG Y, PERDIKARIS P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J]. SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081. DOI: 10.1137/20M1318043.

[10] MCCLENNY L D, BRAGA-NETO U M. Self-adaptive physics-informed neural networks[J]. Journal of Computational Physics, 2023, 474: 111722. DOI: 10.1016/j.jcp.2022.111722.

[11] ANAGNOSTOPOULOS S J, TOSCANO J D, STERGIOPULOS N, et al. Residual-based attention in physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116805. DOI: 10.1016/j.cma.2024.116805.

[12] 徐宝昌, 张学智, 王雅欣, 等. 用于两相流环空压力预测的自适应物理信息神经网络模型[J]. 石油学报, 2023, 44(3): 545-555. DOI: 10.7623/syxb202303012.
