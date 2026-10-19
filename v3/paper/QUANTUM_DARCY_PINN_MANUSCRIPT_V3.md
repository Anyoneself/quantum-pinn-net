# 量子特征增强物理信息神经网络的渗透率场重建

## 摘要

稀疏观测下的渗透率场重建同时受到网络表示和物理约束优化的影响，量子特征替换与训练增强的作用需要在可比条件下分别检验。针对二维稳态单相 Darcy 流动，建立渗透率与辅助压力双网络物理信息神经网络，在保持编码器、读出器及压力网络结构和对应初始化一致的条件下，以参数化量子线路替换经典特征块，并对两类模型对称施加残差注意力和蒙特卡罗阈值筛选。采用 GSTools 生成合成渗透率场，仅以稀疏渗透率观测监督，参考压力由独立有限体积求解器生成并用于训练后评价。结果表明，在固定训练预算下，基础量子模型的平均测试渗透率相对 \(L_2\) 误差由经典基线的0.337094降至0.306759，降低9.00%；加入残差注意力后，量子模型的独立配置点方程残差降低24.07%，但渗透率误差仅降低0.27%。蒙特卡罗筛选单独加入量子模型未改善平均渗透率误差，并使训练时间增加40.01%。量子特征在当前合成场上提供了一定的平均重建收益，但跨场效果、物理响应和计算成本并未同步改善，不能据此推出普遍性能优势或量子硬件加速。

**关键词：** 渗透率场重建；物理信息神经网络；参数化量子线路；Darcy 方程；残差注意力；蒙特卡罗筛选

## 0 引言

渗透率的空间分布决定多孔介质中的局部传导能力，并通过压力梯度影响整体流动响应。当场值只能在有限位置获取时，需要在符合观测的同时重建未观测区域。物理信息神经网络（physics-informed neural network，PINN）将观测误差、控制方程和边界条件纳入共同优化，为参数场与状态场的联合表示提供了途径[1]。与基于微分残差的深度 Galerkin 方法和基于能量泛函的 Deep Ritz 方法类似[2-3]，其关键是通过明确的数学约束限制函数表示，而不是仅给数据拟合网络赋予物理名称。相关综述及计算框架总结了这些方法的适用形式和自动微分实现[4-5]。本文关注获得稀疏渗透率观测后对同一场进行重建，不涉及无需重新拟合的新场预测，也不同于 DeepONet 和 Fourier 神经算子所学习的跨函数映射[29-30]。

物理约束进入损失并不保证优化充分或重建准确。复合目标的梯度失衡、不同损失的收敛速率及问题参数变化均可能影响 PINN 训练[6-8]，因而基线设置和评价协议必须与模型改动一起考虑[9]。已有改进涉及不同数学位置：Fourier 特征、周期激活和多尺度表示改变坐标函数的表达[10-12]，自适应激活调整参数化与优化过程[13-14]；FD-PINN 则对周期方向上的控制方程进行离散傅里叶变换，形成频域约束[33]，不能与单纯输入编码等同。训练策略方面，自适应点权重、残差注意力、配置点重采样及残差梯度约束分别改变损失分配、采样位置或约束内容[15-18]。当表示与训练策略同时变化时，完整模型误差降低不能直接归因于其中某个模块。

参数化量子线路通过输入编码、可训练酉变换和可观测量读出构造特征函数[20]，可微线路、连续变量量子 PINN 及量子—经典混合网络已被用于微分方程求解[21-23]。量子编码决定可表达函数的部分结构，其测量期望值与经典 Fourier 表示存在联系[24]，因此“引入量子线路”本身不足以证明表示优势；较大线路的梯度集中还可能带来训练困难[25]。已有研究已将量子—经典 PINN 用于单相储层渗透率反演[31]，本文不以这一概念组合的首次提出作为贡献。更具体的问题是：在相同观测条件和可比外部网络下，替换渗透率特征块能产生多大的重建收益，常用训练增强是否还能改善该收益，其代价又是什么？

围绕上述问题，本文构建仅由渗透率观测监督的 Darcy 双网络 PINN，以经典压力网络表示辅助未知场；保持渗透率编码与读出接口不变，比较全连接特征和可训练量子特征；对两类模型对称应用残差注意力与蒙特卡罗配对筛选，形成完整交叉对照。数值验证分别评价留出位置的场值误差、独立配置点上的方程残差、预测渗透率导出的流动响应及模拟成本。本文的贡献在于明确的模块化实现与受控比较，以及对表示替换和训练增强实际作用的区分，不预设全部模块均产生正收益。

## 1 渗透率场重建问题

### 1.1 控制方程、边界条件与观测信息

考虑单位正方形 \(\Omega=(0,1)^2\) 上无源、稳态、单相 Darcy 流动。采用标量正渗透率，黏度、长度、渗透率和压差参考尺度经无量纲化后均取1，控制关系为

$$
\boldsymbol q=-k\nabla p,\qquad
\nabla\cdot\boldsymbol q=0,
\tag{1}
$$

其中 \(k(\boldsymbol x)>0\) 为渗透率，\(p(\boldsymbol x)\) 为压力，\(\boldsymbol q(\boldsymbol x)\) 为通量，\(\boldsymbol x=(x,y)\)。左右边界施加定压，上下边界无流：

$$
p(0,y)=1,\quad p(1,y)=0,\quad
q_y(x,0)=q_y(x,1)=0.
\tag{2}
$$

式(1)将渗透率与压力联系起来：给定渗透率后，压力场调整其空间梯度，使通量满足局部守恒。反之，仅要求通量守恒，并不能直接给出唯一的渗透率。因此，在重建渗透率时还需表示压力及其导数，但“需要压力函数”不等于“需要压力观测”。本文边界条件是预先给定的物理信息，域内压力则为辅助未知量。该设置用于研究方法在合成问题上的表现，不直接对应现场压力、渗透率单位或产量。

### 1.2 稀疏观测与场重建目标

记自然对数渗透率为 \(\ell=\log k\)，可用训练观测为

$$
\mathcal D_\ell^{\mathrm{tr}}
=\{(\boldsymbol x_i,\ell_i^{\mathrm{obs}})\}_{i=1}^{N_\ell}.
\tag{3}
$$

另设互斥的渗透率验证位置和测试位置。训练与验证阶段均无域内压力标签，完整渗透率参考和数值压力参考只在模型选择结束后进入评价。目标是在满足观测及给定物理约束的条件下，恢复未观测位置的 \(k\)，同时得到用于构造流动关系的辅助压力函数。

每个生成场均从独立模型初始化开始拟合。网络前向输入是空间坐标，不是完整渗透率图像；稀疏渗透率值通过损失提供监督，也不是拼接到所有坐标后的条件向量。这样的任务属于逐场重建，而非跨场算子学习。

在缺少内部压力观测时，多种正渗透率场可以分别对应满足式(1)-(2)的压力解。稀疏渗透率观测约束了部分场值，但不会自动消除未观测区域的全部不确定性。本文比较的是同一观测条件下的重建误差，不将物理一致性或量子特征解释为反问题唯一可辨识的保证。

## 2 量子特征增强 PINN 与自适应训练

### 2.1 经典 PINN 网络构建

经典基线采用两个坐标网络，分别表示对数渗透率和压力。坐标先线性映射为 \(\widetilde{\boldsymbol x}=2\boldsymbol x-\boldsymbol1\)，渗透率路径包含编码器 \(E\)、有界角度映射、经典特征块 \(F_C\)、二值掩码 \(\boldsymbol m\) 和读出器 \(D\)：

$$
\boldsymbol a=\pi\tanh[E(\widetilde{\boldsymbol x})],\qquad
\hat\ell(\boldsymbol x)=D[\boldsymbol m\odot F_C(\boldsymbol a)],
\qquad \hat k=\exp(\hat\ell).
\tag{4}
$$

编码器与读出器由全连接层及 Tanh 非线性构成，\(F_C\) 采用相同输入输出维度的全连接 Tanh 层。初始掩码全为1，无筛选组始终保留全部特征。指数映射保证正渗透率，避免将正值要求交给额外惩罚或预测后截断。对经典特征块也保留同一角度映射，是为了使后续模块替换具有相同输入接口。

压力网络保持为经典多层感知机，并采用

$$
\hat p(\boldsymbol x)
=1-x+x(1-x)N_p(\widetilde{\boldsymbol x}).
\tag{5}
$$

这一简单矩形域的边界嵌入与精确施加 Dirichlet 条件的思路一致[19]。无论网络参数为何，式(5)均满足左右定压条件；上下法向通量不由该式自动消失，仍需损失约束。

两个子网络并行接受坐标输入，随后通过 Darcy 通量及其散度耦合。渗透率观测直接约束式(4)，压力参数则通过物理项和无流边界项更新。经典基线的 PINN 属性来自这些真实参与优化的微分约束，而非特定激活函数或网络名称。

### 2.2 参数化量子特征模块

量子组只将式(4)中的 \(F_C\) 替换为 \(F_Q\)，其余渗透率路径及压力网络结构保持不变。编码器输出先通过 \(R_Y\) 门作用于初始态，再依次通过可训练层：

$$
|\psi(\boldsymbol a,\boldsymbol\theta)\rangle
=V_L(\boldsymbol\theta_L)\cdots V_1(\boldsymbol\theta_1)
\left[\bigotimes_{j=1}^{n_q}R_Y(a_j)\right]
|0\rangle^{\otimes n_q},
\tag{6}
$$

$$
F_Q(\boldsymbol a)
=[z_1,\ldots,z_{n_q}]^{\mathsf T},\qquad
z_j=\langle\psi|Z_j|\psi\rangle.
\tag{7}
$$

其中 \(n_q\) 为量子比特数，\(L\) 为可训练层数，\(\boldsymbol\theta\) 为旋转参数，\(Z_j\) 为第 \(j\) 个比特的 Pauli-Z 算符。每层先施加具有三个可训练角度的 Rot 门，再按固定顺序执行 CNOT 链及首尾连接。输入只在初始阶段编码一次，不进行逐层数据重上传。测量期望值是供经典读出器使用的特征，不直接等于渗透率。

![图1 经典与量子特征可替换的双网络 PINN](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig01_architecture.png)

**图1 经典与量子特征可替换的双网络 PINN 及量子线路。** （a）量子线路仅替换渗透率路径中的经典特征块，两者互斥选用，不进行分支融合。辅助压力无域内观测监督，经左右定压嵌入后输出；预测的对数渗透率、渗透率和压力共同用于构造三项损失。（b）四个量子比特先接受一次 RY 角度编码，再执行两层参数独立的旋转与纠缠操作。每个 Rot 按时间顺序由 Rz(φ)、Ry(θ)、Rz(ω) 组成；每层 CNOT 按 0→1、1→2、2→3、3→0 顺序连接，实心点为控制端，圆圈加号为目标端。虚线框仅表示重复的可训练层，不含输入编码和末端 Pauli-Z 期望值读出。RBA 作用于方程损失；MC 只屏蔽测量特征，不删除量子门。

图1中的“共同编码器、读出器与压力网络”指对应结构和初始化一致，各组在训练中不绑定参数。通过独立随机作用域初始化，特征块参数数量的差异不会改变外部层初始值。

量子特征由 PennyLane 的可微线路实现[26]，使用精确状态向量模拟、精确期望值和 Torch 反向传播。它不同于用经典正弦层作量子启发编码，也不同于在真实设备上通过有限 shots 和参数移位估计梯度[27-28]。本比较只检验这一量子参数化相对所设经典 Tanh 特征块的效果，并未证明其不能由其他经典表示近似[24]。

### 2.3 物理信息损失函数

根据网络输出构造通量与强形式残差：

$$
\hat{\boldsymbol q}=-\exp(\hat\ell)\nabla\hat p,\qquad
r=-\exp(\hat\ell)
\left(\nabla\hat\ell\cdot\nabla\hat p+\Delta\hat p\right).
\tag{8}
$$

残差求值涉及渗透率的一阶坐标导数和压力的二阶坐标导数。优化还需沿这些导数对经典与量子参数继续反向传播，因而特征模块必须保留坐标到输出的可微计算链[5,21]。本文不把量子线路输出转换为脱离计算图的固定数组后再训练读出器。

渗透率监督、物理与边界项分别定义为

$$
\mathcal L_k=\frac1{N_\ell}\sum_i
[\hat\ell(\boldsymbol x_i)-\ell_i^{\mathrm{obs}}]^2,\qquad
\mathcal L_r=\frac{\sum_i w_i r(\boldsymbol x_i^r)^2}{\sum_i w_i},
\qquad
\mathcal L_b=\frac1{N_b}\sum_h
\hat q_y(\boldsymbol x_h^b)^2,
\tag{9}
$$

其中 \(\boldsymbol x_i^r\) 为域内配置点，\(\boldsymbol x_h^b\) 为上下边界点，\(w_i\) 为残差点权重。总目标为

$$
\mathcal L=\lambda_k\mathcal L_k+
\lambda_r\mathcal L_r+\lambda_b\mathcal L_b.
\tag{10}
$$

各组采用同一组固定全局权重，基础模型中 \(w_i=1\)。训练在对数空间计算观测误差，使优化针对相对尺度变化而非仅由较大渗透率绝对值主导；最终主指标仍在 \(k\) 空间计算，不能混用两种误差。

当前损失仅有式(9)的三项。压力标签为空，不设置压力观测拟合项；左右定压条件由式(5)满足，也不另加边界惩罚。完整场的压力参考没有通过损失、权重或阈值间接进入训练。

### 2.4 训练增强与模型选择

在共同物理信息目标下，RBA 调整固定配置点的残差权重，蒙特卡罗配对消融评估读出特征并筛选候选掩码，验证观测用于掩码接受和最佳模型选择。两种增强策略按对照组分别启用或组合使用，并对经典与量子模型采用相同规则。

为改变方程残差在固定配置点之间的相对贡献，引入残差注意力（residual-based attention，RBA）的有界归一化变体[16]：

$$
u_i^{(t)}=
\frac{|r_i^{(t)}|}
{\max(\max_j|r_j^{(t)}|,\varepsilon)},\qquad
a_i^{(t)}=\gamma a_i^{(t-1)}+(1-\gamma)u_i^{(t)},
\qquad
w_i^{(t)}=1+\alpha a_i^{(t)}.
\tag{11}
$$

其中 \(\varepsilon>0\) 为分母下限，\(0\le\gamma<1\) 为衰减系数，\(\alpha\ge0\) 为权重增益，初始 \(a_i=0\)。更新时使用脱离梯度图的残差，因此权重不是通过反向传播学习的额外参数。由 \(0\le u_i\le1\) 可得 \(0\le a_i\le1\)，故 \(1\le w_i\le1+\alpha\)。

单位下限保留低残差区域的约束，式(9)按权重和归一化，避免仅由整体权重增加直接放大物理项尺度。该方法不同于通过最大化点权重构造自适应训练的方法[15]，也不改变配置点位置[17]或增加残差梯度损失[18]。

RBA 依据的是方程误差，而非渗透率观测误差。它通过共同目标影响压力、渗透率和量子门参数，不能被理解为只修正量子子模块的独立算法。其可能降低残差的动机与是否改善场值的结论分开检验。

与调整残差点权重的 RBA 不同，蒙特卡罗筛选以读出特征为对象，通过随机训练子批次上的配对消融评估其对联合目标的贡献。对于第 \(b\) 次试验，在当前掩码下计算损失 \(\mathcal L^{(b)}\)，再仅将特征 \(j\) 屏蔽，得到 \(\mathcal L_{-j}^{(b)}\)，定义

$$
d_j^{(b)}
=\frac{\mathcal L_{-j}^{(b)}-\mathcal L^{(b)}}
{\max(|\mathcal L^{(b)}|,\varepsilon)},\qquad
U_j=\overline d_j+\frac{2s_j}{\sqrt B},
\tag{12}
$$

其中 \(B\) 为试验次数，\(\overline d_j\) 和 \(s_j\) 分别为相对增量的均值和样本标准差。同一试验中全部候选使用相同的渗透率观测、配置点和边界子批次，以及冻结的 RBA 权重，避免将不同采样引起的损失变化混入单次特征比较。

候选先按 \(\overline d_j\) 升序排序，仅考虑预设比例内的特征；还需满足 \(U_j\le\tau\)、有效特征不少于下限，以及完整验证观测评分不增加，才接受新掩码。阈值和比例在实验前固定，不由测试结果调整。由于随机子批次可能重叠且试验次数有限，\(U_j\) 是工程启发式评分，不是严格置信上界；通过验证接受也不保证后续训练或测试误差单调改善。

蒙特卡罗（Monte Carlo，MC）在这里指随机子批次评估，不是量子测量采样。筛选仅屏蔽送入读出器的输出特征，不删量子门、不减量子比特，也不重置参数或 Adam 状态。被屏蔽读出的比特仍可通过纠缠影响其他特征，故不能等同于线路剪枝。经典组执行相同程序，且额外消融计算计入训练成本。

上述增强策略嵌入统一的联合训练流程。所有方法在共同预算下使用 Adam 和全局梯度裁剪。RBA 在预热后更新，MC 在预热后周期执行；具体时序见补充材料。验证评分仅为对数渗透率观测误差：

$$
S_{\mathrm{val}}
=\frac{\lambda_k}{N_{\mathrm{val}}}
\sum_{i\in\mathcal V_\ell}
[\hat\ell(\boldsymbol x_i)-\ell_i^{\mathrm{obs}}]^2.
\tag{13}
$$

从初始状态及各轮结束状态中选择评分最小者，并同时恢复参数与特征掩码。训练到统一预算后才结束，不因中途得到最佳检查点而提前停止。筛选在较晚阶段被接受，并不意味着最终恢复的较早状态也采用该掩码。

![图2 训练、验证选择与独立评价流程](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig02_workflow.png)

**图2 训练、验证选择与独立评价流程。** RBA 与 MC 按所属对照组启用，量子门参数仅在量子组中参与 Adam 更新；未达到预算时返回前向预测，不重新初始化模型。渗透率验证观测用于初始评分、候选掩码接受及最佳状态选择。预算结束后恢复最佳参数与掩码，再分别评价留出场值、辅助压力与物理残差、独立 FVM 流动响应；右侧数值参考分支只进入最终评价，测试标签和参考压力不反馈至训练。

拟合函数只接收训练与验证观测，不接收参考场。验证集同时参与掩码接受与检查点选择，这隔离了测试信息，但没有消除反复利用小验证集的适应性过拟合风险。物理评价指标不用于事后更换选模目标。

### 2.5 有限体积参考求解与独立物理评价

渗透率场的数值重建误差与其导出的流动响应误差并不等价。仅比较 \(\hat k\) 与 \(k\)，不能判断重建场是否保持了参考场的压力分布和总流量；仅考察 PINN 自身的压力输出与配置点残差，也不能将渗透率表示误差与辅助压力网络误差分开。因此，本文在网络训练之外设置独立有限体积法（finite volume method，FVM）求解器，在相同 Darcy 方程和边界条件下构造参考压力，并评价预测渗透率对应的流动响应。

FVM 将计算域划分为互不重叠的控制体，并对每个控制体积分施加质量守恒。记单元 \(V_P\) 的外法向为 \(\boldsymbol n\)，由稳态无源条件和散度定理有

$$
\int_{V_P}\nabla\cdot\boldsymbol q\,\mathrm dV
=\int_{\partial V_P}\boldsymbol q\cdot\boldsymbol n\,\mathrm dS
=\sum_{f\subset\partial V_P}F_{P,f}=0.
\tag{14}
$$

其中 \(F_{P,f}\) 为穿过单元面 \(f\) 的向外通量积分。在本文的无量纲二维问题中，按单位厚度计算，面测度为网格边长。采用正交均匀网格和单元中心压力未知量，内部相邻单元 \(P\)、\(N\) 之间的两点通量近似为

$$
F_{P\to N}=T_{PN}(p_P-p_N),\qquad
T_{PN}=\frac{|f_{PN}|}{d_{PN}}k_{PN},\qquad
k_{PN}=\frac{2k_Pk_N}{k_P+k_N}.
\tag{15}
$$

这里 \(d_{PN}\) 为两单元中心间距，\(|f_{PN}|\) 为公共面测度，\(k_P\)、\(k_N\) 为单元渗透率。两个半单元的渗流阻力串联，得到式(15)的调和平均；同一内部面在相邻单元中使用大小相等、方向相反的通量，从而在离散层面保持局部守恒。该表达适用于本文的正交网格和标量渗透率，不直接扩展为任意网格或张量渗透率的离散格式。

左右边界分别取定压值1和0。对定压边界面 \(f\)，令 \(T_{P,f}=k_P|f|/d_{P,f}\)，其中 \(d_{P,f}\) 为单元中心到边界面的半单元距离；上下无流边界的面通量直接置零。记 \(\mathcal N(P)\) 为内部相邻单元集合，\(\mathcal D(P)\) 为单元所接定压边界面集合，则式(14)离散为

$$
\begin{aligned}
&\sum_{N\in\mathcal N(P)}T_{PN}(p_P-p_N)
+\sum_{f\in\mathcal D(P)}T_{P,f}(p_P-p_f^{D})=0,\\
&\boldsymbol A(k)\boldsymbol p^{\,h}=\boldsymbol b(k).
\end{aligned}
\tag{16}
$$

其中 \(p_f^{D}\) 为给定边界压力，\(\boldsymbol p^{\,h}\) 为全部单元中心压力组成的离散解向量。组装时，内部传导系数进入对角项及相邻非对角项，定压边界项分别进入对角项和右端项；通过稀疏线性系统求解得到压力，再由面通量计算入口、出口流量。求解器的离散守恒不意味着 PINN 输出也严格守恒，两者的残差和流量平衡分别检查。

独立求解在评价中承担两项不同职责。其一，将 GSTools 生成的真实渗透率送入细网格 FVM，得到辅助压力网络的参考解；该压力仅用于训练后评价。其二，在相同粗网格上分别以 \(k\) 和 \(\hat k\) 求解压力及入口流量，使两次前向计算仅在渗透率输入上存在差异，用于评价重建场的下游响应，而不直接使用辅助压力网络的预测代替前向解。同网格比较控制了两次求解的网格设置，但不消除离散误差，因此另报告求解残差、流量平衡及粗细网格差异。

本文采用 FVM 的理由是其面通量表达与 Darcy 守恒形式直接对应，并能在当前网格上形成可检查的离散参考。需要独立数值参照的是压力和流动响应评价，而不是 PINN 的训练或渗透率预测本身；FVM 也并非实现这一评价的唯一数值方法。其输出不进入观测损失、RBA 权重、MC 阈值或检查点选择，不生成渗透率标签，也不对预测场作回填或修正。因此，引入 FVM 是为了补充场值误差之外的物理评价证据，不能将它本身解释为重建精度的提升来源。

## 3 数值验证与结果分析

### 3.1 GSTools 数据来源与参考场生成

利用 GSTools 的高斯协方差模型构造自然对数渗透率随机场[32]，统计关系写为

$$
\ell(\boldsymbol x)=m_\ell+\eta(\boldsymbol x),\qquad
\operatorname{Cov}[\eta(\boldsymbol x),\eta(\boldsymbol x')]
=C_G(\boldsymbol x-\boldsymbol x'),\qquad
k(\boldsymbol x)=\exp[\ell(\boldsymbol x)].
\tag{17}
$$

均值、方差、相关长度、各向异性及旋转参数均在生成前指定。实际实现采用 SRF 的 RandMeth 随机谱表示；它以协方差模型为依据，不是手工给定频率集合的正弦乘积级数。协方差参数描述空间相关性，不等于将标量渗透率改为张量渗透率。

先在细网格评价同一随机场实现，再抽取与粗网格中心重合的位置，获得观测和测试场值。参考压力由第2.5节的单元中心 FVM 在细网格上求解，再按重合中心抽取，不另生成随机场或插值。

参考检查包括离散方程残差、入口出口流量平衡，以及粗细网格压力和通量差异。当前粗细网格压力 RMSE 约为 \(3.03\times10^{-4}\) 至 \(4.01\times10^{-4}\)，入口通量相对差约为 \(6.40\times10^{-4}\) 至 \(9.10\times10^{-4}\)，详见补充材料S1。它们用于记录离散参考的不确定性，不构成完整网格收敛证明。全部数值场为无噪声合成数据，没有加入通道、裂缝或现场地质条件。

### 3.2 数据处理与观测划分

训练采用自然对数渗透率监督，不使用完整场样本均值、标准差或极值进行逐场归一化和裁剪。生成时指定的随机场统计参数不同于读取测试场后的统计校准，有限场实现也不要求其样本统计量恰好等于预设值。坐标的固定线性映射只调整输入范围，不对标签进行额外变换。

渗透率训练、验证和测试位置互斥。域内配置点及上下边界点只提供坐标，用于计算网络自身的物理残差，并不携带真实 \(k\) 或 \(p\)。同一场同一次重复中，全部方法使用相同观测、配置点及边界点，RBA 不改变点的位置。独立域内评价点与训练配置点分开保存。

压力训练及验证数组均为空，全部压力参考只参与训练后的辅助场评价。软件虽支持可选观测噪声，本实验并未启用，不据此讨论抗噪能力。主结果来自选中模型的原始输出，不使用插值残差校正、完整场校准或已知位置回填。

### 3.3 模型结构配置与评价方法

经典与量子模型的可比结构见表1。具体网络尺寸、参数数量、线路门序列、训练超参数及环境集中列于补充材料S2；正文关注替换位置与不可完全控制的差异。

**表1 经典与量子模型的结构及比较条件**

| 组成或条件 | 经典模型 | 量子模型 | 比较原则 |
| --- | --- | --- | --- |
| 渗透率编码与读出 | 全连接 Tanh 结构 | 同结构 | 对应初始化一致，独立更新 |
| 特征模块 | 全连接 Tanh 层 | 编码、旋转、纠缠与期望值读出 | 输入输出维度一致 |
| 压力网络 | 经典 MLP 与硬边界 | 同结构 | 无内部压力标签 |
| 观测与配置点 | 当前场的固定划分 | 同一划分 | 不增加量子组信息 |
| 优化与选模 | 共同损失、优化器、预算与验证规则 | 同规则 | 控制轮数，不控制等耗时 |
| 模型容量 | 特征块含经典权重与偏置 | 特征块含旋转角参数 | 总参数量接近但不相等 |

在渗透率测试集合 \(\mathcal T_\ell\) 上定义

$$
E_k=
\frac{\|\hat k_{\mathcal T_\ell}-k_{\mathcal T_\ell}\|_2}
{\|k_{\mathcal T_\ell}\|_2},\qquad
E_\ell=
\sqrt{\frac1{|\mathcal T_\ell|}
\sum_{i\in\mathcal T_\ell}(\hat\ell_i-\ell_i)^2}.
\tag{18}
$$

辅助压力误差 \(E_p\) 为网络压力与细网格参考抽取值的 RMSE；方程指标 \(E_r\) 为独立配置点上未加权残差的 RMSE，不是训练时的 RBA 加权损失。另将预测渗透率和真实渗透率分别送入同一粗网格 FVM，以入口总通量 \(Q_{\mathrm{FVM}}\) 定义

$$
E_Q=
\frac{|Q_{\mathrm{FVM}}(\hat k)-Q_{\mathrm{FVM}}(k)|}
{|Q_{\mathrm{FVM}}(k)|}.
\tag{19}
$$

相同粗网格比较避免将参考细化差异全部归于网络。下游压力指标为这两次粗网格 FVM 压力解在全部单元中心的 RMSE，逐次结果保留在补充材料所列档案中。它评价预测渗透率导出的压力响应，不等于压力子网络的 \(E_p\)；FVM 自身满足离散守恒，也不代表网络输出严格守恒。训练时间包括循环内验证及 MC 筛选，不包括模型构造、初始验证、数据生成和最终测试。

实验包含三个独立生成场，每场三次观测划分与初始化重复。每种方法的九次拟合共享其中的场结构，不是九个独立生成场。报告运行均值及样本标准差，并先对同场重复平均后分析场间差异；完整种子与配置见补充材料。标准差仅描述当前运行波动，不作为总体置信区间，也不以空间测试点数量支持显著性推断。

主对照与消融均使用原定统一预算。另对全部场的首次重复开展基础C、Q的配对延长预算诊断，仅改变训练上限，保持观测、初始化、模型与验证选模规则一致。该诊断用于检查短预算下的空间恢复不足，不与主对照合并统计；由于它是在检查原测试图后提出的，属于探索性分析，不作为新的独立外部验证。

### 3.4 对照方法与模块消融

经典特征 C 和量子特征 Q 分别搭配无增强、RBA、MC 及组合策略，构成表2的完整矩阵。

**表2 特征表示与训练增强的交叉对照**

| 特征表示 | 无增强 | RBA | MC | RBA+MC |
| --- | --- | --- | --- | --- |
| 经典 | C | C+RBA | C+MC | C+RBA+MC |
| 量子 | Q | Q+RBA | Q+MC | Q+RBA+MC |

Q 对 C 用于检验基础特征替换；Q+RBA、Q+MC 分别对 Q 用于检验单项增强；组合模型对 Q+RBA 用于观察 MC 的附加作用。相同增强条件下再比较量子与经典模型，以免将策略差异归给量子模块。例如，Q+MC 优于 C+MC 并不能说明 MC 本身有效，后者必须由 Q+MC 对 Q 的增量回答。

各组损失、对应外部层初始化、预算和选模规则一致，但特征块参数量及函数族不同，因此结果反映的是该模块替换及其联合优化变化，不能进一步拆分为纠缠或编码的独立效应。当前未开展纯数据网络、去纠缠线路、等参数扩展基线或等时间对照，不将这些未完成实验用于解释收益。

### 3.5 渗透率重建效果与消融结果

基础量子模型的平均 \(E_k\) 低于经典基线，表3给出全部方法而非只报告最优配置。Q 的误差为0.306759，C 为0.337094，相对降低9.00%。该百分比由运行均值之差除以经典均值得到，不是逐次相对改善的平均值。对数域 RMSE 也下降，但两种指标不具有完全相同的排序，说明训练目标与渗透率空间中的最终评价需要同时保留。

**表3 渗透率留出位置误差，均值±样本标准差**

| 方法 | \(E_k\) | \(E_\ell\) |
| --- | ---: | ---: |
| C | 0.337094±0.045467 | 0.340676±0.052468 |
| C+RBA | 0.338305±0.044029 | 0.343966±0.052075 |
| C+MC | 0.336532±0.045395 | 0.341546±0.054675 |
| C+RBA+MC | 0.337718±0.044709 | 0.345085±0.055212 |
| Q | 0.306759±0.039626 | 0.321822±0.052357 |
| Q+RBA | 0.305936±0.042956 | 0.321645±0.063531 |
| Q+MC | 0.306767±0.039631 | 0.322584±0.052760 |
| Q+RBA+MC | 0.305882±0.042895 | 0.321168±0.062986 |

注：两项指标均越小越好；每种方法均纳入全部场及对应重复。最低均值不代表统计显著或每次训练均占优。

在同场同重复的配对中，Q 有七次优于 C、两次较差；场级均值也存在方向差异，见表4。场1与场3的量子误差较低，场2的基础 Q 则高于 C。由此，平均收益不能表述为全部场上的一致优势，也不能仅依据平均值推断改善集中在哪些局部空间结构。

**表4 场内重复平均的渗透率相对 \(L_2\) 误差**

| 场 | C | Q | Q+RBA | Q+MC | Q+RBA+MC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 场1 | 0.317747 | 0.262047 | 0.258877 | 0.262047 | 0.258877 |
| 场2 | 0.318048 | 0.326464 | 0.317617 | 0.326486 | 0.317608 |
| 场3 | 0.375488 | 0.331768 | 0.341313 | 0.331768 | 0.341162 |

RBA 对量子场值误差的平均改善较小。Q+RBA 相对 Q 的 \(E_k\) 降低0.27%，但九次配对中仅三次改善、六次变差；场1和场2的平均误差下降，场3上升。少量较大的改善与多数较小的退化可以同时产生均值降低，故不能据此将 RBA 描述为稳定提高渗透率精度的方法。

MC 单独加入量子模型后，\(E_k\) 为0.306767，较 Q 略高。九次配对中八次相同、一次小幅变差，没有形成主指标收益。组合模型的均值0.305882虽为最低，但相对 Q 仅降低0.29%，同样只有三次配对改善；相对 Q+RBA 的附加降低仅0.01747%。这些差异说明量子基础特征的平均收益与增强策略增量不是同一量级。

图3将每次拟合与对应基准直接配对。基础量子替换在场1和场3的差值均为负，但场2存在正差值；RBA与组合增强也呈现跨场方向变化。MC面板中的差值接近零，结合前述精确数值，应理解为本轮筛选未形成可见的精度增益，而不是数值意义上每次结果都完全相等。

![图3 量子特征替换与训练增强的配对差异](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig03_paired_effects.png)

**图3 量子特征替换与训练增强的配对差异。** a：Q减C；b：Q+RBA减Q；c：Q+MC减Q；d：Q+RBA+MC减Q。纵轴均为测试渗透率相对 \(L_2\) 误差差值，负值有利于候选模型。不同符号表示同场不同重复，黑色短横线为场内差值均值，全部配对均显示；四面板共用纵轴范围。空间位置和同场重复不作为独立生成场，不作显著性推断。

筛选日志有助于解释 MC 在本轮实验中的实际干预程度，见表5。量子组通过接受条件的事件很少，即使没有接受屏蔽，候选试验仍已执行并产生计算开销。另一方面，训练全过程的接受事件不等于最终模型状态：验证选择可能恢复到筛选之前，因此需要分别报告两者。

**表5 MC 接受事件、最佳模型掩码及筛选耗时**

| 方法 | 接受事件总数 | 最佳模型采用减少特征的拟合数/9 | 平均筛选时间/s |
| --- | ---: | ---: | ---: |
| C+MC | 8 | 5 | 0.185519 |
| C+RBA+MC | 7 | 5 | 0.180087 |
| Q+MC | 1 | 1 | 2.484908 |
| Q+RBA+MC | 2 | 2 | 2.444302 |

这里的接受频率只描述当前阈值与模型共同作用下的结果，不能单独证明阈值过严、量子特征冗余不足或试验次数不足。区分这些解释需要专门改变相应因素，而不是依据当前结果事后归因。

原预算的空间预测存在明显过度平滑，为检验这一现象是否受训练时长影响，图4展示配对延长预算后的原始预测，保留全部三个场的首次重复，不按效果择例。在相同重复子集中，C的平均测试相对误差由0.360815降至0.249121，Q由0.298697降至0.232923，分别相对降低30.96%和22.02%。这些均值不是表3中全部重复的均值，也不能将延长训练的收益全部归因于量子模块。

图4中，Q对场1右下高值区及场3横向结构的恢复更接近参考场，但局部峰值及边界附近仍存在偏差。场2中Q的测试相对误差反而由0.303891升至0.311359，其上部异常的位置和形态仍不准确；场1中C的测试误差也低于Q。因此，增加预算有助于部分场的恢复，但既不保证各场同步改善，也不构成量子模型逐场领先的证据。旧预算空间图及新旧逐场对比保留在补充材料所列记录中。

![图4 延长预算诊断中的原始渗透率场、空间重建与留出位置误差](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_budget5000/fig04_spatial_maps.png)

**图4 延长预算诊断中的原始渗透率场、空间重建与留出位置误差。** 从左到右为场1、场2、场3；从上到下为原始渗透率、C重建、Q重建、C测试点绝对误差、Q测试点绝对误差。C、Q采用相同延长预算，仅展示基础组，不包含RBA或MC。各场统一取首次重复的最佳验证模型；前三行共享渗透率色标，后两行共享误差色标，未独立归一化预测图或裁剪极值。灰色位置为训练或验证点，不表示零误差。网格单元直接显示，不平滑、不做参考场补偿。本图与主消融表、图3及图5—7的训练预算不同，具体设置与完整对比见补充材料S4.4。

图5回到原预算，比较固定横向剖面并加入组合增强模型，检验主对照中平均误差最低是否对应局部恢复的普遍改善。场2右端参考渗透率上升较快，各模型均明显低估；场3左侧Q及组合模型比C更接近参考变化，但中部低值仍存在偏差。组合曲线与Q在多个区段接近，与其较小的平均增量一致；这一实例观察不构成独立的机制或泛化证据。

![图5 固定剖面的渗透率重建对比](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig05_spatial_profiles.png)

**图5 原预算下固定剖面的渗透率重建对比。** a—c分别对应三个场，与图4采用相同观测划分及验证选模原则，但使用原预算的检查点，均取预先固定的中央网格行。参考场、C、Q及Q+RBA+MC使用统一纵轴；曲线连接原始网格点，未作额外插值或平滑。剖面包含训练、验证和测试位置，用于局部形态展示，不能作为纯测试误差；每条曲线来自单个模型，不绘跨运行误差带。

原预算训练记录还显示，基础 Q 的九次拟合中有七次在预算末轮取得最佳验证状态，C 中有五次。这并不证明继续训练必然更好，但说明主比较不宜解释为充分收敛后的极限精度。补充诊断对C和Q对称延长预算，不单独增加某一组训练资源，仍按验证误差选模；局部图像相似性不作为选模标准。

完整训练轨迹见图6。验证观测误差总体随训练降低，而训练点物理残差在早期下降后出现回升或波动，说明联合目标下的数据拟合与局部方程约束并非同步改善。RBA组与无增强组的残差轨迹有所分离，但验证曲线的分离较小，不能仅由训练残差下降推断留出渗透率精度提升。这里采用中位数及运行分位带描述轨迹；表3和表6仍采用均值与样本标准差，两种统计口径不混同。

![图6 观测拟合与物理残差的训练轨迹](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig06_training_curves.png)

**图6 观测拟合与物理残差的训练轨迹。** 上行为经典组，下行为量子组；左列为参数更新后的渗透率验证损失，右列为参数更新前、训练配置点上的未加权残差RMSE，并非独立测试配置点残差。各策略纳入全部场及重复，实线或不同线型表示运行中位数，阴影为25%—75%运行分位区间，不是总体置信区间。显示全部记录轮次，不平滑；同列上下图共用纵轴范围，纵轴为对数刻度。

### 3.6 物理一致性、计算成本与适用范围

物理指标与场值指标的排序并不相同。表6中，Q+RBA 的平均独立点 \(E_r\) 为0.197464，较 Q 的0.260069降低24.07%，其幅度远大于 \(E_k\) 的相对变化。RBA 直接调整物理残差的空间分配，降低方程残差而未同步降低场值误差，与两类目标不等价这一事实相符，但不能据此断言某一种内部机制已被完全验证。

**表6 辅助压力、物理残差、下游通量与训练时间，均值±样本标准差**

| 方法 | \(E_p\) | \(E_r\) | \(E_Q\) | 训练时间/s |
| --- | ---: | ---: | ---: | ---: |
| C | 0.025162±0.005703 | 0.253345±0.088200 | 0.041240±0.026581 | 0.806676±0.039745 |
| C+RBA | 0.025652±0.006153 | 0.205412±0.069168 | 0.037199±0.024120 | 0.829655±0.040073 |
| C+MC | 0.025979±0.006217 | 0.256594±0.102368 | 0.039026±0.023869 | 0.994967±0.130162 |
| C+RBA+MC | 0.026182±0.006507 | 0.201774±0.082320 | 0.035245±0.023626 | 0.995277±0.100124 |
| Q | 0.022394±0.003941 | 0.260069±0.079575 | 0.048361±0.021607 | 6.324138±0.112660 |
| Q+RBA | 0.021980±0.004550 | 0.197464±0.067867 | 0.047102±0.019312 | 6.351136±0.089649 |
| Q+MC | 0.023439±0.004715 | 0.246591±0.090684 | 0.048335±0.021626 | 8.854528±0.176741 |
| Q+RBA+MC | 0.021230±0.003523 | 0.209573±0.055684 | 0.047460±0.018987 | 8.771515±0.171681 |

注：全部结果来自 CPU 双精度的经典计算与量子线路模拟。\(E_Q\) 是同一粗网格下 FVM 重求解的通量误差，不是网络自身的通量不平衡。

基础 Q 的压力平均误差低于 C，但其方程残差略高，下游通量平均误差也更高。四个量子配置的 \(E_Q\) 均高于相同增强条件的经典配置。场值相对 \(L_2\) 误差对位置作范数汇总，而通量还取决于渗透率分布与压力梯度的耦合，两者没有必然一致的排序。当前结果直接支持这种不一致性，但尚未通过局部敏感性分析定位造成通量偏差的具体区域。

基础 Q 平均训练时间约为 C 的7.84倍，Q+MC 又比 Q 增加40.01%。筛选耗时已计入表6总训练时间，不应再次相加。特征掩码没有改变实际线路执行，较少的有效读出也未转化为模拟加速。这里的成本仅属于小规模精确模拟；真实设备还涉及采样和高阶导数估计[27-28]，不能用该时间比外推量子硬件资源。

图7将独立配置点残差与训练耗时并列展示，并保留全部运行点。残差分布存在较多重叠，量子组耗时则整体高于经典组，加入MC后进一步增加。因此，当前量子特征收益应限定在场值重建指标及给定预算下，不能由较低的某一误差推出计算效率优势。

![图7 独立物理残差与训练成本](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_gstools/fig07_physics_cost.png)

**图7 独立物理残差与训练成本。** a：选中模型在独立配置点上的未加权Darcy残差RMSE；b：包含循环内验证和MC开销的训练时间。每行显示对应方法的全部运行，圆点为逐次结果，黑色菱形及横线为均值±样本标准差；纵向微小偏移仅用于分辨重叠点。时间轴为对数刻度。标准差是运行波动描述，不作为独立场总体置信区间；时间属于CPU量子模拟，不是量子设备计时。

本实验还存在三个主要证据边界。其一，仅使用同一协方差设置下的少量无噪声场和单一边界工况，且观测划分与初始化同时变化；误差波动尚不能分解为不同来源，也不足以支持广泛统计推广。其二，未加入纯数据网络消融，不能从当前比较单独证明 Darcy 约束优于仅拟合稀疏渗透率；没有等参数、等时间与更多经典表示对照，也未区分编码、纠缠和线路参数化各自贡献。其三，验证集重复用于筛选和选模，固定预算与参考网格检查均有范围限制，未涉及有限 shots、设备噪声或现场数据。对于更大线路的训练困难，本文同样未提供诊断或解决证据[25]。

## 4 结论

1. 建立了仅由稀疏渗透率观测监督、由辅助压力参与 Darcy 约束的双网络 PINN，在渗透率表示的固定位置比较经典与参数化量子特征，并对两类模型对称施加 RBA 和 MC。该设计明确区分了观测信息、物理先验和训练后参考评价。
2. 在当前 GSTools 合成场与固定预算下，基础量子模型的平均测试渗透率相对 \(L_2\) 误差由0.337094降至0.306759，降低9.00%。收益在场级和运行级均不完全一致，支持的是本设置中的平均重建改善。
3. RBA 使量子模型的平均独立点方程残差降低24.07%，但渗透率误差仅降低0.27%；MC 单独加入量子模型未改善主指标。组合模型虽取得最低平均场值误差，其增量仍小，不能概括为各增强模块均有效。
4. 场值精度、物理响应和计算成本需分别评价。基础量子模拟耗时约为经典基线的7.84倍，下游通量误差未全面改善；当前证据不构成普遍量子优势、未知场泛化或硬件加速的证明。

## 数据与代码说明

本文主对照来自2026年10月7日完成的 GSTools、仅渗透率监督实验，重写时复用并只读核验既有档案。后续为图4补充六次基础模型的配对延长预算训练，与主对照独立归档，不覆盖原结果。完整配置、逐场统计与筛选记录见[补充材料](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/materical/SUPPLEMENTARY_MATERIAL_V3.md)，主实验汇总见[实验结果](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/experiments/quantum_pinn_v3_gstools_konly_20261007/summary.json)，新旧预算的逐场对比见[重建诊断报告](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/figures/manuscript_v3_budget5000/RECONSTRUCTION_BUDGET_REPORT.md)。文献阅读与本轮复用范围见[文献记录](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/materical/LITERATURE_READING_V3.md)，文稿检查见[核验记录](/Users/yuanzhi.liu/Desktop/quantum-pinn-net/v3/paper/materical/MANUSCRIPT_V3_QA.md)。图1、图2对应当前协议，其制作来源与检查范围见补充材料；未沿用旧压力监督实验的图像或数值。

## 参考文献

[1] RAISSI M, PERDIKARIS P, KARNIADAKIS G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707. DOI: 10.1016/j.jcp.2018.10.045.

[2] SIRIGNANO J, SPILIOPOULOS K. DGM: A deep learning algorithm for solving partial differential equations[J]. Journal of Computational Physics, 2018, 375: 1339-1364. DOI: 10.1016/j.jcp.2018.08.029.

[3] E W, YU B. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems[J]. Communications in Mathematics and Statistics, 2018, 6(1): 1-12. DOI: 10.1007/s40304-018-0127-z.

[4] CUOMO S, DI COLA V S, GIAMPAOLO F, et al. Scientific machine learning through physics-informed neural networks: Where we are and what's next[J]. Journal of Scientific Computing, 2022, 92: 88. DOI: 10.1007/s10915-022-01939-z.

[5] LU L, MENG X, MAO Z, et al. DeepXDE: A deep learning library for solving differential equations[J]. SIAM Review, 2021, 63(1): 208-228. DOI: 10.1137/19M1274067.

[6] WANG S, TENG Y, PERDIKARIS P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J]. SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081. DOI: 10.1137/20M1318043.

[7] WANG S, YU X, PERDIKARIS P. When and why PINNs fail to train: A neural tangent kernel perspective[J]. Journal of Computational Physics, 2022, 449: 110768. DOI: 10.1016/j.jcp.2021.110768.

[8] KRISHNAPRIYAN A S, GHOLAMI A, ZHE S, et al. Characterizing possible failure modes in physics-informed neural networks[C]//Advances in Neural Information Processing Systems. 2021, 34: 26548-26560.

[9] WANG S, SANKARAN S, WANG H, et al. An expert's guide to training physics-informed neural networks[EB/OL]. arXiv:2308.08468, 2023. DOI: 10.48550/arXiv.2308.08468.

[10] TANCIK M, SRINIVASAN P P, MILDENHALL B, et al. Fourier features let networks learn high frequency functions in low dimensional domains[C]//Advances in Neural Information Processing Systems. 2020, 33: 7537-7547.

[11] SITZMANN V, MARTEL J N P, BERGMAN A W, et al. Implicit neural representations with periodic activation functions[C]//Advances in Neural Information Processing Systems. 2020, 33: 7462-7473.

[12] WANG S, WANG H, PERDIKARIS P. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 384: 113938. DOI: 10.1016/j.cma.2021.113938.

[13] JAGTAP A D, KAWAGUCHI K, KARNIADAKIS G E. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks[J]. Journal of Computational Physics, 2020, 404: 109136. DOI: 10.1016/j.jcp.2019.109136.

[14] JAGTAP A D, KAWAGUCHI K, KARNIADAKIS G E. Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks[J]. Proceedings of the Royal Society A, 2020, 476(2239): 20200334. DOI: 10.1098/rspa.2020.0334.

[15] MCCLENNY L D, BRAGA-NETO U M. Self-adaptive physics-informed neural networks[J]. Journal of Computational Physics, 2023, 474: 111722. DOI: 10.1016/j.jcp.2022.111722.

[16] ANAGNOSTOPOULOS S J, TOSCANO J D, STERGIOPULOS N, et al. Residual-based attention in physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116805. DOI: 10.1016/j.cma.2024.116805.

[17] WU C, ZHU M, TAN Q, et al. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2023, 403: 115671. DOI: 10.1016/j.cma.2022.115671.

[18] YU J, LU L, MENG X, et al. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems[J]. Computer Methods in Applied Mechanics and Engineering, 2022, 393: 114823. DOI: 10.1016/j.cma.2022.114823.

[19] SUKUMAR N, SRIVASTAVA A. Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2022, 389: 114333. DOI: 10.1016/j.cma.2021.114333.

[20] MITARAI K, NEGORO M, KITAGAWA M, et al. Quantum circuit learning[J]. Physical Review A, 2018, 98: 032309. DOI: 10.1103/PhysRevA.98.032309.

[21] KYRIIENKO O, PAINE A E, ELFVING V E. Solving nonlinear differential equations with differentiable quantum circuits[J]. Physical Review A, 2021, 103: 052416. DOI: 10.1103/PhysRevA.103.052416.

[22] MARKIDIS S. On physics-informed neural networks for quantum computers[J]. Frontiers in Applied Mathematics and Statistics, 2022, 8: 1036711. DOI: 10.3389/fams.2022.1036711.

[23] TRAHAN C, LOVELAND M, DENT S. Quantum physics-informed neural networks[J]. Entropy, 2024, 26(8): 649. DOI: 10.3390/e26080649.

[24] SCHULD M, SWEKE R, MEYER J J. Effect of data encoding on the expressive power of variational quantum-machine-learning models[J]. Physical Review A, 2021, 103: 032430. DOI: 10.1103/PhysRevA.103.032430.

[25] MCCLEAN J R, BOIXO S, SMELYANSKIY V N, et al. Barren plateaus in quantum neural network training landscapes[J]. Nature Communications, 2018, 9: 4812. DOI: 10.1038/s41467-018-07090-4.

[26] BERGHOLM V, IZAAC J, SCHULD M, et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations[EB/OL]. arXiv:1811.04968, 2018. DOI: 10.48550/arXiv.1811.04968.

[27] SCHULD M, BERGHOLM V, GOGOLIN C, et al. Evaluating analytic gradients on quantum hardware[J]. Physical Review A, 2019, 99: 032331. DOI: 10.1103/PhysRevA.99.032331.

[28] MARI A, BROMLEY T R, KILLORAN N. Estimating the gradient and higher-order derivatives on quantum hardware[J]. Physical Review A, 2021, 103: 012405. DOI: 10.1103/PhysRevA.103.012405.

[29] LU L, JIN P, PANG G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators[J]. Nature Machine Intelligence, 2021, 3: 218-229. DOI: 10.1038/s42256-021-00302-5.

[30] LI Z, KOVACHKI N, AZIZZADENESHELI K, et al. Fourier neural operator for parametric partial differential equations[C]//International Conference on Learning Representations. 2021. arXiv:2010.08895.

[31] RAO X, OUYANG X, LIU Y. Quantum-classical physics-informed neural networks for permeability inversion in single-phase reservoir flow[J]. Advanced Quantum Technologies, 2026, 9(8): e70409. DOI: 10.1002/qute.70409.

[32] MÜLLER S, SCHÜLER L, ZECH A, et al. GSTools v1.3: a toolbox for geostatistical modelling in Python[J]. Geoscientific Model Development, 2022, 15(7): 3161-3182. DOI: 10.5194/gmd-15-3161-2022.

[33] 宋家豪, 曹文博, 张伟伟. FD-PINN: 频域物理信息神经网络[J]. 力学学报, 2023, 55(5): 1195-1205. DOI: 10.6052/0459-1879-23-169.
