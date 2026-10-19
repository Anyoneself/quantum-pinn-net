# 量子启发PINN网络架构图设计记录

## 当前图件（2026-10-06）

主稿图2由[Python绘图脚本](../../../src/scripts/generate_quantum_pinn_architecture.py)重绘，输出同名[PNG](phase_periodic_network.png)、[PDF](phase_periodic_network.pdf)和[SVG](phase_periodic_network.svg)。PDF和SVG保留可编辑文字；下面的生成提示与图像生成记录仅用于追溯被替换的旧版PNG，不描述当前图件。

当前图以183 mm宽的矢量布局表现三条边界：网络前向由空间坐标、场级条件与场嵌入出发，经过经典相位特征、原始输入旁路、门控正弦首层、三层SiLU残差变换和线性读出；训练使用观测拟合、场均值先验、邻域平滑及边缘均值目标；训练后的观测残差可选地进行邻域校正，稳态流动仅用于评价。标题中的“量子PINN”沿用项目命名，不意味着量子线路、局部Darcy方程残差训练或量子优势。动态目标权重、训练点残差加权和MC筛选只在相应对照组中启用。

最终输出为183×117 mm；已目视查看PNG，并对PDF执行文字和碰撞审计：最小文字6.6 pt，无低于5 pt的文字，0项碰撞失败、0项警告。静态预检的宽度警告源于将 `183 / 25.4` 的表达式误作毫米字面量；实际PDF尺寸已独立读取确认。未额外导出TIFF，需按目标期刊要求决定是否需要。

## 旧图设计记录（2026-09-30，已替换）

## 图的用途

- 类型：网络架构示意图，不是实验结果图。
- 读者：需要理解网络信息流与训练方法的中文学术读者。
- 核心信息：坐标、场级条件和可学习场嵌入通过周期相位特征、门控正弦首层与残差主干，映射到归一化对数渗透率。
- 范围：当前基准周期网络；消融配置不是同时启用的独立分支。
- 状态：内部设计稿，尚未核验具体目标期刊对生成式插图的接收要求。

## 已核对的结构

1. 场标识通过可学习嵌入表转为场嵌入；它与空间坐标和标准化场级条件拼接。
2. 相位编码器先进行可学习线性投影，再构造基础正弦、余弦及通道混合后的周期响应。
3. 周期特征与原始输入再次拼接，原始输入存在保留路径。
4. 首层包含两条分支：正弦映射，以及线性变换、Sigmoid、正弦组成的门控；两者逐元素相乘。
5. 当前基础配置使用三个 SiLU 残差隐藏层，每层具有恒等跳接与相加操作。
6. 线性读出生成单个查询位置上的归一化对数渗透率，不是压力或流量。
7. 基础训练目标包括观测拟合、场均值先验、相邻差分平滑和边缘均值约束；采用 AdamW 更新参数。
8. 动态目标权重、残差逐点加权和 MC 神经元筛选属于可选训练机制，不属于固定预测主干。
9. 观测残差的邻域插值校正属于训练后的可选处理，不是额外的可训练输出层。

## 不应出现的内容

- 量子线路、量子硬件、注意力层或代码中不存在的模块。
- 将当前训练目标画成局部 Darcy 方程约束。
- 将下游流动诊断画成网络的压力预测分支。
- 将测试标签画成训练或残差校正输入。
- 向量展开、张量维度、随机种子、性能提升数字或伪造的场图。

## 生成提示

Create one polished Chinese-language scientific neural-network architecture
schematic on a clean white background, wide landscape composition, crisp
publication-style flat shapes, generous whitespace, dark readable text and
thin orthogonal arrows. This is a conceptual architecture drawing, not a
quantitative plot. Use muted blue for periodic encoding, restrained teal for
the residual backbone, warm ochre for gating, and neutral gray for auxiliary
training. No logos, decorative atoms, quantum circuits, brains, 3D blocks,
gradients, fabricated data, or tensor dimensions.

Title: 周期相位网络架构

The dominant upper two-thirds shows a left-to-right prediction path:
three input sources labelled 空间坐标, 场级条件, and 场标识.
Only 场标识 passes through 可学习嵌入. These inputs merge at 输入拼接.
Then show 可学习相位编码, 周期特征拼接, 门控正弦首层,
SiLU 残差块 ×3, 线性读出, and 归一化对数渗透率 in this order.
A clear bypass labelled 原始输入保留 connects 输入拼接 directly to
周期特征拼接, not to the network output.

Use small, readable internal structures within the encoding and gated-layer
groups. The phase encoder contains 相位投影 branching into 基础周期响应
and 通道混合 followed by 混合周期响应; both feed the periodic feature
concatenation. In the gated first layer, the same input branches into
正弦映射 and 线性 → Sigmoid → sin, and both terminate at a single
elementwise multiplication symbol ⊙. A residual-block thumbnail shows
线性 → SiLU and an identity skip meeting at a plus node. Never add
LayerNorm or attention to this periodic network.

To the far right, outside the main network boundary, attach a smaller
dashed optional block 残差校正（可选） fed by 原始预测 and 训练点残差,
then 校正结果. Make it visually clear that the correction is not a
trainable neural layer and that raw predictions remain available.

In a subordinate lower training strip, show four short labels:
观测拟合, 场均值先验, 邻域平滑, 边缘均值.
These join 加权目标 → AdamW 更新. A restrained dashed feedback arrow
returns to the network, labelled 训练阶段. No PDE residual is present.
Below this strip, one small optional-training annotation reads:
可选训练机制：动态目标权重 · 残差逐点加权 · MC 神经元筛选.

Keep labels in simplified Chinese exactly as given. Prioritize legibility,
clear branch connections, non-overlapping text and correctly directed arrows.
No empirical plots or numerical performance values. Use no explicit vector
or tensor shapes. The hierarchy must emphasize the neural backbone rather
than the optional mechanisms. Render as a high-resolution landscape image.

## 生成与修订记录

- 生成日期：2026-09-30。
- 工具：内置 `image_gen`；返回结果未提供具体模型版本。
- 首版生成记录：`call_e1fQWp8fPRSke4H0QltYY4DG`。
- 校对版生成记录：`call_2Uqwaf6dJ0Lmlh9D9JFzlCf6`。
- 首版为纯文本描述生成，校对版以首版为唯一图像参考；未使用外部论文图或第三方模板。
- 最终资源：[phase_periodic_network.png](phase_periodic_network.png)。

校对版修正了以下问题：

1. 将不准确的恒等连接中文标注改为“恒等跳接”。
2. 仅在残差组标题保留三次重复标记，避免重复次数产生歧义。
3. 正弦首层上分支明确为线性映射后接正弦，避免暗示两次正弦变换。
4. 门控分支保持线性、Sigmoid、正弦的顺序，与上分支逐元素相乘。
5. 将原始相位保留关系改画为直接连接混合周期响应的旁路。
6. 残差校正中注明训练点残差的邻域插值，并用相加节点连接原始预测。

可视检查确认：输入拼接、周期特征拼接、门控分支、残差跳接、线性输出、四项训练目标和可选机制的表达可读；图中没有张量维度、实验数值、量子线路或局部方程损失。

该图有意省略通道宽度、相位混合的逐元素表达、正弦频率参数、校正系数及优化超参数；具体计算以方法说明和实现为准。它表示当前基础结构，不表示各消融配置同时启用，也不支持未知场泛化或性能提升的额外主张。

交付为生成式 PNG 位图，而非可编辑矢量终稿。未执行适用于矢量 PDF 的几何碰撞审计，也未核验目标期刊的生成式图像政策；投稿前仍需作者科学复核及期刊适用性检查。
