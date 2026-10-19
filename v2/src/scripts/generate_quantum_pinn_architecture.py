"""Render the implemented quantum-inspired permeability network schematic."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantum-paper-mpl")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "v2/paper/figures/architecture/phase_periodic_network"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
fontManager.addfont(FONT)
mpl.rcParams.update({
    "font.family": FontProperties(fname=FONT).get_name(),
    "font.size": 7.3,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

INK = "#1E303A"
LINE = "#536773"
BLUE = "#DFECF3"
TEAL = "#DDEDE8"
OCHRE = "#F5E9D5"
GRAY = "#ECEFF0"


def draw():
    fig = plt.figure(figsize=(183 / 25.4, 117 / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 183), ylim=(0, 117))
    ax.set_axis_off()

    def text(x, y, label, size=7.3, weight="normal", color=INK, **kwargs):
        ax.text(x, y, label, fontsize=size, fontweight=weight, color=color,
                ha=kwargs.pop("ha", "center"), va=kwargs.pop("va", "center"),
                linespacing=1.28, **kwargs)

    def box(x, y, width, height, label, fill, size=7.0):
        ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                               facecolor=fill, edgecolor=LINE, linewidth=0.65))
        text(x, y, label, size=size)

    def arrow(points, dashed=False, color=LINE):
        style = "--" if dashed else "-"
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot((start[0], end[0]), (start[1], end[1]),
                    color=color, lw=0.7, ls=style)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=7, lw=0.7, linestyle=style,
                                    color=color, shrinkA=0, shrinkB=0))

    text(5, 112, "量子启发 PINN 网络模型：渗透率场内预测", size=10.1,
         weight="bold", ha="left")
    text(5, 106.5, "经典相位特征与场级弱约束；无量子线路或局部 Darcy 方程残差",
         size=7.1, color=LINE, ha="left")

    text(5, 100, "网络前向", size=7.6, weight="bold", ha="left")
    for y, label in ((94, "空间坐标"), (86, "场级条件"), (78, "场标识→嵌入")):
        box(16, y, 24, 6.8, label, GRAY, size=6.8)
        ax.plot((28, 31), (y, y), color=LINE, lw=0.7)
    ax.plot((31, 31), (78, 94), color=LINE, lw=0.7)

    box(43, 86, 20, 12, "输入\n拼接", GRAY)
    box(70, 86, 23, 12, "相位投影\n周期响应", BLUE)
    box(98, 86, 22, 12, "特征\n拼接", BLUE)
    box(126, 86, 26, 12, "门控正弦\n首层", OCHRE)
    box(153, 86, 20, 12, "SiLU 残差\n×3", TEAL)
    box(175, 86, 14, 12, "线性\n读出", TEAL, size=6.6)
    arrow([(31, 86), (33, 86)])
    for start, end in ((53, 58.5), (81.5, 87), (109, 113),
                       (139, 143), (163, 168)):
        arrow([(start, 86), (end, 86)])
    arrow([(43, 80), (43, 72.5), (98, 72.5), (98, 80)])
    text(69, 69.7, "原始输入保留", size=6.6, color=LINE)
    text(70, 96.2, "基础 / 混合相位", size=6.6, color=LINE)
    text(126, 96.2, "正弦 × 门控", size=6.6, color=LINE)
    text(153, 96.2, "恒等跳接", size=6.6, color=LINE)
    text(175, 76.8, "归一化对数渗透率", size=6.6, color=LINE, ha="right")

    ax.plot((5, 178), (65, 65), color="#BEC9CC", lw=0.65)
    text(5, 61, "训练阶段", size=7.6, weight="bold", ha="left")
    for x, label, fill in ((24, "观测拟合", BLUE),
                           (69, "场均值先验", OCHRE),
                           (114, "邻域平滑", GRAY),
                           (159, "边缘均值", GRAY)):
        box(x, 51, 38, 9, label, fill)
        ax.plot((x, x), (46.5, 42.5), color=LINE, lw=0.7)
    ax.plot((24, 159), (42.5, 42.5), color=LINE, lw=0.7)
    box(77, 32, 47, 11, "加权目标", GRAY)
    box(131, 32, 47, 11, "AdamW 更新网络参数", TEAL, size=6.9)
    arrow([(77, 42.5), (77, 37.5)])
    arrow([(100.5, 32), (107.5, 32)])
    text(91.5, 20.3, "可选对照：动态目标权重 · 训练点残差加权 · MC 神经元筛选",
         size=6.8, color=LINE)

    ax.plot((5, 178), (16, 16), color="#BEC9CC", lw=0.65)
    text(5, 12.4, "训练后", size=7.6, weight="bold", ha="left")
    box(49, 10.5, 29, 9, "原始预测", TEAL)
    box(87, 10.5, 32, 9, "训练观测残差", BLUE)
    box(127, 10.5, 38, 9, "邻域校正（可选）", OCHRE)
    box(164, 10.5, 28, 9, "校正结果", OCHRE)
    arrow([(63.5, 10.5), (67.5, 10.5)])
    arrow([(103, 10.5), (108, 10.5)])
    arrow([(146, 10.5), (150, 10.5)])
    text(119, 3.5, "原始预测与校正结果分别评价；稳态流动诊断不回传网络",
         size=6.7, color=LINE)

    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
