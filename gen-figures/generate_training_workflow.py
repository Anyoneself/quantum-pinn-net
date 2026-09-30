"""Draw the implemented training, model selection and evaluation workflow."""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/paper-revision-mpl")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/architecture/training_workflow"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
fontManager.addfont(FONT)
mpl.rcParams.update({
    "font.family": FontProperties(fname=FONT).get_name(),
    "font.size": 7.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})
sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment


def draw():
    # Single schematic: training labels, validation labels and test references
    # have separate destinations. No numerical data are drawn.
    fig = plt.figure(figsize=(7.2047244, 5.4724409), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 183), ylim=(0, 139))
    ax.set_axis_off()
    boxes = {}
    colors = {"train": "#E8F1F8", "val": "#E8F2EE", "test": "#F2F2F2", "output": "#FAF0DD"}

    def box(key, x, y, label, kind, width=48, height=13):
        boxes[key] = (x, y, width, height)
        ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                               facecolor=colors[kind], edgecolor="#52616A", linewidth=0.65))
        ax.text(x, y, label, ha="center", va="center", linespacing=1.5, color="#182B35")

    def anchor(key, side):
        x, y, w, h = boxes[key]
        return {"top": (x, y + h / 2), "bottom": (x, y - h / 2),
                "left": (x - w / 2, y), "right": (x + w / 2, y)}[side]

    def arrow(points, dashed=False):
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#52616A", lw=0.7,
                    ls="--" if dashed else "-")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=7, linewidth=0.7, color="#52616A",
                                    linestyle="--" if dashed else "-", shrinkA=0, shrinkB=1))

    box("data", 91, 128, "场级条件、空间位置与观测场值", "test", width=102, height=11)
    box("train", 33, 105, "训练观测", "train")
    box("val", 91, 105, "验证观测", "val")
    box("test", 151, 105, "测试参考", "test")
    ax.text(61, 116, "互斥划分", ha="center", va="center", fontsize=7, color="#52616A")
    ax.plot([91, 91], [122.5, 120], color="#52616A", lw=0.7)
    ax.plot([33, 151], [120, 120], color="#52616A", lw=0.7)
    for key in ("train", "val", "test"):
        x = boxes[key][0]
        arrow([(x, 120), anchor(key, "top")])

    box("prepare", 33, 84, "条件标准化\n训练观测标定均值先验", "train")
    box("loss", 33, 63, "网络前向计算\n观测拟合与辅助目标", "train")
    box("update", 33, 42, "反向传播\nAdamW 参数更新", "train")
    box("validate", 91, 42, "逐轮计算验证误差\n保存最优模型状态", "val")
    box("select", 91, 24, "训练预算结束\n恢复验证最优模型", "val")
    box("infer", 91, 6.8, "位置查询与训练残差校正", "output", height=10)
    box("output", 151, 6.8, "原始预测与校正场", "output", height=10)
    box("evaluate", 151, 42, "场值误差评价\n压力与通量响应比较", "test")

    for first, second in (("train", "prepare"), ("prepare", "loss"), ("loss", "update"),
                          ("val", "validate"), ("validate", "select"), ("select", "infer"),
                          ("test", "evaluate")):
        arrow([anchor(first, "bottom"), anchor(second, "top")])
    arrow([anchor("update", "right"), anchor("validate", "left")])
    arrow([anchor("infer", "right"), anchor("output", "left")])
    arrow([anchor("output", "top"), anchor("evaluate", "bottom")])
    arrow([anchor("update", "left"), (3, 42), (3, 63), anchor("loss", "left")])
    ax.text(33, 22, "训练循环", ha="center", color="#52616A", fontsize=7)
    ax.text(164, 74, "仅用于最终评价", ha="center", fontsize=7, color="#52616A",
            rotation=90, rotation_mode="anchor")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(OUT) + ".alignment.json",
                                       overlay_svg=str(OUT) + ".alignment.svg", strict=True)
    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
