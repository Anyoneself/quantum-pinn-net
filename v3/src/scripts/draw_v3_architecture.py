"""Chinese, implementation-grounded architecture schematic for manuscript V3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath


INK = "#243438"
MUTED = "#637174"
TEAL = "#24776F"
BLUE = "#246D9A"
OCHRE = "#9A701D"
NEUTRAL = "#F3F5F5"


def chinese_font():
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    raise RuntimeError("A Chinese font is required to render the V3 architecture.")


def draw_architecture(save):
    """Render only figure 1; reuse the manuscript's export and alignment checks."""
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": [chinese_font(), "DejaVu Sans"],
        "font.size": 7.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(7.204724, 6.8))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
        ax.set(xlim=(0, 18), ylim=(0, 17))
        ax.axis("off")

        def text(x, y, label, size=7.5, color=INK, ha="center", weight="normal"):
            return ax.text(x, y, label, fontsize=size, color=color, ha=ha,
                           va="center", fontweight=weight, linespacing=1.45)

        def frame(x, y, w, h, color=MUTED, fill="white", dashed=False):
            patch = FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.075",
                facecolor=fill, edgecolor=color, linewidth=0.8,
                linestyle=(0, (3, 2)) if dashed else "solid",
            )
            ax.add_patch(patch)
            return patch

        def block(x, y, w, h, title, detail="", color=MUTED, fill=NEUTRAL,
                  detail_size=7.2):
            frame(x, y, w, h, color, fill)
            if detail:
                text(x+w/2, y+h*0.70, title, color=color, weight="bold")
                text(x+w/2, y+h*0.30, detail, size=detail_size)
            else:
                text(x+w/2, y+h/2, title, color=color, weight="bold")

        def route(points, color=MUTED, dashed=False, arrow=True):
            path = DrawingPath(points, [DrawingPath.MOVETO] +
                               [DrawingPath.LINETO]*(len(points)-1))
            ax.add_patch(FancyArrowPatch(
                path=path, arrowstyle="-|>" if arrow else "-",
                mutation_scale=8, lw=0.85, color=color,
                linestyle=(0, (3, 2)) if dashed else "solid",
                capstyle="round", joinstyle="round",
            ))

        def section(letter, label, y, detail=""):
            text(0.35, y, letter, size=10, color=TEAL, ha="left", weight="bold")
            text(0.85, y, label, size=9.2, ha="left", weight="bold")
            if detail:
                text(17.65, y, detail, size=6.9, color=MUTED, ha="right")
            ax.plot([0.35, 17.65], [y-0.32, y-0.32], color="#D8E1E0", lw=0.7)

        section("a", "双网络场表示", 16.65)
        text(2.55, 15.99, "渗透率网络", color=TEAL, ha="left", weight="bold")
        block(0.35, 13.0, 1.5, 1.3, "空间坐标", r"$(x,y)$")
        frame(2.55, 14.4, 2.6, 1.15, TEAL, "#EFF7F5")
        text(3.85, 15.31, "编码与角度映射", color=TEAL, weight="bold")
        text(3.85, 14.97, r"$h=E(2\boldsymbol{x}-1)$", size=7.2)
        text(3.85, 14.62, r"$a=\pi\tanh(h)$", size=7.2)
        block(5.55, 14.4, 2.8, 1.15, "可替换特征模块 F",
              "经典 / 量子，二选一", BLUE, "#EAF3F9", 6.8)
        block(8.75, 14.4, 2.15, 1.15, "特征掩码",
              r"$m\odot F(a)$", OCHRE, "#FCF6E9")
        block(11.30, 14.4, 2.20, 1.15, "经典读出器 D",
              "输出对数渗透率", TEAL, "#EFF7F5", 6.7)
        block(13.9, 14.4, 3.75, 1.15, "渗透率预测",
              r"$\hat{\ell}=D[m\odot F(a)],\ \hat{k}=e^{\hat{\ell}}$",
              TEAL, "#EFF7F5", 7.2)
        route([(1.85, 13.65), (2.15, 13.65), (2.15, 14.975), (2.52, 14.975)])
        for start, end in ((5.15, 5.52), (8.35, 8.72), (10.9, 11.27), (13.5, 13.87)):
            route([(start, 14.975), (end, 14.975)])
        text(9.825, 16.02, "MC：可选筛选", size=7, color=OCHRE)
        route([(9.825, 15.83), (9.825, 15.58)], color=OCHRE, dashed=True)

        text(2.55, 13.60, "压力网络：始终采用经典结构", ha="left", weight="bold")
        block(2.55, 11.95, 4.4, 1.2, "经典 Tanh 全连接网络",
              r"$N_p(2\boldsymbol{x}-1)$")
        block(7.8, 11.95, 5.7, 1.2, "左右压力边界的硬约束",
              r"$\hat{p}=1-x+x(1-x)N_p$")
        block(13.9, 11.95, 3.75, 1.2, "压力预测", r"$\hat{p}(x,y)$")
        route([(2.15, 13.65), (2.15, 12.55), (2.52, 12.55)])
        route([(6.95, 12.55), (7.77, 12.55)])
        route([(13.5, 12.55), (13.87, 12.55)])
        text(9, 11.48, "对照组对应层的结构与初始化一致；各模型独立训练，不绑定参数。",
             size=7, color=MUTED)

        section("b", "特征模块的两种实现", 10.83, "仅替换 F，不做双分支融合")
        text(2.90, 10.10, "经典特征块", color=MUTED, weight="bold")
        text(11.86, 10.10, "参数化量子特征块", color=BLUE, weight="bold")
        block(0.75, 8.78, 2.0, 0.86, "全连接层")
        block(3.15, 8.78, 2.0, 0.86, "Tanh 激活")
        route([(2.75, 9.21), (3.12, 9.21)])
        text(5.91, 9.21, "或", size=9, color=BLUE, weight="bold")
        text(2.95, 8.32, "按特征层数重复", size=6.9, color=MUTED)
        quantum_steps = (
            (6.70, "角度编码", r"$R_Y(a)$"),
            (9.45, "可训练旋转", r"$\mathrm{Rot}(\theta)$"),
            (12.20, "纠缠操作", "CNOT 链与闭环"),
            (14.95, "测量期望", r"$\langle Z_j\rangle$"),
        )
        for x, title, detail in quantum_steps:
            block(x, 8.63, 2.25, 1.13, title, detail, BLUE, "#EAF3F9", 7.2)
        for x, _, _ in quantum_steps[:-1]:
            route([(x+2.25, 9.195), (x+2.72, 9.195)], BLUE)
        # Only Rot and CNOT repeat: input encoding and measurement lie outside.
        route([(9.45, 8.48), (9.45, 8.27), (14.45, 8.27), (14.45, 8.48)],
              BLUE, arrow=False)
        text(11.95, 7.99, "旋转与纠缠按线路层数重复", size=6.8, color=BLUE)

        section("c", "物理约束与联合训练", 7.37,
                "a 中的预测量分别进入观测项与物理项")
        block(0.35, 5.61, 3.85, 1.03, "对数渗透率观测",
              r"$(\hat{\ell},\ell_{\mathrm{obs}})$", TEAL, "#EFF7F5")
        block(4.83, 5.61, 3.85, 1.03, "压力观测",
              r"$(\hat{p},p_{\mathrm{obs}})$")
        block(9.31, 5.61, 8.34, 1.03, "自动微分：由渗透率与压力计算通量",
              r"$\hat{\boldsymbol{q}}=-\hat{k}\nabla\hat{p},"
              r"\qquad r=\nabla\cdot\hat{\boldsymbol{q}}$", BLUE, "#EAF3F9", 8)
        block(0.35, 3.63, 3.85, 1.40, "渗透率观测损失",
              r"$\mathcal{L}_{\ell}$", TEAL, "#EFF7F5", 9)
        block(4.83, 3.63, 3.85, 1.40, "压力观测损失",
              r"$\mathcal{L}_{p}$", MUTED, NEUTRAL, 9)
        frame(9.31, 3.63, 3.85, 1.40, BLUE, "#EAF3F9")
        text(11.235, 4.72, "方程残差损失", color=BLUE, weight="bold")
        text(11.235, 4.30, r"$\mathcal{L}_{r}\quad (r\to 0)$", size=8)
        text(11.235, 3.88, "RBA：可选残差加权", size=6.7, color=OCHRE)
        block(13.80, 3.63, 3.85, 1.40, "上下无流边界损失",
              r"$\mathcal{L}_{b}\quad (\hat{q}_y\to 0)$", BLUE, "#EAF3F9", 8)
        centers = (2.275, 6.755, 11.235, 15.725)
        for x in centers:
            route([(x, 5.61), (x, 5.06)])
        # The junction collects losses, never feeds observation losses into physics.
        for x in centers:
            route([(x, 3.63), (x, 3.04)], arrow=False)
        route([(centers[0], 3.04), (centers[-1], 3.04)], arrow=False)
        route([(9, 3.04), (9, 2.72)])
        block(3.1, 1.39, 11.8, 1.30, "联合损失：反向传播更新两个网络",
              r"$\mathcal{L}=\lambda_{\ell}\mathcal{L}_{\ell}"
              r"+\lambda_p\mathcal{L}_p+\lambda_r\mathcal{L}_r"
              r"+\lambda_b\mathcal{L}_b$", TEAL, "#EFF7F5", 9)
        text(9, 0.94, "量子门参数参与同一损失的反向传播；RBA 与 MC 均为经典训练辅助。",
             size=7, color=MUTED)
        text(9, 0.46, "MC：训练子批次配对消融 → 阈值判断 → 验证接受 → 仅更新特征掩码。",
             size=7, color=MUTED)
        save(fig, "fig01_architecture")


if __name__ == "__main__":
    from analyze_manuscript_v3 import OUT, save

    OUT.mkdir(parents=True, exist_ok=True)
    draw_architecture(save)
