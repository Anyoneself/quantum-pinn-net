"""从冻结数据绘制单面板诊断图，不挑场、不平滑、不隐藏退化结果。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"classical_plain": "#59636D", "quantum_plain": "#008B8B"}
LABELS = {"classical_plain": "Classical", "quantum_plain": "Quantum"}


def read_csv(path):
    """标准CSV解析保留原始行，不增加绘图专用数据筛选。"""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def field_values(rows, key):
    """同场先平均重复，再显示独立场之间的均值和样本标准差。"""
    return np.array([np.mean([float(r[key]) for r in rows if r["field_seed"] == field])
                     for field in sorted({r["field_seed"] for r in rows})])


def plot(output, audit_tools):
    """单图对应单个诊断问题；所有数值直接来自analysis目录的CSV。"""
    sys.path.insert(0, str(audit_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from audit_figure_collisions import audit_pdf as audit_collisions
    from audit_pdf_text import audit_pdf as audit_text
    analysis = output / "analysis"
    destination = analysis / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8})

    def save(fig, name):
        # 先测量最终渲染布局；单面板将记录不适用多面板对齐。
        fig.tight_layout(pad=1.2)
        require_matplotlib_panel_alignment(
            fig, json_out=destination / f"{name}.alignment.json",
            overlay_svg=destination / f"{name}.alignment.svg", strict=True)
        fig.savefig(destination / f"{name}.png", dpi=300)
        fig.savefig(destination / f"{name}.pdf")
        fig.savefig(destination / f"{name}.svg")
        plt.close(fig)
        # 每次导出都检查最终PDF的文字和几何，不用源码检查代替渲染检查。
        pdf = destination / f"{name}.pdf"
        collision = audit_collisions(pdf)
        text = audit_text(pdf.read_bytes())
        (destination / f"{name}.collision-audit.json").write_text(json.dumps(collision, indent=2) + "\n")
        (destination / f"{name}.text-audit.json").write_text(json.dumps(text, indent=2) + "\n")
        if collision["summary"]["fail"] or not text["auditable"] or text["below_minimum_count"]:
            raise RuntimeError(f"rendered figure QA failed: {name}")

    # 主要终点：展示每个独立场的配对变化，不用边际柱状图掩盖场异质性。
    fields = read_csv(analysis / "field_paired.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for row in fields:
        values = [float(row[f"{method}_k_relative_l2"]) for method in COLORS]
        ax.plot([0, 1], values, color="#A7ADB2", linewidth=1, zorder=1)
        for i, (method, value) in enumerate(zip(COLORS, values)):
            ax.scatter(i, value, color=COLORS[method], s=24, zorder=2)
    ax.set_xticks([0, 1], ["Classical", "Quantum"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Held-out permeability relative L2")
    ax.set_title("Paired field means")
    save(fig, "paired_reconstruction")

    # 训练轨迹按同一固定轮次比较；误差棒为独立场均值的样本标准差。
    checkpoints = read_csv(analysis / "checkpoint_metrics.csv")
    epochs = sorted({int(r["epoch"]) for r in checkpoints if r["is_scheduled"] == "True"})
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method in COLORS:
        means, spreads = [], []
        for epoch in epochs:
            rows = [r for r in checkpoints if r["method"] == method and int(r["epoch"]) == epoch]
            values = field_values(rows, "k_relative_l2")
            means.append(values.mean())
            spreads.append(values.std(ddof=1) if len(values) > 1 else 0)
        ax.errorbar(epochs, means, yerr=spreads, marker="o", capsize=3,
                    color=COLORS[method], label=LABELS[method], linewidth=1.3)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Held-out permeability relative L2")
    ax.set_title("Fixed-epoch diagnostics: mean and field SD")
    ax.legend()
    save(fig, "training_diagnostics")

    # 归一化谱排除整体核幅度差异；半透明线为各场的种子平均谱。
    spectra = [r for r in read_csv(analysis / "eigenspectra.csv")
               if r["is_best"] == "True" and r["kernel"] == "output"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method in COLORS:
        for i, field in enumerate(sorted({r["field_seed"] for r in spectra})):
            rows = [r for r in spectra if r["method"] == method and r["field_seed"] == field]
            ranks = sorted({int(r["rank"]) for r in rows})
            values = [np.mean([float(r["trace_normalized"]) for r in rows if int(r["rank"]) == rank])
                      for rank in ranks]
            # 非正谱值是浮点舍入尾部；不绘于对数轴，完整值仍保存在CSV。
            positive = np.asarray(values) > 0
            ax.semilogy(np.asarray(ranks)[positive], np.asarray(values)[positive],
                        color=COLORS[method], alpha=0.55, linewidth=1,
                        label=LABELS[method] if i == 0 else None)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue / kernel trace")
    ax.set_title("Output NTK at validation-selected checkpoints")
    ax.legend()
    save(fig, "normalized_output_spectrum")

    # 核变化与误差并列观察：这是描述性关联，不拟合暗示因果的回归线。
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method in COLORS:
        ax.scatter([float(r[f"{method}_output_shape_drift"]) for r in fields],
                   [float(r[f"{method}_k_relative_l2"]) for r in fields],
                   s=32, color=COLORS[method], label=LABELS[method],
                   marker="o" if method.startswith("classical") else "^")
    ax.set_xlabel("Trace-normalized output NTK shape drift")
    ax.set_ylabel("Held-out permeability relative L2")
    ax.set_title("Kernel drift is not a quality score")
    ax.legend()
    save(fig, "shape_drift_vs_error")
    # 绘图版本单独冻结，允许修复排版而不改写已完成训练的源码快照或原始数据。
    shutil.copyfile(Path(__file__), destination / "plot_source.py")
    provenance = {"backend": "matplotlib", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "data_sha256": {name: hashlib.sha256((analysis / name).read_bytes()).hexdigest()
                                  for name in ("field_paired.csv", "checkpoint_metrics.csv", "eigenspectra.csv")},
                  "excluded_rows": "only nonpositive roundoff spectrum entries omitted from logarithmic display",
                  "unit": "independent field, repeats averaged within field",
                  "uncertainty": "sample SD across field means for fixed-epoch curves; other plots show all fields",
                  "width_mm": 132.08, "target": "project diagnostic report, not journal submission"}
    (destination / "render_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit-tools", type=Path, required=True,
                        help="directory containing the figure panel-alignment audit module")
    args = parser.parse_args()
    plot(args.output.resolve(), args.audit_tools.resolve())
