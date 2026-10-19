"""Rebuild manuscript evidence and figures from the immutable V3 archive."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "v3/experiments/quantum_pinn_v3_initial_20261007"
OUT = ROOT / "v3/paper/figures/manuscript_v3"
AUDIT_SCRIPTS = Path(os.environ.get(
    "NATURE_FIGURE_SCRIPTS",
    str(Path.home() / ".codex/skills/nature-figure/scripts"),
))
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

LABELS = {
    "classical_plain": "C", "classical_rba": "C + RBA",
    "classical_mc": "C + MC", "classical_rba_mc": "C + RBA + MC",
    "quantum_plain": "Q", "quantum_rba": "Q + RBA",
    "quantum_mc": "Q + MC", "quantum_rba_mc": "Q + RBA + MC",
}
COLORS = {"classical_plain": "#646464", "quantum_plain": "#0072B2",
          "quantum_mc": "#009E73", "quantum_rba_mc": "#CC79A7"}


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def csv_rows(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(OUT / f"{name}.alignment.json"),
        overlay_svg=str(OUT / f"{name}.alignment.svg"),
        require_panel_labels=len(fig.axes) > 1, strict=True,
    )
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.preview.png", dpi=300)
    plt.close(fig)


def panel(ax, letter, title):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-29, 13), textcoords="offset points",
                fontsize=9, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", fontsize=9, pad=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def box(ax, center, text, width=0.2, height=0.12, color="#F1F1F1"):
    x, y = center
    ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                           facecolor=color, edgecolor="#666666", lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def arrow(ax, start, end, color="#555555"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=9, lw=0.8, color=color))


def diagrams():
    from draw_v3_architecture import draw_architecture

    draw_architecture(save)

    fig, ax = plt.subplots(figsize=(7.204724, 3.937008))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.96)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, (0.19, 0.91), "Training observations\nFixed physics points", 0.34, 0.13)
    box(ax, (0.70, 0.91), "Validation observations", 0.40, 0.13, "#E4F1EA")
    box(ax, (0.19, 0.64), "PINN residuals\nOptional RBA weights\nJoint Adam update", 0.34, 0.18)
    arrow(ax, (0.19, 0.845), (0.19, 0.73))
    box(ax, (0.70, 0.64), "Optional MC: paired feature ablation\nMean + 2 SE threshold\nAccept only if validation does not rise",
        0.54, 0.18, "#DEEEF6")
    arrow(ax, (0.36, 0.64), (0.43, 0.64))
    arrow(ax, (0.70, 0.845), (0.70, 0.73))
    box(ax, (0.70, 0.36), "Validation checkpoint selection\nSave parameters AND feature mask",
        0.54, 0.15, "#E4F1EA")
    arrow(ax, (0.70, 0.55), (0.70, 0.435))
    ax.text(0.18, 0.425, "Continue training\nwithin fixed budget", ha="center", fontsize=8)
    arrow(ax, (0.43, 0.36), (0.32, 0.36))
    arrow(ax, (0.09, 0.40), (0.09, 0.55))
    box(ax, (0.29, 0.10), "Held-out labels\nEvaluation only", 0.32, 0.13)
    box(ax, (0.75, 0.10), "Selected-model evaluation\nField / pressure / PDE / cost", 0.44, 0.13)
    arrow(ax, (0.70, 0.285), (0.70, 0.165))
    arrow(ax, (0.45, 0.10), (0.53, 0.10))
    save(fig, "fig02_workflow")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    })
    manifest = read(ARCHIVE / "manifest.json")
    # Check the frozen sources and outputs before deriving any manuscript data.
    for group in ("source_sha256", "artifact_sha256"):
        for name, expected in manifest[group].items():
            path = ARCHIVE / ("source" if group == "source_sha256" else "") / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    summary = read(ARCHIVE / "summary.json")
    methods = manifest["methods"]
    runs = [read(p) for p in sorted(ARCHIVE.glob("field_*/seed_*/*/result.json"))]
    assert len(runs) == 72
    csv_rows("run_metrics.csv", [
        {k: v for k, v in run.items() if isinstance(v, (int, float, str, bool))}
        for run in runs
    ])
    lookup = {(r["field_seed"], r["seed"], r["method"]): r for r in runs}
    history = {}
    records = []
    for field in manifest["field_seeds"]:
        for seed in manifest["seeds"]:
            for method in methods:
                path = ARCHIVE / f"field_{field}/seed_{seed}/{method}"
                hist = read(path / "history.json")
                history[field, seed, method] = hist
                records.extend({"field": field, "seed": seed, "method": method,
                                "epoch": row["epoch"], "validation": row["validation"],
                                "active_features": row["active_features"],
                                "physics_unweighted_mse_pre_step":
                                    row.get("physics_unweighted_mse_pre_step", "")}
                               for row in hist)
    csv_rows("training_source.csv", records)
    comparisons = {}
    for name, pair in summary["paired_comparisons"].items():
        a, b = name.split("_vs_")
        ma = summary["methods"][a]["k_relative_l2"]["mean"]
        mb = summary["methods"][b]["k_relative_l2"]["mean"]
        differences = [r["candidate_minus_baseline"] for r in pair["runs"]]
        comparisons[name] = {
            "relative_reduction_percent": 100 * (mb-ma) / mb,
            "better_equal_worse": [sum(d < -1e-12 for d in differences),
                                   sum(abs(d) <= 1e-12 for d in differences),
                                   sum(d > 1e-12 for d in differences)],
        }
    screening = {}
    for method in methods:
        events = [e for p in ARCHIVE.glob(f"field_*/seed_*/{method}/screening.json")
                  for e in read(p)]
        selected = [r for r in runs if r["method"] == method]
        screening[method] = {
            "accepted_events": sum(e["accepted"] for e in events),
            "logged_events": len(events),
            "selected_mask_reduced_runs": sum(r["active_features_at_best"] < 4 for r in selected),
            "best_epoch_300_runs": sum(r["best_epoch"] == 300 for r in selected),
            "mean_screening_seconds": float(np.mean([r["screening_seconds"] for r in selected])),
            "active_features_at_best": [r["active_features_at_best"] for r in selected],
        }
    evidence = {"archive": str(ARCHIVE.relative_to(ROOT)), "runs": len(runs),
                "summary_sha256": hashlib.sha256((ARCHIVE / "summary.json").read_bytes()).hexdigest(),
                "comparisons": comparisons, "screening": screening,
                "methods": summary["methods"],
                "statistical_scope": "3 fields x 3 repeated fits; descriptive SD, no significance test",
                "new_training": False}
    write_json(OUT / "evidence.json", evidence)
    diagrams()

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 3.385827))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.77, wspace=0.40)
    paired_rows = []
    for ax, strategy, letter in zip(axes, ("plain", "mc"), "ab"):
        panel(ax, letter, "Quantum block effect" if strategy == "plain" else "Quantum block effect with MC")
        for fi, field in enumerate(manifest["field_seeds"]):
            diffs = []
            for si, seed in enumerate(manifest["seeds"]):
                q = lookup[field, seed, f"quantum_{strategy}"]["k_relative_l2"]
                c = lookup[field, seed, f"classical_{strategy}"]["k_relative_l2"]
                delta = q-c
                diffs.append(delta)
                ax.scatter(fi+(si-1)*0.09, delta, s=20, color="#0072B2", marker=("o", "s", "^")[si])
                paired_rows.append({"field": field, "seed": seed, "strategy": strategy,
                                    "q_minus_c": delta})
            ax.plot([fi-0.19, fi+0.19], [np.mean(diffs)]*2, color="#222222", lw=1.5)
        ax.axhline(0, color="#777777", lw=0.7, linestyle="--")
        ax.set(xticks=range(3), xticklabels=["Field 1", "Field 2", "Field 3"],
               ylim=(-0.014, 0.006), ylabel="Relative L2 error: Q minus C")
    fig.text(0.5, 0.94, "Negative values favor Q; symbols are repeated fits; bars are within-field means",
             ha="center", fontsize=8)
    csv_rows("paired_source.csv", paired_rows)
    save(fig, "fig03_paired_effect")

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 3.464567))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.76, wspace=0.40)
    for ax, letter, title in zip(axes, "ab", ("Observation validation loss", "Training PDE residual")):
        panel(ax, letter, title)
    for method, color in COLORS.items():
        hs = [history[field, seed, method] for field in manifest["field_seeds"]
              for seed in manifest["seeds"]]
        for ai, key in enumerate(("validation", "physics_unweighted_mse_pre_step")):
            vals = np.array([[row[key] for row in h if key in row] for h in hs])
            if ai == 1:
                vals = np.sqrt(vals)
            mean, sd = vals.mean(0), vals.std(0, ddof=1)
            epochs = np.arange(len(mean)) + ai
            axes[ai].plot(epochs, mean, color=color, label=LABELS[method])
            axes[ai].fill_between(epochs, mean-sd, mean+sd, color=color, alpha=0.10, lw=0)
    axes[0].set(xlabel="Epoch", ylabel="Validation loss")
    axes[1].set(xlabel="Epoch", ylabel="Unweighted residual RMSE")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 300)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    save(fig, "fig04_training")

    fig, axes = plt.subplots(3, 1, figsize=(7.204724, 6.496063))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.88, hspace=0.72)
    spatial = []
    for ax, letter, field in zip(axes, "abc", manifest["field_seeds"]):
        reference = np.load(ARCHIVE / f"field_{field}/reference.npz")
        xy = reference["xy"]
        n = manifest["data_config"]["grid"]
        indices = np.arange(n*(n//2), n*(n//2+1))
        x = xy[indices, 0]
        assert np.all(np.diff(x) > 0)
        true = np.exp(reference["log_k"][indices])
        panel(ax, letter, f"Field {manifest['field_seeds'].index(field)+1}")
        ax.plot(x, true, color="#222222", lw=1.5, label="Reference")
        row_data = {"reference": true}
        for method in ("classical_plain", "quantum_plain", "quantum_mc"):
            arr = np.load(ARCHIVE / f"field_{field}/seed_101/{method}/predictions.npz")
            values = np.exp(arr["log_k"][indices])
            row_data[method] = values
            ax.plot(x, values, color=COLORS[method], label=LABELS[method],
                    linestyle="--" if method == "classical_plain" else "-")
        obs = np.load(ARCHIVE / f"field_{field}/seed_101/observations.npz")
        for j, index in enumerate(indices):
            split = next(s for s in ("train", "validation", "test")
                         if index in obs[f"k_{s}_indices"])
            spatial.append({"field": field, "seed": 101, "index": int(index),
                            "x": float(x[j]), "y": float(xy[index, 1]), "split": split,
                            **{k: float(v[j]) for k, v in row_data.items()}})
        ax.set(xlim=(0, 1), ylabel="Permeability k")
    all_values = np.array([[row[k] for k in
                           ("reference", "classical_plain", "quantum_plain", "quantum_mc")]
                          for row in spatial])
    pad = 0.08 * np.ptp(all_values)
    for ax in axes:
        ax.set_ylim(float(all_values.min()-pad), float(all_values.max()+pad))
    axes[-1].set_xlabel("x (fixed y = 0.520833)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    csv_rows("spatial_source.csv", spatial)
    save(fig, "fig05_spatial")
    print(json.dumps({"runs": len(runs), "comparisons": comparisons, "screening": screening},
                     indent=2))


if __name__ == "__main__":
    main()
