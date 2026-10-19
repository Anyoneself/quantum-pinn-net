"""Render Chinese result figures directly from the frozen GSTools V3 archive."""

import csv
import hashlib
import json
import os
from pathlib import Path
import string
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "v3/experiments/quantum_pinn_v3_gstools_konly_20261007"
OUT = ROOT / "v3/paper/figures/manuscript_v3_gstools"
AUDIT_SCRIPTS = Path(os.environ.get(
    "NATURE_FIGURE_SCRIPTS",
    str(Path.home() / ".codex/skills/nature-figure/scripts"),
))
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


METHODS = [f"{kind}_{strategy}" for kind in ("classical", "quantum")
           for strategy in ("plain", "rba", "mc", "rba_mc")]
LABELS = dict(zip(METHODS, (
    "C", "C+RBA", "C+MC", "C+RBA+MC", "Q", "Q+RBA", "Q+MC", "Q+RBA+MC",
)))
COLORS = {"plain": "#286C9B", "rba": "#258875",
          "mc": "#B56825", "rba_mc": "#97577D"}
STYLES = {"plain": "-", "rba": "--", "mc": ":", "rba_mc": "-."}
FIGURE_RECORDS = []


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_font():
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        raise RuntimeError("A Chinese font is required.")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [name, "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.65, "lines.linewidth": 1.25,
        "legend.frameon": False, "savefig.facecolor": "white",
    })


def panel(ax, letter, title=None):
    ax.annotate(letter, (0, 1), xycoords="axes fraction",
                xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom")
    if title:
        ax.set_title(title, loc="center", pad=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.65)
    ax.set_axisbelow(True)


def save(fig, name, axes, metadata):
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, axes=list(np.asarray(axes).flat),
        json_out=OUT / f"{name}.alignment.json",
        overlay_svg=OUT / f"{name}.alignment.svg",
        require_panel_labels=True, strict=True,
    )
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.preview.png", dpi=300)
    FIGURE_RECORDS.append({
        "name": name, "size_inches": fig.get_size_inches().tolist(),
        "panels": len(np.asarray(axes).flat),
        "alignment_verdict": alignment.get("verdict"),
        **metadata,
    })
    plt.close(fig)


def load_archive():
    manifest = read(ARCHIVE / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["data_config"]["field_generator"] == "gstools"
    assert manifest["data_config"]["p_train"] == 0
    for group, prefix in (("artifact_sha256", ""), ("source_sha256", "source")):
        for relative, expected in manifest[group].items():
            path = (ARCHIVE / prefix / relative).resolve()
            assert path.is_relative_to(ARCHIVE.resolve())
            assert digest(path) == expected, path
    runs, histories = {}, {}
    for field in manifest["field_seeds"]:
        for seed in manifest["seeds"]:
            for method in METHODS:
                folder = ARCHIVE / f"field_{field}/seed_{seed}/{method}"
                key = (field, seed, method)
                runs[key] = read(folder / "result.json")
                histories[key] = read(folder / "history.json")
                assert [r["epoch"] for r in histories[key]] == list(
                    range(runs[key]["epochs"] + 1))
    rows = [{
        "field": f, "seed": s, "method": m,
        **{metric: row[metric] for metric in (
            "k_relative_l2", "log_k_rmse", "pressure_rmse", "pde_residual_rmse",
            "downstream_flux_relative_error", "training_seconds", "screening_seconds",
            "best_epoch", "active_features_at_best",
        )},
    } for (f, s, m), row in runs.items()]
    write_csv("result_run_metrics.csv", rows)
    summary = read(ARCHIVE / "summary.json")
    for method in METHODS:
        for metric in ("k_relative_l2", "pressure_rmse", "pde_residual_rmse",
                       "downstream_flux_relative_error", "training_seconds"):
            values = np.array([r[metric] for r in rows if r["method"] == method])
            assert np.isclose(values.mean(), summary["methods"][method][metric]["mean"])
            assert np.isclose(values.std(ddof=1), summary["methods"][method][metric]["sd"])
    return manifest, runs, histories


def paired_effects(manifest, runs):
    pairs = [
        ("quantum_plain", "classical_plain", "基础量子替换：Q - C"),
        ("quantum_rba", "quantum_plain", "残差加权增量：Q+RBA - Q"),
        ("quantum_mc", "quantum_plain", "特征筛选增量：Q+MC - Q"),
        ("quantum_rba_mc", "quantum_plain", "组合增强增量：Q+RBA+MC - Q"),
    ]
    rows = []
    for candidate, baseline, _ in pairs:
        for field in manifest["field_seeds"]:
            for seed in manifest["seeds"]:
                rows.append({
                    "candidate": candidate, "baseline": baseline,
                    "field": field, "seed": seed,
                    "delta_k_relative_l2": (
                        runs[field, seed, candidate]["k_relative_l2"]
                        - runs[field, seed, baseline]["k_relative_l2"]),
                })
    write_csv("paired_effects_source.csv", rows)
    values = np.array([r["delta_k_relative_l2"] for r in rows])
    padding = max(np.ptp(values) * 0.12, 0.005)
    limits = (min(0, values.min()) - padding, max(0, values.max()) + padding)
    fig, axes = plt.subplots(2, 2, figsize=(7.204724, 5.0))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.85,
                        wspace=0.32, hspace=0.56)
    for ax, letter, (candidate, baseline, title), color in zip(
            axes.flat, "abcd", pairs, COLORS.values()):
        panel(ax, letter, title)
        ax.axhline(0, color="#888888", lw=0.8, linestyle="--", zorder=1)
        for fi, field in enumerate(manifest["field_seeds"]):
            differences = [r["delta_k_relative_l2"] for r in rows
                           if r["candidate"] == candidate and r["field"] == field]
            for si, delta in enumerate(differences):
                ax.scatter(fi + (si - 1) * 0.10, delta, s=25, color=color,
                           marker=("o", "s", "^")[si], zorder=3)
            ax.plot([fi - 0.24, fi + 0.24], [np.mean(differences)] * 2,
                    color="#222222", lw=1.8, zorder=4)
        ax.set(xlim=(-0.5, 2.5), ylim=limits, xticks=range(3),
               xticklabels=["场1", "场2", "场3"], ylabel="测试相对 L2 误差差值")
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    fig.legend(handles=[
        Line2D([], [], ls="", marker=m, color="#555555", label=f"重复{i+1}")
        for i, m in enumerate(("o", "s", "^"))
    ] + [Line2D([], [], color="#222222", lw=1.8, label="场内差值均值")],
        loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4)
    save(fig, "fig03_paired_effects", axes, {
        "claim": "Average improvement is not a uniform paired improvement.",
        "source": "paired_effects_source.csv", "rows": len(rows),
        "uncertainty": "all paired observations and within-field means; no CI",
        "axis_rule": "same y range in all four panels; negative favors candidate",
    })


def spatial_data(manifest):
    fields = manifest["field_seeds"]
    seed = manifest["seeds"][0]
    data, rows = {}, []
    for field in fields:
        with np.load(ARCHIVE / f"field_{field}/reference.npz") as ref:
            xy, truth = ref["xy"], np.exp(ref["log_k"])
        with np.load(ARCHIVE / f"field_{field}/seed_{seed}/observations.npz") as obs:
            splits = np.full(len(truth), "test", dtype="<U10")
            splits[obs["k_train_indices"]] = "train"
            splits[obs["k_validation_indices"]] = "validation"
        predictions = {}
        for method in ("classical_plain", "quantum_plain", "quantum_rba_mc"):
            with np.load(ARCHIVE / f"field_{field}/seed_{seed}/{method}/predictions.npz") as p:
                predictions[method] = np.exp(p["log_k"])
        assert np.isfinite(truth).all() and (truth > 0).all()
        assert all(np.isfinite(p).all() and (p > 0).all() for p in predictions.values())
        data[field] = {"xy": xy, "truth": truth, "splits": splits, **predictions}
        for index in range(len(truth)):
            rows.append({
                "field": field, "seed": seed, "index": index,
                "x": xy[index, 0], "y": xy[index, 1], "split": splits[index],
                "reference_k": truth[index],
                **{m: values[index] for m, values in predictions.items()},
            })
    write_csv("spatial_results_source.csv", rows)
    return data, rows


def spatial_maps(manifest, data):
    n = manifest["data_config"]["grid"]
    fields = manifest["field_seeds"]
    displayed = np.concatenate([data[f][key] for f in fields
                                for key in ("truth", "classical_plain", "quantum_plain")])
    k_min, k_max = displayed.min(), displayed.max()
    errors = [np.abs(data[f][m] - data[f]["truth"])[data[f]["splits"] == "test"]
              for f in fields for m in ("classical_plain", "quantum_plain")]
    error_max = max(float(e.max()) for e in errors)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#DADADA")
    fig, axes = plt.subplots(5, 3, figsize=(7.204724, 10.6))
    fig.subplots_adjust(left=0.15, right=0.87, bottom=0.055, top=0.94,
                        wspace=0.16, hspace=0.36)
    row_labels = ("原始渗透率", "经典 C 重建", "量子 Q 重建",
                  "C 测试绝对误差", "Q 测试绝对误差")
    for ci, field in enumerate(fields):
        item = data[field]
        expected_xy = np.stack(np.meshgrid(
            (np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n), axis=-1).reshape(-1, 2)
        np.testing.assert_allclose(item["xy"], expected_xy)
        arrays = [item["truth"], item["classical_plain"], item["quantum_plain"]]
        arrays += [np.ma.masked_where(
            item["splits"] != "test", np.abs(item[m] - item["truth"]))
            for m in ("classical_plain", "quantum_plain")]
        for ri, (ax, values) in enumerate(zip(axes[:, ci], arrays)):
            panel(ax, string.ascii_lowercase[ri * 3 + ci])
            image = ax.imshow(values.reshape(n, n), extent=(0, 1, 0, 1),
                              origin="lower", interpolation="nearest",
                              cmap="viridis" if ri < 3 else cmap,
                              vmin=k_min if ri < 3 else 0,
                              vmax=k_max if ri < 3 else error_max)
            ax.set(xticks=[0, 1], yticks=[0, 1])
            if ri == 0:
                ax.set_title(f"场{ci+1}", fontsize=9, pad=18)
            if ci == 0:
                ax.set_ylabel(f"{row_labels[ri]}   y", labelpad=5)
            else:
                ax.set_yticklabels([])
            if ri == 4:
                ax.set_xlabel("x", labelpad=1)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if ri == 0:
                field_image = image
            if ri == 4:
                error_image = image
    cax1 = fig.add_axes((0.90, 0.49, 0.015, 0.36))
    cax2 = fig.add_axes((0.90, 0.12, 0.015, 0.20))
    cb1 = fig.colorbar(field_image, cax=cax1)
    cb2 = fig.colorbar(error_image, cax=cax2)
    cb1.set_label("渗透率 k（统一色标）", labelpad=4)
    cb2.set_label("绝对误差", labelpad=4)
    save(fig, "fig04_spatial_maps", axes, {
        "claim": "Reconstruction differences are spatially nonuniform across all three fields.",
        "source": "spatial_results_source.csv",
        "selection": "all fields, first manifest seed, no best-case selection",
        "uncertainty": "single selected checkpoint per field/method; not an aggregate",
        "field_scale": [float(k_min), float(k_max)], "error_scale": [0, error_max],
        "layout": "columns: all three fields; rows: truth, C, Q, C error, Q error",
        "processing": "exp(log_k); nearest cell rendering, no clipping/interpolation",
        "error_mask": "training and validation positions masked in gray",
    })


def spatial_profiles(manifest, data, rows):
    n = manifest["data_config"]["grid"]
    indices = np.arange(n * (n // 2), n * (n // 2 + 1))
    chosen = [r for r in rows if r["index"] in indices]
    write_csv("profile_results_source.csv", chosen)
    fig, axes = plt.subplots(3, 1, figsize=(7.204724, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.87, hspace=0.63)
    displayed = np.array([[r[key] for key in (
        "reference_k", "classical_plain", "quantum_plain", "quantum_rba_mc")]
        for r in chosen])
    padding = np.ptp(displayed) * 0.08
    for ax, letter, field in zip(axes, "abc", manifest["field_seeds"]):
        item = data[field]
        x, y = item["xy"][indices].T
        assert np.all(np.diff(x) > 0) and np.allclose(y, y[0])
        panel(ax, letter, f"场{manifest['field_seeds'].index(field)+1}")
        ax.plot(x, item["truth"][indices], color="#222222", lw=1.6, label="参考场")
        ax.plot(x, item["classical_plain"][indices], color="#888888",
                ls="--", label="C")
        ax.plot(x, item["quantum_plain"][indices], color=COLORS["plain"], label="Q")
        ax.plot(x, item["quantum_rba_mc"][indices], color=COLORS["rba_mc"],
                ls="-.", label="Q+RBA+MC")
        ax.set(xlim=(0, 1), ylim=(max(0, displayed.min()-padding), displayed.max()+padding),
               ylabel="渗透率 k")
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    axes[-1].set_xlabel("x")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=4)
    save(fig, "fig05_spatial_profiles", axes, {
        "claim": "An aggregate gain does not ensure every local peak or trough is recovered.",
        "source": "profile_results_source.csv", "rows": len(chosen),
        "selection": "all fields, first manifest seed, fixed middle grid row",
        "y": float(data[manifest["field_seeds"][0]]["xy"][indices[0], 1]),
        "uncertainty": "single selected checkpoint, no uncertainty band",
        "scope": "all split types on section, not a held-out-only metric",
    })


def training_curves(manifest, histories):
    rows, aggregated = [], []
    fig, axes = plt.subplots(2, 2, figsize=(7.204724, 5.2))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.85,
                        wspace=0.34, hspace=0.60)
    for ri, kind in enumerate(("classical", "quantum")):
        prefix = "经典" if ri == 0 else "量子"
        for ci, metric in enumerate(("validation", "physics_unweighted_mse_pre_step")):
            ax = axes[ri, ci]
            panel(ax, "abcd"[ri * 2 + ci],
                  prefix + ("：验证观测误差" if ci == 0 else "：训练点物理残差"))
            for strategy in COLORS:
                method = f"{kind}_{strategy}"
                histories_for_method = [(f, s, histories[f, s, method])
                                        for f in manifest["field_seeds"]
                                        for s in manifest["seeds"]]
                all_values = []
                for field, seed, history in histories_for_method:
                    selected = [entry for entry in history if metric in entry]
                    epoch = np.array([entry["epoch"] for entry in selected])
                    values = np.array([entry[metric] for entry in selected])
                    assert np.isfinite(values).all() and (values >= 0).all()
                    if ci == 1:
                        values = np.sqrt(values)
                    all_values.append(values)
                    rows.extend({"field": field, "seed": seed, "method": method,
                                 "metric": "validation_loss" if ci == 0 else "training_residual_rmse",
                                 "epoch": int(e), "value": float(v)}
                                for e, v in zip(epoch, values))
                array = np.asarray(all_values)
                lower, median, upper = np.quantile(array, [0.25, 0.5, 0.75], axis=0)
                assert (lower > 0).all()
                ax.plot(epoch, median, color=COLORS[strategy], ls=STYLES[strategy],
                        label={"plain": "无增强", "rba": "RBA",
                               "mc": "MC", "rba_mc": "RBA+MC"}[strategy])
                ax.fill_between(epoch, lower, upper, color=COLORS[strategy], alpha=0.10, lw=0)
                aggregated.extend({
                    "method": method, "metric": metric, "epoch": int(e),
                    "q25": float(lo), "median": float(md), "q75": float(hi),
                } for e, lo, md, hi in zip(epoch, lower, median, upper))
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(
                [0.5, 1, 2, 3, 5] if ci == 0 else [0.05, 0.1, 0.2, 0.5, 1]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set(xlim=(0, manifest["train_config"]["epochs"]),
                   xlabel="训练轮次",
                   ylabel="验证损失（对数轴）" if ci == 0 else "残差 RMSE（对数轴）")
            ax.grid(which="major", color="#EEEEEE", lw=0.5)
    # Keep comparable classical/quantum panels on identical axes.
    for ci in range(2):
        limits = [ax.get_ylim() for ax in axes[:, ci]]
        for ax in axes[:, ci]:
            ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=4)
    write_csv("training_results_source.csv", rows)
    write_csv("training_results_summary.csv", aggregated)
    save(fig, "fig06_training_curves", axes, {
        "claim": "Observation fitting and physical-residual optimization follow distinct trajectories.",
        "source": "training_results_source.csv", "summary": "training_results_summary.csv",
        "rows": len(rows), "runs": len(histories),
        "uncertainty": "median and 25th-75th run percentiles, descriptive only",
        "scope": "validation after update; training PDE before update, not held-out PDE",
        "processing": "all logged epochs, no smoothing; positive log axes",
    })


def physics_cost(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 4.1))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.80, wspace=0.53)
    rows = []
    for ax, metric, letter, title in zip(
            axes, ("pde_residual_rmse", "training_seconds"), "ab",
            ("独立配置点物理残差", "包含筛选开销的训练成本")):
        panel(ax, letter, title)
        for mi, method in enumerate(METHODS):
            values = np.array([row[metric] for key, row in runs.items() if key[2] == method])
            assert np.all(values > 0)
            color = "#737373" if method.startswith("classical") else "#286C9B"
            ax.scatter(values, mi + np.linspace(-0.17, 0.17, len(values)),
                       color=color, alpha=0.55, s=13, zorder=2)
            ax.errorbar(values.mean(), mi, xerr=values.std(ddof=1), fmt="D",
                        color="#222222", markersize=3.2, capsize=2.5, lw=0.9, zorder=3)
            rows.append({"method": method, "metric": metric, "mean": float(values.mean()),
                         "sample_sd": float(values.std(ddof=1)), "runs": len(values)})
        ax.set(yticks=range(len(METHODS)), yticklabels=[LABELS[m] for m in METHODS],
               ylim=(7.5, -0.5))
        ax.axhline(3.5, color="#CCCCCC", lw=0.65, ls="--")
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
    axes[0].set(xlabel="未加权残差 RMSE", xlim=(0, None))
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("训练时间 / s（对数轴）")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#888888", label="经典组：逐次运行"),
        Line2D([], [], marker="o", ls="", color="#286C9B", label="量子组：逐次运行"),
        Line2D([], [], marker="D", color="#222222", label="均值 ± 样本标准差"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
    write_csv("physics_cost_summary.csv", rows)
    save(fig, "fig07_physics_cost", axes, {
        "claim": "Physical-residual gains and simulation cost must be evaluated jointly.",
        "source": "result_run_metrics.csv", "summary": "physics_cost_summary.csv",
        "runs": len(runs), "uncertainty": "all runs plus mean and sample SD; no CI",
        "scope": "selected-checkpoint independent residual; full training-loop cost",
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    manifest, runs, histories = load_archive()
    paired_effects(manifest, runs)
    data, spatial_rows = spatial_data(manifest)
    spatial_maps(manifest, data)
    spatial_profiles(manifest, data, spatial_rows)
    training_curves(manifest, histories)
    physics_cost(runs)
    sources = [
        "result_run_metrics.csv", "paired_effects_source.csv", "spatial_results_source.csv",
        "profile_results_source.csv", "training_results_source.csv",
        "training_results_summary.csv", "physics_cost_summary.csv",
    ]
    outputs = list(sources)
    for figure in FIGURE_RECORDS:
        outputs += [f"{figure['name']}.{suffix}" for suffix in (
            "png", "preview.png", "pdf", "svg", "alignment.json", "alignment.svg")]
    report = {
        "archive": str(ARCHIVE.relative_to(ROOT)), "new_training": False,
        "runs": len(runs), "independent_fields": len(manifest["field_seeds"]),
        "manifest_sha256": digest(ARCHIVE / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "backend": "Python/Matplotlib", "figures": FIGURE_RECORDS,
        "artifact_sha256": {name: digest(OUT / name) for name in outputs},
        "note": "Descriptive figures from saved runs; no significance claim or retraining.",
    }
    write_json(OUT / "result_figures.json", report)
    print(json.dumps({"figures": len(FIGURE_RECORDS), "runs": len(runs),
                      "source_files": len(sources), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
