from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantum-pinn-v2-mpl")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/supplement_v2"
OUT = ROOT / "figures/manuscript_v2"
sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

SEEDS = [101, 202, 303]
COLORS = {"fourier": "#777777", "phase": "#0072B2", "idw": "#BA7745", "rbf": "#19856F",
          "dynamic": "#9A6EAC", "full": "#C34A59"}
DISPLAY = {"fourier": "Fourier", "phase": "Phase", "idw": "IDW", "rbf": "RBF",
           "no-mixer": "No mixer", "no-phase": "No phase features",
           "no-sine": "SiLU first layer", "no-gate": "No gate",
           "no-condition": "No condition input", "data-only": "Data only",
           "rba": "Phase + RBA", "dynamic": "Dynamic", "full": "Full"}
RECORDS = {}
COMPARISONS = [("fourier", "raw"), ("phase", "raw"), ("fourier", "corrected"),
               ("phase", "corrected"), ("idw", "raw"), ("rbf", "raw")]


def output_style(method, stage):
    name = DISPLAY[method] + (" + C" if stage == "corrected" else "")
    style = "--" if method in ("fourier", "phase") and stage == "raw" else "-"
    return name, COLORS[method], style


def plot_band(ax, x, series, label, color, style="-", positive=False):
    x, series = np.asarray(x), np.asarray(series, dtype=float)
    assert series.ndim == 2 and series.shape[1] == len(x)
    assert np.all(np.diff(x) > 0) and np.isfinite(series).all()
    mean, sd = series.mean(0), series.std(0, ddof=1)
    lower = mean - sd
    if positive:
        assert np.all(series > 0)
        lower = np.maximum(lower, np.finfo(float).tiny)
    ax.plot(x, mean, label=label, color=color, linestyle=style, lw=1.25)
    ax.fill_between(x, lower, mean + sd, color=color, alpha=0.10, linewidth=0)
    return {"label": label, "x": x.tolist(), "runs": series.tolist(),
            "mean": mean.tolist(), "sd": sd.tolist()}


def validation_series(method, scenario="random70"):
    histories = [json.loads((EXP / f"seed_{s}/{scenario}/{method}/history.json").read_text())
                 for s in SEEDS]
    return np.array([[h["validation_mse"] ** 0.5 for h in run] for run in histories])


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def record(seed, scenario, method):
    return RECORDS[(seed, scenario, method)]


def vals(scenario, method, metric="rmse", stage="raw"):
    return np.array([record(s, scenario, method)[stage][metric] for s in SEEDS])


def label_axes(axes):
    for i, ax in enumerate(np.asarray(axes).ravel()):
        ax.annotate(chr(97 + i), xy=(0, 1), xycoords="axes fraction", xytext=(-12, 7),
                    textcoords="offset points", fontweight="bold", fontsize=8,
                    ha="left", va="bottom", annotation_clip=False)


def finish(fig, axes, name, claim, roles):
    label_axes(axes)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(np.asarray(axes).ravel()),
        json_out=OUT / f"{name}.alignment.json",
        overlay_svg=OUT / f"{name}.alignment.svg", strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
    )
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    dump(OUT / f"{name}.provenance.json", {
        "claim": claim, "panel_roles": roles, "source": "experiments/supplement_v2",
        "source_table": "source_metrics.csv", "field_count": 128,
        "run_ids": SEEDS, "exclusions": [],
        "uncertainty": "Sample SD across three repeated runs unless explicitly labelled otherwise",
        "curve_policy": "Adjacent recorded epochs, grid locations or empirical ranks are joined "
                        "by straight segments; no spline smoothing or invented observations.",
    })
    plt.close(fig)


def load_and_aggregate():
    paths = list(EXP.glob("seed_*/*/*/result.json"))
    if len(paths) != 101:
        raise RuntimeError(f"Expected 101 completed runs, found {len(paths)}")
    rows = []
    for path in sorted(paths):
        r = json.loads(path.read_text())
        assert r["raw"]["rmse"] > 0 and r["training_seconds"] > 0
        RECORDS[(r["seed"], r["scenario"], r["method"])] = r
        for stage in ("raw", "corrected"):
            if stage not in r:
                continue
            scalar = {k: v for k, v in r[stage].items() if not isinstance(v, list)}
            row = {"seed": r["seed"], "scenario": r["scenario"], "method": r["method"],
                   "stage": stage, **scalar, "training_seconds": r["training_seconds"],
                   "inference_seconds": r["inference_seconds"], "parameters": r["parameters"]}
            for k in ("pressure_rmse_mean", "flux_relative_error_mean"):
                row[k] = r.get(f"flow_{stage}", {}).get(k, "")
            rows.append(row)
    with (OUT / "source_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = sorted(set((r["scenario"], r["method"], r["stage"]) for r in rows))
    aggregate = []
    for scenario, method, stage in keys:
        selected = [r for r in rows if (r["scenario"], r["method"], r["stage"]) == (scenario, method, stage)]
        item = {"scenario": scenario, "method": method, "stage": stage, "n": len(selected)}
        for key in list(rows[0])[4:]:
            numbers = [r[key] for r in selected if r[key] != ""]
            if numbers:
                item[key + "_mean"] = float(np.mean(numbers))
                item[key + "_sd"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregate.append(item)
    dump(OUT / "summary.json", aggregate)
    with (OUT / "summary.csv").open("w", newline="") as f:
        columns = list(dict.fromkeys(k for row in aggregate for k in row))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregate)
    return rows


def task_figure(truth):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6))
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.26, top=0.87, wspace=0.40)
    coordinate = np.linspace(0, 1, 32)
    im = axes[0].contourf(coordinate, coordinate, truth[42], levels=np.linspace(-1, 1, 21),
                          cmap="viridis", extend="both")
    axes[0].axhline(coordinate[16], color="white", lw=0.9)
    axes[0].axvline(coordinate[16], color="white", lw=0.9, linestyle="--")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Reference field")
    axes[1].plot(coordinate, truth[42, 16, :], color=COLORS["phase"], lw=1.4,
                 label="Horizontal section")
    axes[1].plot(coordinate, truth[42, :, 16], color=COLORS["idw"], lw=1.4,
                 linestyle="--", label="Vertical section")
    axes[1].set_xlabel("Position along section")
    axes[1].set_ylabel("Normalized log-permeability")
    axes[1].set_title("Central sections")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.73, 0.995), ncol=2, fontsize=6.5)
    for ax in axes:
        ax.set_box_aspect(1)
    cax = fig.add_axes([0.12, 0.105, 0.29, 0.024])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Normalized log-permeability")
    dump(OUT / "field_sections.json", {"field": 42, "section_index": 16, "x": coordinate.tolist(),
                                      "horizontal": truth[42, 16, :].tolist(),
                                      "vertical": truth[42, :, 16].tolist()})
    finish(fig, axes, "fig01_observations", "Spatial heterogeneity includes different variations along intersecting sections.",
           ["Reference field, contours interpolate between original grid nodes for display only",
            "Two central sections on all 32 original grid nodes; one prespecified field, no uncertainty"])


def overall():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.77, wspace=0.43)
    curves = []
    percentile = np.arange(1, 129) / 128 * 100
    for method, stage in COMPARISONS:
        values = np.sort(vals("random70", method, "per_field_rmse", stage), axis=1)
        label, color, style = output_style(method, stage)
        curves.append(plot_band(axes[0], percentile, values, label, color, style, positive=True))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Empirical field percentile (%)")
    axes[0].set_ylabel("Per-field test RMSE")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=6)
    a = np.mean(vals("random70", "fourier", "per_field_rmse"), axis=0)
    b = np.mean(vals("random70", "phase", "per_field_rmse"), axis=0)
    order = np.argsort(a - b)
    axes[1].plot(np.arange(1, 129), (a - b)[order], color=COLORS["phase"], lw=1.5)
    axes[1].axhline(0, color="#555555", lw=0.65)
    axes[1].set_xlabel("Field rank by paired difference")
    axes[1].set_ylabel("RMSE(Fourier) - RMSE(Phase)")
    axes[1].set_title("Paired network difference")
    dump(OUT / "paired_fields.json", {"fourier": a.tolist(), "phase": b.tolist(),
                                     "difference": (a - b).tolist(), "improved_count": int(np.sum(a > b))})
    dump(OUT / "overall_curves.json", curves)
    finish(fig, axes, "fig02_overall", "Phase improves the raw network comparison, while spatial baselines bound its task-level value.",
           ["All 128 field RMSEs sorted separately within each run; pointwise mean and SD of empirical order statistics",
            "All-field paired differences after averaging three runs; no sampling CI"])


def spatial(truth):
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 5.6))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.10, top=0.88, wspace=0.36, hspace=0.47)
    test = np.load(EXP / "seed_101/block/observations.npz")["test"]
    methods = [("phase", "raw"), ("phase", "corrected"), ("idw", "raw"), ("rbf", "raw")]
    x = np.arange(9, 23) / 31
    rows = []
    for row, index in enumerate([0, 42, 85]):
        assert test[index, 16, 9:23].all()
        actual = truth[index, 16, 9:23]
        axes[row, 0].plot(x, actual, color="#222222", lw=1.65, label="Reference")
        for method, stage in methods:
            pred = np.load(EXP / f"seed_101/block/{method}/{stage}.npy")[index, 16, 9:23]
            label, color, style = output_style(method, stage)
            axes[row, 0].plot(x, pred, color=color, linestyle=style, lw=1.1, label=label)
            axes[row, 1].plot(x, pred - actual, color=color, linestyle=style, lw=1.1)
            rows.append({"field": index, "method": method, "stage": stage, "x": x.tolist(),
                         "reference": actual.tolist(), "prediction": pred.tolist(),
                         "signed_error": (pred - actual).tolist()})
        axes[row, 0].set_ylabel("Field value")
        axes[row, 1].set_ylabel("Prediction - reference")
        axes[row, 0].set_title(f"Field {index}: central section", fontsize=8)
        axes[row, 1].set_title(f"Field {index}: local deviation", fontsize=8)
        axes[row, 1].axhline(0, lw=0.65, color="#999999")
        for ax in axes[row]:
            ax.set_xlim(x[0], x[-1])
            ax.set_xlabel("x")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.99), fontsize=7)
    dump(OUT / "spatial_profiles.json", {"scenario": "block", "run": 101, "row": 16,
                                        "columns": list(range(9, 23)), "series": rows})
    finish(fig, axes, "fig03_spatial", "Matched missing-region sections reveal local shape and signed reconstruction deviations.",
           ["Prespecified fields 0, 42, 85, row 16; all 14 displayed locations are test points",
            "Reference versus four prediction profiles and signed errors; one run, no uncertainty estimate"])


def ablation():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.5))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.73, wspace=0.35)
    groups = [["phase", "no-mixer", "no-phase", "no-sine", "no-gate", "no-condition"],
              ["phase", "data-only", "rba", "dynamic", "full"]]
    palette = {**COLORS, "no-mixer": "#777777", "no-phase": "#19856F",
               "no-sine": "#C34A59", "no-gate": "#AD8523", "no-condition": "#9A6EAC",
               "data-only": "#777777", "rba": "#19856F"}
    styles = ["-", "--", "-.", ":", "--", "-."]
    curves = []
    for panel, (ax, group) in enumerate(zip(axes, groups)):
        for method, style in zip(group, styles):
            series = validation_series(method)
            item = plot_band(ax, np.arange(1, 41), series, DISPLAY[method], palette[method], style)
            curves.append({"panel": panel, "method": method, **item})
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation RMSE")
        ax.set_xlim(1, 40)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.3)
    limits = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    dump(OUT / "ablation_curves.json", curves)
    finish(fig, axes, "fig04_ablation", "Controlled interventions separate periodic representation from adaptive optimization.",
           ["Validation trajectories for representation controls with a shared backbone",
            "Validation trajectories for objective and adaptive controls; pointwise three-run SD"])


def dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.78, wspace=0.4)
    for method in ["fourier", "phase", "dynamic", "full"]:
        histories = [json.loads((EXP / f"seed_{s}/random70/{method}/history.json").read_text()) for s in SEEDS]
        series = np.array([[h["validation_mse"] ** 0.5 for h in run] for run in histories])
        x = np.arange(1, 41)
        axes[0].plot(x, series.mean(0), color=COLORS[method], label=DISPLAY[method], lw=1.2)
        axes[0].fill_between(x, series.mean(0) - series.std(0, ddof=1),
                             series.mean(0) + series.std(0, ddof=1), color=COLORS[method], alpha=0.13)
    histories = [json.loads((EXP / f"seed_{s}/random70/dynamic/history.json").read_text()) for s in SEEDS]
    for key, label, color, style in zip(["data", "physics", "smoothness", "boundary"],
                                       ["Data", "Mean prior", "Smoothness", "Edge mean"],
                                       ["#0072B2", "#BA7745", "#19856F", "#9A6EAC"],
                                       ["-", "--", "-.", ":"]):
        a = np.array([[h["weights_used"][key] for h in run] for run in histories])
        axes[1].plot(x, a.mean(0), label=label, color=color, linestyle=style, lw=1.25)
        axes[1].fill_between(x, np.maximum(a.mean(0) - a.std(0, ddof=1), 1e-9),
                             a.mean(0) + a.std(0, ddof=1), color=color, alpha=0.1)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.5)
    axes[0].set_ylabel("Validation RMSE")
    axes[1].set_ylabel("Weight used in gradient updates")
    axes[1].set_yscale("log")
    finish(fig, axes, "fig05_training", "Inverse-loss balancing shifts optimization away from observations.",
           ["Matched validation trajectories with run SD", "Weights actually used, not post-epoch updates; run SD"])


def sensitivity(truth):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.78, wspace=0.4)
    noise_curves = []
    for scenario, label, color, style in [
            ("random70", "No noise", COLORS["phase"], "-"),
            ("noise05", "Noise 5%", COLORS["idw"], "--"),
            ("noise10", "Noise 10%", COLORS["full"], "-.")]:
        noise_curves.append(plot_band(axes[0], np.arange(1, 41),
                                     validation_series("phase", scenario), label, color, style))
    distance_source = []
    position_source = []
    for method in ["fourier", "phase", "idw", "rbf"]:
        stage = "corrected" if method in ("fourier", "phase") else "raw"
        display = DISPLAY[method] + (" + C" if stage == "corrected" else "")
        positions = []
        bins = [0, 2, 4, 6, 8]
        for seed in SEEDS:
            observation = np.load(EXP / f"seed_{seed}/block/observations.npz")
            pred = np.load(EXP / f"seed_{seed}/block/{method}/{stage}.npy")
            assert observation["test"][:, 9:23, 9:23].all()
            gap_error = pred[:, 9:23, 9:23].astype(float) - truth[:, 9:23, 9:23].astype(float)
            positions.append(np.sqrt(np.mean(gap_error ** 2, axis=(0, 1))))
            coords = np.indices((32, 32)).reshape(2, -1).T
            errors, distances = [], []
            for i in range(len(truth)):
                tr, test = observation["train"][i].ravel(), observation["test"][i].ravel()
                d, _ = cKDTree(coords[tr]).query(coords[test], k=1)
                distances.extend(d)
                errors.extend((pred[i].ravel()[test] - truth[i].ravel()[test]) ** 2)
            d, e = np.array(distances), np.array(errors)
            assert np.all((d > bins[0]) & (d <= bins[-1]))
            y = [float(np.sqrt(e[(d > lo) & (d <= hi)].mean()))
                 for lo, hi in zip(bins[:-1], bins[1:])]
            distance_source.append({"seed": seed, "method": method, "bins": bins, "rmse": y,
                                    "counts": [int(np.sum((d > lo) & (d <= hi))) for lo, hi in zip(bins[:-1], bins[1:])]})
        item = plot_band(axes[1], np.arange(9, 23) / 31, positions,
                         display, COLORS[method], positive=True)
        position_source.append({"method": method, "stage": stage, **item})
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation RMSE")
    axes[1].set_xlabel("x inside the central gap")
    axes[1].set_ylabel("Column-wise test RMSE")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.5)
    dump(OUT / "block_distance.json", distance_source)
    dump(OUT / "sensitivity_curves.json", {"noise_validation": noise_curves,
                                          "gap_columns": position_source})
    finish(fig, axes, "fig06_sensitivity", "Noisy fitting and contiguous missing regions have distinct reconstruction effects.",
           ["Phase validation trajectories evaluated against the corresponding noisy validation targets",
            "RMSE by gap column over all 128 fields and 14 missing rows; pointwise three-run SD"])


def engineering(truth):
    from run_supplement_v2 import flow_response

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.9))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.84, wspace=0.36, hspace=0.43)
    reference = [flow_response(field) for field in truth]
    pressure0 = np.array([r[0] for r in reference])
    inlet0 = []
    for field, pressure in zip(truth, pressure0):
        k = np.exp(field.astype(float) - float(field.mean()))
        inlet0.append(2 * k[:, 0] * field.shape[1] * (1 - pressure[:, 0]))
    inlet0 = np.array(inlet0)
    curves, audit = [], []
    coordinate = (np.arange(32) + 0.5) / 32
    for col, scenario in enumerate(["random70", "block"]):
        for method, stage in COMPARISONS:
            pressure_runs, inlet_runs = [], []
            for seed in SEEDS:
                pred = np.load(EXP / f"seed_{seed}/{scenario}/{method}/{stage}.npy")
                solved = [flow_response(field) for field in pred]
                pressure = np.array([r[0] for r in solved])
                inlet = []
                for field, p, response in zip(pred, pressure, solved):
                    k = np.exp(field.astype(float) - float(field.mean()))
                    local_flux = 2 * k[:, 0] * field.shape[1] * (1 - p[:, 0])
                    np.testing.assert_allclose(local_flux.mean(), response[1], atol=1e-12)
                    inlet.append(local_flux)
                pressure_error = pressure - pressure0
                pressure_runs.append(np.sqrt(np.mean(pressure_error ** 2, axis=(0, 1))))
                inlet_runs.append(np.mean(abs(np.array(inlet) - inlet0), axis=0))
                field_rmse = np.sqrt(np.mean(pressure_error ** 2, axis=(1, 2)))
                expected = record(seed, scenario, method)[f"flow_{stage}"]["pressure_per_field"]
                np.testing.assert_allclose(field_rmse, expected, rtol=1e-11, atol=1e-13)
                q0, q1 = np.array([r[1] for r in reference]), np.array([r[1] for r in solved])
                flux_error = abs(q1 - q0) / abs(q0)
                expected_flux = record(seed, scenario, method)[f"flow_{stage}"]["flux_per_field"]
                np.testing.assert_allclose(flux_error, expected_flux, rtol=1e-11, atol=1e-13)
                audit.append({"seed": seed, "scenario": scenario, "method": method, "stage": stage,
                              "max_pressure_metric_difference": float(np.max(abs(field_rmse - expected))),
                              "max_flux_metric_difference": float(np.max(abs(flux_error - expected_flux)))})
            label, color, style = output_style(method, stage)
            for row, values in enumerate([pressure_runs, inlet_runs]):
                item = plot_band(axes[row, col], coordinate, values, label, color, style, positive=True)
                curves.append({"scenario": scenario, "method": method, "stage": stage,
                               "metric": "pressure_column_rmse" if row == 0 else "inlet_local_flux_mae",
                               **item})
        axes[0, col].set_title("Random missing" if col == 0 else "Central gap")
        axes[0, col].set_xlabel("x")
        axes[1, col].set_xlabel("y along inlet")
        axes[0, col].set_ylabel("Pressure RMSE by x")
        axes[1, col].set_ylabel("Local inlet-flux MAE")
    for ax in axes.ravel():
        ax.set_yscale("log")
    for row in range(2):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=6,
               bbox_to_anchor=(0.53, 0.99), fontsize=6.5)
    dump(OUT / "flow_profiles.json", {"series": curves, "metric_checks": audit,
                                     "inlet_definition": "Flux density, not integrated total flux"})
    finish(fig, axes, "fig07_flow", "Pressure and inlet-flow deviations have different spatial distributions.",
           ["Pressure RMSE at each x over all fields and y cells, random and block missing",
            "Mean absolute local inlet-flux-density error at each y over all fields, not total-flux error",
            "Profiles use all 128 fields and original 32 cell centres; pointwise three-run SD"])


def cost_budget():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.76, wspace=0.38)
    curves = []
    for ax, method in zip(axes, ["fourier", "phase"]):
        for scenario, linestyle, color in [("random70", "--", "#777777"),
                                            ("budget120", "-", COLORS[method])]:
            hist = json.loads((EXP / f"seed_101/{scenario}/{method}/history.json").read_text())
            x, y = [r["epoch"] for r in hist], [r["validation_mse"] ** 0.5 for r in hist]
            ax.plot(x, y, linestyle=linestyle, color=color,
                    label=f"{len(hist)} epochs", lw=1.35)
            curves.append({"method": method, "scenario": scenario, "x": x, "validation_rmse": y})
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation RMSE")
        ax.set_title(DISPLAY[method])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=7)
    limits = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    dump(OUT / "budget_curves.json", curves)
    finish(fig, axes, "fig08_cost_budget", "Short-budget architecture differences must be separated from eventual convergence and cost.",
           ["Fourier validation trajectories under two training budgets",
            "Phase validation trajectories under the same two budgets; one prespecified run, schedule length also changes"])


def configure():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.65, "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 9), default=list(range(1, 9)))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    load_and_aggregate()
    truth = np.load(ROOT / "experiments/rba_qpinn_submission_v1/seed_42/processed/fields.npy")
    builders = [lambda: task_figure(truth), overall, lambda: spatial(truth), ablation, dynamics,
                lambda: sensitivity(truth), lambda: engineering(truth), cost_budget]
    for number in args.figures:
        builders[number - 1]()
        print(f"Figure {number} written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
