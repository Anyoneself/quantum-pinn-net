from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantum_pinn_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/quantum_pinn_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "dataset" / "processed"
DATASET_MODULE_DIR = BASE_DIR / "dataset"
if str(DATASET_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_MODULE_DIR))

from physics_constraints import PhysicalConfig, expected_log_perm_index  # noqa: E402


METHODS = [
    ("standard-pinn", "Baseline PINN"),
    ("standard-loss", "Classical + Loss Balancer"),
    ("standard-rba", "Classical + RBA"),
    ("standard-loss-rba", "Classical + Loss + RBA"),
    ("quantum-feature", "Quantum Feature PINN"),
    ("quantum-mc-only", "Quantum + MC"),
    ("quantum-rba", "Quantum + RBA"),
    ("quantum-loss", "Quantum + Loss Balancer"),
    ("quantum-mc", "Quantum + Loss + MC"),
    ("rba-no-mc", "Quantum + Loss + RBA"),
    ("rba-qpinn", "Full RBA-QPINN"),
]


METHOD_MODULES = {
    "standard-pinn": {"Quantum": 0, "Loss": 0, "MC": 0, "RBA": 0},
    "standard-loss": {"Quantum": 0, "Loss": 1, "MC": 0, "RBA": 0},
    "standard-rba": {"Quantum": 0, "Loss": 0, "MC": 0, "RBA": 1},
    "standard-loss-rba": {"Quantum": 0, "Loss": 1, "MC": 0, "RBA": 1},
    "quantum-feature": {"Quantum": 1, "Loss": 0, "MC": 0, "RBA": 0},
    "quantum-mc-only": {"Quantum": 1, "Loss": 0, "MC": 1, "RBA": 0},
    "quantum-rba": {"Quantum": 1, "Loss": 0, "MC": 0, "RBA": 1},
    "quantum-loss": {"Quantum": 1, "Loss": 1, "MC": 0, "RBA": 0},
    "quantum-mc": {"Quantum": 1, "Loss": 1, "MC": 1, "RBA": 0},
    "rba-no-mc": {"Quantum": 1, "Loss": 1, "MC": 0, "RBA": 1},
    "rba-qpinn": {"Quantum": 1, "Loss": 1, "MC": 1, "RBA": 1},
}


SHORT_LABELS = {
    "standard-pinn": "Base",
    "standard-loss": "C+Loss",
    "standard-rba": "C+RBA",
    "standard-loss-rba": "C+Loss+RBA",
    "quantum-feature": "QF",
    "quantum-mc-only": "QF+MC",
    "quantum-rba": "QF+RBA",
    "quantum-loss": "QF+Loss",
    "quantum-mc": "QF+Loss+MC",
    "rba-no-mc": "QF+Loss+RBA",
    "rba-qpinn": "Full",
}


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | float]], dict[str, np.ndarray]]:
    fields = np.load(DATA_DIR / "fields.npy")
    conditions = np.load(DATA_DIR / "conditions.npy")
    train_mask = np.load(DATA_DIR / "train_point_mask.npy")
    test_mask = np.load(DATA_DIR / "test_point_mask.npy")
    metrics_rows = json.loads((FIGURE_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    predictions = {
        key: np.load(FIGURE_DIR / "runs" / key / "predicted_log_perm.npy")
        for key, _ in METHODS
        if (FIGURE_DIR / "runs" / key / "predicted_log_perm.npy").exists()
    }
    return fields, conditions, train_mask, test_mask, metrics_rows, predictions


def method_label(key: str) -> str:
    return dict(METHODS).get(key, key)


def key_from_label(label: str) -> str:
    reverse = {label: key for key, label in METHODS}
    return reverse[label]


def short_label_from_row(row: dict[str, str | float]) -> str:
    label = str(row["Model"])
    try:
        return SHORT_LABELS[key_from_label(label)]
    except KeyError:
        return label.replace(" ", "\n")


def best_method(metrics_rows: list[dict[str, str | float]]) -> str:
    candidates = [row for row in metrics_rows if row["Model"] != "Baseline PINN"]
    return key_from_label(str(min(candidates, key=lambda row: float(row["RMSE"]))["Model"]))


def method_order(predictions: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    return [(key, label) for key, label in METHODS if key in predictions]


def heldout_values(array: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    return array[test_mask.astype(bool)]


def per_sample_rmse(real: np.ndarray, pred: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    values = []
    for idx in range(real.shape[0]):
        mask = test_mask[idx].astype(bool)
        values.append(float(np.sqrt(np.mean((pred[idx][mask] - real[idx][mask]) ** 2))))
    return np.asarray(values, dtype=np.float32)


def select_representative_samples(real: np.ndarray, test_mask: np.ndarray, baseline: np.ndarray, best: np.ndarray, count: int = 4) -> np.ndarray:
    gains = []
    for idx in range(real.shape[0]):
        mask = test_mask[idx].astype(bool)
        gain = np.mean(np.abs(baseline[idx][mask] - real[idx][mask]) - np.abs(best[idx][mask] - real[idx][mask]))
        gains.append(float(gain + 0.02 * np.std(real[idx])))
    return np.argsort(gains)[-count:][::-1]


def save_metric_radar(metrics_rows: list[dict[str, str | float]], path: Path) -> None:
    metrics = ["Accuracy", "R2", "SSIM", "Physical Score", "Composite Score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.2, 6.0), dpi=240)
    ax = fig.add_subplot(111, polar=True)
    for row in metrics_rows:
        values = [float(row[name]) for name in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=str(row["Model"]))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0.55, 1.01)
    ax.set_title("Normalized Performance Radar", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12), frameon=False, fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_ranking(metrics_rows: list[dict[str, str | float]], path: Path) -> None:
    metrics = ["RMSE", "MAE", "R2", "SSIM", "Composite Score"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.2, 3.9), dpi=240)
    for ax, metric in zip(axes, metrics):
        rows = sorted(metrics_rows, key=lambda row: float(row[metric]), reverse=metric in {"R2", "SSIM", "Composite Score"})
        labels = [str(row["Model"]).replace(" ", "\n") for row in rows]
        values = [float(row[metric]) for row in rows]
        colors = plt.cm.viridis(np.linspace(0.18, 0.86, len(values)))
        ax.barh(np.arange(len(values)), values, color=colors)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Metric-wise Method Ranking", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_module_contribution(metrics_rows: list[dict[str, str | float]], path: Path) -> None:
    baseline = metrics_rows[0]
    labels = [str(row["Model"]) for row in metrics_rows[1:]]
    rmse_gain = [
        100.0 * (float(baseline["RMSE"]) - float(row["RMSE"])) / (float(baseline["RMSE"]) + 1e-8)
        for row in metrics_rows[1:]
    ]
    mae_gain = [
        100.0 * (float(baseline["MAE"]) - float(row["MAE"])) / (float(baseline["MAE"]) + 1e-8)
        for row in metrics_rows[1:]
    ]
    r2_gain = [100.0 * (float(row["R2"]) - float(baseline["R2"])) for row in metrics_rows[1:]]
    physics_gain = [
        100.0
        * (float(baseline["Physics Residual"]) - float(row["Physics Residual"]))
        / (float(baseline["Physics Residual"]) + 1e-8)
        for row in metrics_rows[1:]
    ]

    x = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=240)
    ax.bar(x - 1.5 * width, rmse_gain, width, label="RMSE gain", color="#457b9d")
    ax.bar(x - 0.5 * width, mae_gain, width, label="MAE gain", color="#2a9d8f")
    ax.bar(x + 0.5 * width, r2_gain, width, label="R2 gain x100", color="#f4a261")
    ax.bar(x + 1.5 * width, physics_gain, width, label="Physics residual gain", color="#e76f51")
    ax.axhline(0.0, color="#263746", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" ", "\n") for label in labels], fontsize=7)
    ax.set_ylabel("Relative improvement over baseline (%)")
    ax.set_title("Ablation Module Contribution Relative to Baseline")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_design_matrix(path_png: Path, path_csv: Path) -> None:
    rows = []
    for key, label in METHODS:
        modules = METHOD_MODULES[key]
        rows.append({"Method": label, **modules})
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Method", "Quantum", "Loss", "MC", "RBA"])
        writer.writeheader()
        writer.writerows(rows)

    matrix = np.asarray([[row[name] for name in ["Quantum", "Loss", "MC", "RBA"]] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=240)
    im = ax.imshow(matrix, cmap="GnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Quantum\nfeatures", "Dynamic\nloss", "MC neuron\nthreshold", "Residual\nadaptive"], fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["Method"] for row in rows], fontsize=7.5)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, "on" if matrix[row, col] > 0 else "-", ha="center", va="center", fontsize=8, color="#102033")
    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02, ticks=[0, 1], label="Module enabled")
    ax.set_title("Ablation Design Matrix")
    plt.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)


def save_ablation_path_lines(metrics_rows: list[dict[str, str | float]], path: Path) -> None:
    steps = [short_label_from_row(row) for row in metrics_rows]
    metrics = ["RMSE", "MAE", "Physics Residual", "Composite Score"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), dpi=240)
    for ax, metric in zip(axes.ravel(), metrics):
        values = np.asarray([float(row[metric]) for row in metrics_rows], dtype=np.float32)
        ax.plot(np.arange(len(values)), values, marker="o", linewidth=2.0, color="#355070")
        best_idx = int(np.argmin(values) if metric != "Composite Score" else np.argmax(values))
        ax.scatter([best_idx], [values[best_idx]], s=58, color="#e76f51", zorder=5)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(steps[: len(values)], rotation=20, ha="right", fontsize=7)
        ax.set_title(metric)
        ax.grid(alpha=0.24)
    fig.suptitle("Ablation Path Across Module Combinations", fontsize=12)
    fig.text(0.5, 0.01, "Markers highlight the best value for each metric; lower is better for RMSE, MAE, and physics residual.", ha="center", fontsize=8.5, color="#586a7c")
    plt.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_metric_heatmap(metrics_rows: list[dict[str, str | float]], path: Path) -> None:
    metrics = ["RMSE", "MAE", "MAPE", "R2", "SSIM", "Physics Residual", "Composite Score"]
    labels = [str(row["Model"]) for row in metrics_rows]
    raw = np.asarray([[float(row[metric]) for metric in metrics] for row in metrics_rows], dtype=np.float32)
    normalized = np.zeros_like(raw)
    lower_is_better = {"RMSE", "MAE", "MAPE", "Physics Residual"}
    for col, metric in enumerate(metrics):
        values = raw[:, col]
        span = float(values.max() - values.min()) + 1e-8
        if metric in lower_is_better:
            normalized[:, col] = (values.max() - values) / span
        else:
            normalized[:, col] = (values - values.min()) / span
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=240)
    im = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=28, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            ax.text(col, row, f"{raw[row, col]:.3f}", ha="center", va="center", fontsize=6.5, color="#102033")
    fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02, label="Normalized score")
    ax.set_title("Ablation Metric Heatmap")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_sample_rmse_heatmap(
    real: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    ordered = method_order(predictions)
    rmse = np.stack([per_sample_rmse(real, predictions[key], test_mask) for key, _ in ordered], axis=0)
    sample_score = rmse[0] - rmse.min(axis=0)
    sample_indices = np.argsort(sample_score)[-40:][::-1]
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=240)
    im = ax.imshow(rmse[:, sample_indices], cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([label for _, label in ordered], fontsize=8)
    ax.set_xticks(np.arange(len(sample_indices)))
    ax.set_xticklabels(sample_indices.astype(str), rotation=90, fontsize=6)
    ax.set_xlabel("Representative sample index")
    ax.set_title("Per-sample RMSE Heatmap Across Ablation Models")
    fig.colorbar(im, ax=ax, shrink=0.88, pad=0.01, label="RMSE")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_study_summary(
    real: np.ndarray,
    test_mask: np.ndarray,
    metrics_rows: list[dict[str, str | float]],
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    ordered = method_order(predictions)
    labels = [label for _, label in ordered]
    short = [SHORT_LABELS.get(key, label) for key, label in ordered]
    rmse = np.asarray([float(row["RMSE"]) for row in metrics_rows], dtype=np.float32)
    mae = np.asarray([float(row["MAE"]) for row in metrics_rows], dtype=np.float32)
    phys = np.asarray([float(row["Physics Residual"]) for row in metrics_rows], dtype=np.float32)
    comp = np.asarray([float(row["Composite Score"]) for row in metrics_rows], dtype=np.float32)
    sample_rmse = np.stack([per_sample_rmse(real, predictions[key], test_mask) for key, _ in ordered], axis=0)

    fig = plt.figure(figsize=(13.2, 8.2), dpi=240)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], width_ratios=[1.1, 1.0, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    x = np.arange(len(short))
    ax1.plot(x, rmse, marker="o", linewidth=2.0, label="RMSE")
    ax1.plot(x, mae, marker="s", linewidth=2.0, label="MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=18, ha="right")
    ax1.set_title("Prediction error")
    ax1.grid(alpha=0.24)
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(x, phys, marker="o", linewidth=2.0, color="#e76f51")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=18, ha="right")
    ax2.set_title("Physics residual")
    ax2.grid(alpha=0.24)

    ax3.plot(x, comp, marker="o", linewidth=2.0, color="#2a9d8f")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short, rotation=18, ha="right")
    ax3.set_title("Composite score")
    ax3.grid(alpha=0.24)

    im = ax4.imshow(sample_rmse, cmap="magma", aspect="auto")
    ax4.set_yticks(np.arange(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8)
    tick_positions = np.linspace(0, sample_rmse.shape[1] - 1, 12, dtype=int)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_positions.astype(str), fontsize=7)
    ax4.set_xlabel("Sample index")
    ax4.set_title("Per-sample RMSE for all ablation models")
    fig.colorbar(im, ax=ax4, shrink=0.85, pad=0.01, label="RMSE")
    fig.suptitle("Ablation Study Summary", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Ablations isolate quantum feature encoding, dynamic loss balancing, Monte Carlo neuron thresholding, and residual-adaptive weighting.",
        ha="center",
        fontsize=8.8,
        color="#586a7c",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_per_sample_rmse(real: np.ndarray, test_mask: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=240)
    for key, label in method_order(predictions):
        rmse = per_sample_rmse(real, predictions[key], test_mask)
        ax.plot(np.arange(rmse.size), rmse, linewidth=1.7, label=label)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Held-out RMSE")
    ax.set_title("Per-sample Held-out RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_improvement_histogram(real: np.ndarray, test_mask: np.ndarray, baseline: np.ndarray, best: np.ndarray, best_label: str, path: Path) -> None:
    baseline_rmse = per_sample_rmse(real, baseline, test_mask)
    best_rmse = per_sample_rmse(real, best, test_mask)
    gain = baseline_rmse - best_rmse
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=240)
    ax.hist(gain, bins=24, color="#2a9d8f", alpha=0.86, edgecolor="white")
    ax.axvline(0.0, color="#263746", linewidth=1.5)
    ax.set_xlabel("RMSE reduction vs baseline")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Per-sample Improvement Distribution: {best_label}")
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_parity_scatter(real: np.ndarray, test_mask: np.ndarray, baseline: np.ndarray, best: np.ndarray, best_label: str, path: Path) -> None:
    mask = test_mask.astype(bool)
    rng = np.random.default_rng(2026)
    target = real[mask]
    base = baseline[mask]
    improved = best[mask]
    size = min(9000, target.size)
    idx = rng.choice(target.size, size=size, replace=False)
    lo = float(np.percentile(target[idx], 1))
    hi = float(np.percentile(target[idx], 99))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=240, sharex=True, sharey=True)
    for ax, pred, title in [(axes[0], base, "Baseline PINN"), (axes[1], improved, best_label)]:
        ax.scatter(target[idx], pred[idx], s=4, alpha=0.22, color="#355070", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="#e76f51", linewidth=1.6)
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Predicted value")
    for ax in axes:
        ax.set_xlabel("Target value")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.suptitle("Held-out Prediction Parity Plot", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_error_histogram(real: np.ndarray, test_mask: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    mask = test_mask.astype(bool)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=240)
    for key, label in method_order(predictions):
        err = np.abs(predictions[key] - real)[mask]
        ax.hist(err, bins=60, density=True, histtype="step", linewidth=1.8, label=label)
    ax.set_xlabel("Held-out absolute error")
    ax.set_ylabel("Density")
    ax.set_title("Absolute Error Density")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_error_cdf(real: np.ndarray, test_mask: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    mask = test_mask.astype(bool)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=240)
    for key, label in method_order(predictions):
        err = np.sort(np.abs(predictions[key] - real)[mask])
        y = np.linspace(0.0, 1.0, err.size)
        ax.plot(err, y, linewidth=1.8, label=label)
    ax.set_xlabel("Held-out absolute error")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Absolute Error CDF")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_absolute_error_maps(real: np.ndarray, test_mask: np.ndarray, baseline: np.ndarray, best: np.ndarray, best_label: str, path: Path) -> None:
    samples = select_representative_samples(real, test_mask, baseline, best, count=4)
    fig, axes = plt.subplots(len(samples), 4, figsize=(9.8, 2.35 * len(samples)), dpi=240)
    if len(samples) == 1:
        axes = axes[None, :]
    errors = np.concatenate([np.abs(baseline[samples] - real[samples]).ravel(), np.abs(best[samples] - real[samples]).ravel()])
    vmax = max(float(np.percentile(errors, 98)), 1e-6)
    for row, idx in enumerate(samples):
        mask = test_mask[idx].astype(bool)
        base_err = np.where(mask, np.abs(baseline[idx] - real[idx]), np.nan)
        best_err = np.where(mask, np.abs(best[idx] - real[idx]), np.nan)
        gain = np.where(mask, np.abs(baseline[idx] - real[idx]) - np.abs(best[idx] - real[idx]), np.nan)
        images = [real[idx], baseline[idx], best_err, gain]
        titles = ["Target", "Baseline prediction", f"{best_label} error", "Error reduction"]
        for col, image in enumerate(images):
            ax = axes[row, col]
            if col < 2:
                ax.imshow(image, cmap="viridis", interpolation="nearest")
            elif col == 2:
                ax.imshow(image, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            else:
                limit = max(float(np.nanpercentile(np.abs(gain), 98)), 1e-6)
                ax.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Sample {idx}", fontsize=8)
    fig.suptitle("Held-out Error Maps and Reductions", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_residual_correction_effect(real: np.ndarray, test_mask: np.ndarray, best_key: str, best_label: str, path: Path) -> None:
    raw_path = FIGURE_DIR / "runs" / best_key / "raw_predicted_log_perm.npy"
    corrected_path = FIGURE_DIR / "runs" / best_key / "predicted_log_perm.npy"
    if not raw_path.exists() or not corrected_path.exists():
        return
    raw = np.load(raw_path)
    corrected = np.load(corrected_path)
    raw_rmse = per_sample_rmse(real, raw, test_mask)
    corrected_rmse = per_sample_rmse(real, corrected, test_mask)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=240)
    axes[0].scatter(raw_rmse, corrected_rmse, s=12, alpha=0.65, color="#457b9d", linewidths=0)
    limit = max(float(raw_rmse.max()), float(corrected_rmse.max()))
    axes[0].plot([0, limit], [0, limit], color="#e76f51", linewidth=1.5)
    axes[0].set_xlabel("Raw RMSE")
    axes[0].set_ylabel("Corrected RMSE")
    axes[0].set_title("Per-sample RMSE")
    gain = raw_rmse - corrected_rmse
    axes[1].hist(gain, bins=24, color="#52b788", edgecolor="white")
    axes[1].axvline(0.0, color="#263746", linewidth=1.5)
    axes[1].set_xlabel("RMSE reduction")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Correction gain")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle(f"Observed-residual Correction Effect: {best_label}", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_physics_correlation(fields: np.ndarray, conditions: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    expected = expected_log_perm_index(conditions, PhysicalConfig())
    actual_mean = fields.reshape(fields.shape[0], -1).mean(axis=1)
    slope, intercept = np.polyfit(expected.astype(np.float64), actual_mean.astype(np.float64), deg=1)
    fitted_expected = slope * expected + intercept
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=240)
    ax.scatter(fitted_expected, actual_mean, s=18, alpha=0.55, label="Target", color="#111827")
    for key, label in method_order(predictions):
        pred_mean = predictions[key].reshape(predictions[key].shape[0], -1).mean(axis=1)
        ax.scatter(fitted_expected, pred_mean, s=12, alpha=0.28, label=label)
    ax.set_xlabel("Fitted physical expected mean")
    ax.set_ylabel("Predicted field mean")
    ax.set_title("Physics Expected-mean Correlation")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_observation_density(train_mask: np.ndarray, test_mask: np.ndarray, path: Path) -> None:
    train_density = train_mask.mean(axis=0)
    test_density = test_mask.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), dpi=240)
    images = [train_density, test_density, train_density - test_density]
    titles = ["Train observation frequency", "Held-out frequency", "Train - held-out"]
    for ax, image, title in zip(axes, images, titles):
        cmap = "viridis" if "frequency" in title else "coolwarm"
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Spatial Mask Density", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_contact_sheet(real: np.ndarray, test_mask: np.ndarray, baseline: np.ndarray, best: np.ndarray, best_label: str, path: Path) -> None:
    base_rmse = per_sample_rmse(real, baseline, test_mask)
    best_rmse = per_sample_rmse(real, best, test_mask)
    choices = np.unique(
        np.concatenate(
            [
                np.argsort(best_rmse)[:3],
                np.argsort(best_rmse)[len(best_rmse) // 2 : len(best_rmse) // 2 + 3],
                np.argsort(best_rmse)[-3:],
            ]
        )
    )[:9]
    fig, axes = plt.subplots(len(choices), 3, figsize=(7.4, 2.2 * len(choices)), dpi=220)
    if len(choices) == 1:
        axes = axes[None, :]
    vmin = float(np.percentile(real[choices], 1))
    vmax = float(np.percentile(real[choices], 99))
    for row, idx in enumerate(choices):
        for col, image in enumerate([real[idx], baseline[idx], best[idx]]):
            ax = axes[row, col]
            ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(["Target", "Baseline", best_label][col], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"#{idx}", fontsize=8)
    fig.suptitle("Representative Reconstruction Contact Sheet", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def update_manifest(extra_files: list[str]) -> None:
    manifest_path = FIGURE_DIR / "figure_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest["extended_paper_figures"] = extra_files
    manifest["kept_images"] = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    fields, conditions, train_mask, test_mask, metrics_rows, predictions = load_inputs()
    best_key = best_method(metrics_rows)
    best_label = method_label(best_key)
    baseline = predictions["standard-pinn"]
    best = predictions[best_key]

    outputs = {
        "ablation_design_matrix.png": lambda path: save_ablation_design_matrix(path, FIGURE_DIR / "ablation_design_matrix.csv"),
        "ablation_study_summary.png": lambda path: save_ablation_study_summary(fields, test_mask, metrics_rows, predictions, path),
        "ablation_module_contribution.png": lambda path: save_ablation_module_contribution(metrics_rows, path),
        "ablation_path_metrics.png": lambda path: save_ablation_path_lines(metrics_rows, path),
        "ablation_metric_heatmap.png": lambda path: save_ablation_metric_heatmap(metrics_rows, path),
        "ablation_sample_rmse_heatmap.png": lambda path: save_ablation_sample_rmse_heatmap(fields, test_mask, predictions, path),
        "metric_radar.png": lambda path: save_metric_radar(metrics_rows, path),
        "metric_ranking_bars.png": lambda path: save_metric_ranking(metrics_rows, path),
        "per_sample_rmse_curve.png": lambda path: save_per_sample_rmse(fields, test_mask, predictions, path),
        "rmse_improvement_histogram.png": lambda path: save_improvement_histogram(fields, test_mask, baseline, best, best_label, path),
        "parity_scatter_best_vs_baseline.png": lambda path: save_parity_scatter(fields, test_mask, baseline, best, best_label, path),
        "absolute_error_density.png": lambda path: save_error_histogram(fields, test_mask, predictions, path),
        "absolute_error_cdf.png": lambda path: save_error_cdf(fields, test_mask, predictions, path),
        "heldout_error_maps.png": lambda path: save_absolute_error_maps(fields, test_mask, baseline, best, best_label, path),
        "residual_correction_effect.png": lambda path: save_residual_correction_effect(fields, test_mask, best_key, best_label, path),
        "physics_correlation_scatter.png": lambda path: save_physics_correlation(fields, conditions, predictions, path),
        "spatial_mask_density.png": lambda path: save_observation_density(train_mask, test_mask, path),
        "representative_reconstruction_contact_sheet.png": lambda path: save_prediction_contact_sheet(fields, test_mask, baseline, best, best_label, path),
    }

    written = []
    for name, fn in outputs.items():
        path = FIGURE_DIR / name
        fn(path)
        if path.exists():
            written.append(name)

    update_manifest(written)
    summary = {
        "best_method": best_key,
        "best_label": best_label,
        "generated": written,
        "count": len(written),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
