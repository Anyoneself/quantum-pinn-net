from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantum_pinn_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/quantum_pinn_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy import stats


PRIMARY_METRICS = [
    "RMSE",
    "MAE",
    "R2",
    "SSIM",
    "Physics Residual",
    "Darcy Continuity Residual",
    "Darcy Flux Imbalance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate immutable multi-seed RBA-QPINN experiments.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        half_width = float(stats.t.ppf(0.975, values.size - 1) * std / math.sqrt(values.size))
    else:
        half_width = 0.0
    return mean, std, mean - half_width, mean + half_width


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260728, trials: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(trials, values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_evaluation(
    experiment_root: Path,
    output_dir: Path,
    seeds: list[int],
    evaluation: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_seed_rows: list[dict[str, Any]] = []
    method_order: list[str] | None = None
    values_by_method: dict[str, dict[str, list[float]]] = {}

    for seed in seeds:
        seed_dir = experiment_root / f"seed_{seed}"
        rows = load_json(seed_dir / f"metrics_summary_{evaluation}.json")
        labels = [str(row["Model"]) for row in rows]
        if method_order is None:
            method_order = labels
        elif labels != method_order:
            raise ValueError(f"Method order mismatch in {seed_dir}")
        for row in rows:
            model = str(row["Model"])
            values_by_method.setdefault(model, {metric: [] for metric in PRIMARY_METRICS})
            per_seed_row: dict[str, Any] = {"evaluation": evaluation, "seed": seed, "Model": model}
            for metric in PRIMARY_METRICS:
                value = float(row[metric])
                values_by_method[model][metric].append(value)
                per_seed_row[metric] = value
            per_seed_rows.append(per_seed_row)

    if method_order is None:
        raise ValueError("No experiment rows found")

    aggregate_rows: list[dict[str, Any]] = []
    for model in method_order:
        row: dict[str, Any] = {"evaluation": evaluation, "Model": model, "n_seeds": len(seeds)}
        for metric in PRIMARY_METRICS:
            values = np.asarray(values_by_method[model][metric], dtype=np.float64)
            mean, std, ci_low, ci_high = mean_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        aggregate_rows.append(row)

    baseline = method_order[0]
    best = min(method_order[1:], key=lambda model: np.mean(values_by_method[model]["RMSE"]))
    baseline_rmse = np.asarray(values_by_method[baseline]["RMSE"], dtype=np.float64)
    best_rmse = np.asarray(values_by_method[best]["RMSE"], dtype=np.float64)
    difference = baseline_rmse - best_rmse
    t_result = stats.ttest_rel(baseline_rmse, best_rmse)
    try:
        wilcoxon = stats.wilcoxon(baseline_rmse, best_rmse, alternative="greater")
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    bootstrap_low, bootstrap_high = bootstrap_mean_ci(difference)

    per_sample_differences: list[float] = []
    best_key = None
    baseline_key = None
    for seed in seeds:
        manifest = load_json(experiment_root / f"seed_{seed}" / "experiment_manifest.json")
        for method in manifest["methods"]:
            if method["label"] == best:
                best_key = method["key"]
            if method["label"] == baseline:
                baseline_key = method["key"]
        if best_key is None or baseline_key is None:
            raise ValueError("Could not resolve method keys from experiment manifest")
        baseline_samples = load_json(
            experiment_root
            / f"seed_{seed}"
            / "runs"
            / baseline_key
            / f"per_sample_metrics_{evaluation}.json"
        )
        best_samples = load_json(
            experiment_root
            / f"seed_{seed}"
            / "runs"
            / best_key
            / f"per_sample_metrics_{evaluation}.json"
        )
        per_sample_differences.extend(
            float(base["RMSE"]) - float(candidate["RMSE"])
            for base, candidate in zip(baseline_samples, best_samples, strict=True)
        )

    sample_diff = np.asarray(per_sample_differences, dtype=np.float64)
    sample_wilcoxon = stats.wilcoxon(sample_diff, alternative="greater")
    statistics = {
        "evaluation": evaluation,
        "primary_inference_unit": "random seed",
        "baseline": baseline,
        "best_non_baseline_by_mean_rmse": best,
        "n_seeds": len(seeds),
        "baseline_rmse_by_seed": baseline_rmse.tolist(),
        "best_rmse_by_seed": best_rmse.tolist(),
        "paired_rmse_improvement_by_seed": difference.tolist(),
        "mean_rmse_improvement": float(difference.mean()),
        "relative_rmse_improvement_percent": float(100.0 * difference.mean() / baseline_rmse.mean()),
        "paired_t_test_greater": {
            "statistic": float(t_result.statistic),
            "two_sided_pvalue": float(t_result.pvalue),
            "one_sided_pvalue": float(t_result.pvalue / 2.0)
            if float(t_result.statistic) > 0
            else float(1.0 - t_result.pvalue / 2.0),
        },
        "seed_level_wilcoxon_greater": {
            "statistic": wilcoxon_stat,
            "pvalue": wilcoxon_p,
        },
        "bootstrap_95_ci_mean_rmse_improvement": [bootstrap_low, bootstrap_high],
        "supporting_per_sample_wilcoxon_greater": {
            "n_paired_sample_seed_observations": int(sample_diff.size),
            "statistic": float(sample_wilcoxon.statistic),
            "pvalue": float(sample_wilcoxon.pvalue),
            "note": "Supporting analysis only; observations within a seed are not treated as independent primary replicates.",
        },
    }

    (output_dir / f"aggregate_metrics_{evaluation}.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / f"aggregate_metrics_{evaluation}.csv", aggregate_rows)
    return per_seed_rows, aggregate_rows, statistics


def plot_rmse(
    raw_rows: list[dict[str, Any]],
    corrected_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [str(row["Model"]) for row in raw_rows]
    x = np.arange(len(labels))
    width = 0.38
    raw_mean = np.asarray([row["RMSE_mean"] for row in raw_rows], dtype=np.float64)
    raw_std = np.asarray([row["RMSE_std"] for row in raw_rows], dtype=np.float64)
    corrected_mean = np.asarray([row["RMSE_mean"] for row in corrected_rows], dtype=np.float64)
    corrected_std = np.asarray([row["RMSE_std"] for row in corrected_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=220)
    ax.bar(x - width / 2, raw_mean, width, yerr=raw_std, capsize=3, label="Raw", color="#315b7d")
    ax.bar(
        x + width / 2,
        corrected_mean,
        width,
        yerr=corrected_std,
        capsize=3,
        label="IDW-corrected",
        color="#d08c4c",
    )
    ax.set_ylabel("Independent-test RMSE (mean ± SD across seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    output_dir = experiment_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_per_seed, raw_aggregate, raw_statistics = aggregate_evaluation(
        experiment_root,
        output_dir,
        seeds,
        "raw",
    )
    corrected_per_seed, corrected_aggregate, corrected_statistics = aggregate_evaluation(
        experiment_root,
        output_dir,
        seeds,
        "corrected",
    )
    write_csv(output_dir / "per_seed_metrics.csv", raw_per_seed + corrected_per_seed)
    statistics = {"raw": raw_statistics, "corrected": corrected_statistics}
    (output_dir / "statistical_tests.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "torch": torch.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_rmse(raw_aggregate, corrected_aggregate, output_dir / "multiseed_rmse_raw_vs_corrected.png")

    raw_best = raw_statistics["best_non_baseline_by_mean_rmse"]
    corrected_best = corrected_statistics["best_non_baseline_by_mean_rmse"]
    summary = f"""# Multi-seed submission experiment summary

- Seeds: {", ".join(str(seed) for seed in seeds)}
- Primary metric: raw network output on an independent test mask
- Primary inference unit: random seed
- Raw best non-baseline model: {raw_best}
- Raw relative RMSE improvement: {raw_statistics["relative_rmse_improvement_percent"]:.2f}%
- Raw paired t-test one-sided p-value: {raw_statistics["paired_t_test_greater"]["one_sided_pvalue"]:.6g}
- Raw seed-level Wilcoxon one-sided p-value: {raw_statistics["seed_level_wilcoxon_greater"]["pvalue"]:.6g}
- Raw bootstrap 95% CI for mean RMSE improvement: [{raw_statistics["bootstrap_95_ci_mean_rmse_improvement"][0]:.6f}, {raw_statistics["bootstrap_95_ci_mean_rmse_improvement"][1]:.6f}]
- Corrected best non-baseline model: {corrected_best}
- Corrected relative RMSE improvement: {corrected_statistics["relative_rmse_improvement_percent"]:.2f}%
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
