from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


MODEL_KEYS = [
    "standard-pinn",
    "quantum-feature",
    "quantum-loss",
    "quantum-mc",
    "rba-no-mc",
    "rba-qpinn",
]

ARCHIVE_MODEL_LABELS = {
    "standard-pinn": "Baseline PINN",
    "quantum-feature": "Quantum Feature PINN",
    "quantum-loss": "Quantum + Loss Balancer",
    "quantum-mc": "Quantum + Loss + MC",
    "rba-no-mc": "Quantum + Loss + RBA",
    "rba-qpinn": "Full RBA-QPINN",
}

MODEL_LABELS = {
    "standard-pinn": "Fourier 对照网络",
    "quantum-feature": "周期相位网络",
    "quantum-loss": "周期相位 + 动态权重",
    "quantum-mc": "周期相位 + 动态权重 + MC",
    "rba-no-mc": "周期相位 + 动态权重 + RBA",
    "rba-qpinn": "周期相位 + 动态权重 + RBA + MC",
}

DYNAMIC_KEYS = ["quantum-loss", "quantum-mc", "rba-no-mc", "rba-qpinn"]
RBA_KEYS = ["rba-no-mc", "rba-qpinn"]
MC_KEYS = ["quantum-mc", "rba-qpinn"]
LOSS_SOURCE_KEYS = {
    "data": "data",
    "mean_prior": "physics",
    "smoothness": "smoothness",
    "boundary": "boundary",
}

PRIMARY_METRICS = ["RMSE", "MAE", "R2", "SSIM", "Physics Residual"]
T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute supporting statistics used by the phase-periodic manuscript."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("v1/experiments/rba_qpinn_submission_v1"),
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated randomized run identifiers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v1/experiments/rba_qpinn_submission_v1/aggregate/"
            "manuscript_supporting_statistics.json"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot describe an empty collection")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.quantile(array, 0.50)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {name}")


def simple_global_similarity(real: np.ndarray, pred: np.ndarray) -> float:
    values: list[float] = []
    c1 = 0.01**2
    c2 = 0.03**2
    for index in range(real.shape[0]):
        x = real[index].astype(np.float64)
        y = pred[index].astype(np.float64)
        mean_x = float(x.mean())
        mean_y = float(y.mean())
        variance_x = float(x.var())
        variance_y = float(y.var())
        covariance = float(((x - mean_x) * (y - mean_y)).mean())
        values.append(
            ((2 * mean_x * mean_y + c1) * (2 * covariance + c2))
            / (
                (mean_x**2 + mean_y**2 + c1)
                * (variance_x + variance_y + c2)
            )
        )
    return float(np.mean(values))


def expected_field_mean(fields: np.ndarray, conditions: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pressure = conditions[:, 0].astype(np.float64)
    gradient = conditions[:, 1].astype(np.float64)
    viscosity = conditions[:, 2].astype(np.float64)
    porosity = conditions[:, 3].astype(np.float64)
    flow_rate = conditions[:, 4].astype(np.float64)
    index = (
        np.log(flow_rate / 120.0)
        + np.log(viscosity)
        - np.log(gradient / 0.012)
        + 2.8 * (porosity - 0.22) / 0.22
        - 0.015 * (pressure - 30.0)
    )
    actual = fields.reshape(fields.shape[0], -1).mean(axis=1, dtype=np.float64)
    slope, intercept = np.polyfit(index, actual, deg=1)
    fitted = slope * index + intercept
    return fitted, {"slope": float(slope), "intercept": float(intercept)}


def compute_primary_metrics(
    real: np.ndarray,
    pred: np.ndarray,
    test_mask: np.ndarray,
    expected_mean: np.ndarray,
) -> dict[str, float]:
    mask = test_mask.astype(bool)
    error = pred[mask].astype(np.float64) - real[mask].astype(np.float64)
    real_eval = real[mask].astype(np.float64)
    mse = float(np.mean(error**2))
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum((real_eval - real_eval.mean()) ** 2))
    pred_mean = pred.reshape(pred.shape[0], -1).mean(axis=1, dtype=np.float64)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - residual_sum / (total_sum + 1e-8)),
        "SSIM": simple_global_similarity(real, pred),
        "Physics Residual": float(np.mean(np.abs(pred_mean - expected_mean))),
    }


def validate_masks(seed_dir: Path, field_shape: tuple[int, ...]) -> np.ndarray:
    masks: dict[str, np.ndarray] = {}
    for name in ["train", "validation", "test"]:
        path = seed_dir / "processed" / f"{name}_point_mask.npy"
        raw = np.load(path, allow_pickle=False)
        if raw.shape != field_shape:
            raise ValueError(f"Unexpected mask shape in {path}: {raw.shape}")
        require_finite(str(path), raw)
        if not np.isin(raw, [0, 1]).all():
            raise ValueError(f"Non-binary mask values in {path}")
        masks[name] = raw.astype(bool)
    coverage = sum(mask.astype(np.uint8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise ValueError(f"Masks are not disjoint and complete in {seed_dir}")
    if np.any(masks["test"].reshape(field_shape[0], -1).sum(axis=1) == 0):
        raise ValueError(f"A field has no held-out points in {seed_dir}")
    return masks["test"]


def read_metric_rows(path: Path) -> dict[str, dict[str, float | str]]:
    rows = load_json(path)
    output = {str(row["Model"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate model labels in {path}")
    return output


def validate_aggregate_csv(
    path: Path,
    evaluation: str,
    run_metrics: dict[str, dict[str, list[float]]],
) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["Model"]: row for row in csv.DictReader(handle)}
    maximum_difference = 0.0
    for model_key in MODEL_KEYS:
        label = ARCHIVE_MODEL_LABELS[model_key]
        if label not in rows:
            raise ValueError(f"Missing {label} in {path}")
        row = rows[label]
        if row["evaluation"] != evaluation:
            raise ValueError(f"Unexpected evaluation label in {path}: {row['evaluation']}")
        for metric in PRIMARY_METRICS:
            values = np.asarray(run_metrics[model_key][metric], dtype=np.float64)
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1))
            half_width = T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(values.size)
            expected = {
                f"{metric}_mean": mean,
                f"{metric}_std": standard_deviation,
                f"{metric}_ci95_low": mean - half_width,
                f"{metric}_ci95_high": mean + half_width,
            }
            for column, value in expected.items():
                difference = abs(float(row[column]) - value)
                maximum_difference = max(maximum_difference, difference)
                if difference > 1e-6:
                    raise ValueError(
                        f"Aggregate mismatch in {path}, {label}, {column}: "
                        f"{row[column]} versus {value}"
                    )
    return maximum_difference


def recompute_array_evidence(
    experiment_root: Path,
    seeds: list[int],
) -> tuple[dict[str, dict[tuple[int, int], float]], dict[str, Any]]:
    values: dict[str, dict[tuple[int, int], float]] = {key: {} for key in MODEL_KEYS}
    run_metrics = {
        evaluation: {
            model_key: {metric: [] for metric in PRIMARY_METRICS}
            for model_key in MODEL_KEYS
        }
        for evaluation in ["raw", "corrected"]
    }
    reference_arrays: dict[str, np.ndarray] = {}
    reference_hashes: dict[str, str] = {}
    maximum_metric_difference = 0.0
    maximum_field_rmse_difference = 0.0

    for seed_index, seed in enumerate(seeds):
        seed_dir = experiment_root / f"seed_{seed}"
        processed = seed_dir / "processed"
        paths = {
            "fields": processed / "fields.npy",
            "conditions": processed / "conditions.npy",
            "source_ids": processed / "source_ids.npy",
        }
        arrays = {
            name: np.load(path, allow_pickle=False)
            for name, path in paths.items()
        }
        fields = arrays["fields"]
        conditions = arrays["conditions"]
        source_ids = arrays["source_ids"]
        if fields.shape != (128, 32, 32):
            raise ValueError(f"Unexpected field shape in {paths['fields']}: {fields.shape}")
        if conditions.shape != (128, 5):
            raise ValueError(
                f"Unexpected condition shape in {paths['conditions']}: {conditions.shape}"
            )
        if source_ids.shape[0] != fields.shape[0]:
            raise ValueError(f"Source identifiers do not align in {paths['source_ids']}")
        for name, array in arrays.items():
            require_finite(str(paths[name]), array)
            if seed_index == 0:
                reference_arrays[name] = array.copy()
                reference_hashes[name] = sha256_file(paths[name])
            elif not np.array_equal(array, reference_arrays[name]):
                raise ValueError(f"The fixed dataset differs across runs: {paths[name]}")

        test_mask = validate_masks(seed_dir, fields.shape)
        expected_mean, expected_fit = expected_field_mean(fields, conditions)
        summary = load_json(processed / "dataset_summary.json")
        for key in ["slope", "intercept"]:
            if abs(float(summary["expected_mean_fit"][key]) - expected_fit[key]) > 1e-6:
                raise ValueError(f"Expected-mean fit mismatch in {processed}")

        archived_metrics = {
            evaluation: read_metric_rows(seed_dir / f"metrics_summary_{evaluation}.json")
            for evaluation in ["raw", "corrected"]
        }
        for model_key in MODEL_KEYS:
            run_dir = seed_dir / "runs" / model_key
            raw_prediction = np.load(
                run_dir / "raw_predicted_log_perm.npy", allow_pickle=False
            )
            corrected_prediction = np.load(
                run_dir / "corrected_predicted_log_perm.npy", allow_pickle=False
            )
            predictions = {
                "raw": raw_prediction,
                "corrected": corrected_prediction,
            }
            for evaluation, prediction in predictions.items():
                if prediction.shape != fields.shape:
                    raise ValueError(f"Unexpected prediction shape in {run_dir}: {prediction.shape}")
                require_finite(f"{run_dir}/{evaluation}", prediction)
                metrics = compute_primary_metrics(
                    fields, prediction, test_mask, expected_mean
                )
                archived_row = archived_metrics[evaluation][ARCHIVE_MODEL_LABELS[model_key]]
                for metric, value in metrics.items():
                    difference = abs(float(archived_row[metric]) - value)
                    maximum_metric_difference = max(maximum_metric_difference, difference)
                    if difference > 1e-6:
                        raise ValueError(
                            f"Metric mismatch for {seed}, {model_key}, {evaluation}, {metric}: "
                            f"{archived_row[metric]} versus {value}"
                        )
                    run_metrics[evaluation][model_key][metric].append(value)

            archived_per_field = {
                int(row["sample_index"]): float(row["RMSE"])
                for row in load_json(run_dir / "per_sample_metrics_raw.json")
            }
            if set(archived_per_field) != set(range(fields.shape[0])):
                raise ValueError(f"Incomplete per-field metrics in {run_dir}")
            for field_index in range(fields.shape[0]):
                mask = test_mask[field_index]
                error = (
                    raw_prediction[field_index][mask].astype(np.float64)
                    - fields[field_index][mask].astype(np.float64)
                )
                value = float(np.sqrt(np.mean(error**2)))
                difference = abs(archived_per_field[field_index] - value)
                maximum_field_rmse_difference = max(
                    maximum_field_rmse_difference, difference
                )
                if difference > 1e-6:
                    raise ValueError(
                        f"Per-field RMSE mismatch for {seed}, {model_key}, {field_index}"
                    )
                pair = (seed, field_index)
                values[model_key][pair] = value

    aggregate_differences = {
        evaluation: validate_aggregate_csv(
            experiment_root / "aggregate" / f"aggregate_metrics_{evaluation}.csv",
            evaluation,
            run_metrics[evaluation],
        )
        for evaluation in ["raw", "corrected"]
    }
    expected_pairs = set(values["standard-pinn"])
    for model_key in MODEL_KEYS[1:]:
        if set(values[model_key]) != expected_pairs:
            raise ValueError(f"Unmatched field-run pairs for {model_key}")
    return values, {
        "fixed_dataset_shared_across_runs": True,
        "field_shape": list(reference_arrays["fields"].shape),
        "condition_shape": list(reference_arrays["conditions"].shape),
        "source_identifiers": sorted(
            int(value) for value in np.unique(reference_arrays["source_ids"])
        ),
        "fixed_array_sha256": reference_hashes,
        "mask_check": "binary, pairwise disjoint, and complete in every randomized run",
        "maximum_absolute_metric_difference": maximum_metric_difference,
        "maximum_absolute_per_field_rmse_difference": maximum_field_rmse_difference,
        "maximum_absolute_aggregate_difference": aggregate_differences,
        "note": (
            "Primary metrics and per-field RMSE values were recomputed from fields, "
            "predictions, and held-out masks before comparison with derived JSON/CSV files."
        ),
    }


def field_run_statistics(
    values: dict[str, dict[tuple[int, int], float]],
    seeds: list[int],
) -> dict[str, Any]:
    baseline = values["standard-pinn"]
    pairs = sorted(baseline)
    models: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        model_values = [values[model_key][pair] for pair in pairs]
        row: dict[str, Any] = {
            "label": MODEL_LABELS[model_key],
            "rmse_distribution": describe(model_values),
        }
        if model_key != "standard-pinn":
            wins = sum(values[model_key][pair] < baseline[pair] for pair in pairs)
            row["wins_vs_baseline"] = wins
            row["win_rate_vs_baseline"] = wins / len(pairs)
        models[model_key] = row

    field_ids = sorted({field_id for _, field_id in pairs})
    field_average_differences: list[float] = []
    field_wins = 0
    for field_id in field_ids:
        baseline_mean = float(
            np.mean([baseline[(seed, field_id)] for seed in seeds])
        )
        phase_mean = float(
            np.mean([values["quantum-feature"][(seed, field_id)] for seed in seeds])
        )
        difference = baseline_mean - phase_mean
        field_average_differences.append(difference)
        field_wins += int(difference > 0.0)

    return {
        "pair_count": len(pairs),
        "field_count": len(field_ids),
        "models": models,
        "quantum_feature_field_average_vs_baseline": {
            "fields_with_lower_rmse": field_wins,
            "difference_distribution": describe(field_average_differences),
        },
        "note": (
            "Field-run pairs share fields and fitted model parameters. "
            "These statistics are descriptive and are not independent inferential units."
        ),
    }


def load_run_history(
    experiment_root: Path,
    seed: int,
    model_key: str,
) -> dict[str, Any]:
    path = (
        experiment_root
        / f"seed_{seed}"
        / "runs"
        / model_key
        / "run_summary.json"
    )
    return load_json(path)["history"]


def dynamic_weight_statistics(
    experiment_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in DYNAMIC_KEYS:
        terminal: list[dict[str, float]] = []
        dominant = Counter()
        for seed in seeds:
            history = load_run_history(experiment_root, seed, model_key)
            archived_weights = {
                name: float(value)
                for name, value in history["loss_weights"][-1].items()
            }
            weights = {
                semantic_name: archived_weights[source_name]
                for semantic_name, source_name in LOSS_SOURCE_KEYS.items()
            }
            terminal.append(weights)
            dominant[max(LOSS_SOURCE_KEYS, key=weights.__getitem__)] += 1
        output[model_key] = {
            "label": MODEL_LABELS[model_key],
            "dominant_objective_counts": dict(dominant),
            "terminal_weights": {
                name: describe([row[name] for row in terminal])
                for name in LOSS_SOURCE_KEYS
            },
            "note": (
                "The archive key 'physics' denotes the field-mean prior because Darcy training "
                "is disabled. These weights are updated after training ends, do not enter a "
                "subsequent gradient step, and need not match the validation-selected checkpoint."
            ),
        }
    return output


def rba_weight_statistics(
    experiment_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in RBA_KEYS:
        snapshots: list[dict[str, float]] = []
        for seed in seeds:
            history = load_run_history(experiment_root, seed, model_key)
            valid = [item for item in history["residual_point_weights"] if item]
            if not valid:
                raise ValueError(f"Missing residual weight history for {model_key}, {seed}")
            snapshots.append(valid[-1])
        output[model_key] = {
            "label": MODEL_LABELS[model_key],
            "terminal_point_weights": {
                statistic: describe([float(item[statistic]) for item in snapshots])
                for statistic in ["mean", "p95", "max"]
            },
        }
    return output


def mc_statistics(
    experiment_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in MC_KEYS:
        totals = []
        for seed in seeds:
            history = load_run_history(experiment_root, seed, model_key)
            totals.append(
                float(np.sum(np.asarray(history["mc_reinitialized_neurons"], dtype=np.float64)))
            )
        output[model_key] = {
            "label": MODEL_LABELS[model_key],
            "neuron_level_reinitializations_per_run": describe(totals),
        }
    return output


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one randomized run identifier is required")

    field_values, array_crosscheck = recompute_array_evidence(
        args.experiment_root, seeds
    )
    output = {
        "experiment_root": str(args.experiment_root),
        "randomized_run_ids": seeds,
        "array_crosscheck": array_crosscheck,
        "field_run_statistics": field_run_statistics(field_values, seeds),
        "dynamic_weight_statistics": dynamic_weight_statistics(
            args.experiment_root, seeds
        ),
        "rba_weight_statistics": rba_weight_statistics(args.experiment_root, seeds),
        "mc_statistics": mc_statistics(args.experiment_root, seeds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
