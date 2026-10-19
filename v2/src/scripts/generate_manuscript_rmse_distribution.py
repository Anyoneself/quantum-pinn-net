from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


NATURE_FIGURE_SCRIPTS = (
    Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
)
sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment


SEEDS = (42, 43, 44, 45, 46)
MODEL_PATHS = {
    "Fourier comparator": "standard-pinn",
    "Phase-periodic network": "quantum-feature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the distribution-wide manuscript RMSE comparison."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("v1/experiments/rba_qpinn_submission_v1"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("v2/paper/figures/manuscript/field_average_rmse_distribution"),
    )
    return parser.parse_args()


def load_field_rmse(
    experiment_root: Path,
    model_key: str,
) -> np.ndarray:
    per_seed: list[np.ndarray] = []
    expected_indices: np.ndarray | None = None
    for seed in SEEDS:
        path = (
            experiment_root
            / f"seed_{seed}"
            / "runs"
            / model_key
            / "per_sample_metrics_raw.json"
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=int)
        values = np.asarray([float(row["RMSE"]) for row in rows], dtype=float)
        if expected_indices is None:
            expected_indices = indices
        elif not np.array_equal(indices, expected_indices):
            raise ValueError(f"Field ordering differs in {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite RMSE values in {path}")
        per_seed.append(values)
    return np.stack(per_seed, axis=0)


def write_source_data(
    path: Path,
    field_indices: np.ndarray,
    baseline: np.ndarray,
    phase: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "field_index_zero_based",
                "fourier_rmse_mean_across_runs",
                "phase_periodic_rmse_mean_across_runs",
                "paired_difference",
            ]
        )
        for rank, (field_index, baseline_value, phase_value) in enumerate(
            zip(field_indices, baseline, phase, strict=True),
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    int(field_index),
                    float(baseline_value),
                    float(phase_value),
                    float(baseline_value - phase_value),
                ]
            )


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    baseline_runs = load_field_rmse(
        args.experiment_root,
        MODEL_PATHS["Fourier comparator"],
    )
    phase_runs = load_field_rmse(
        args.experiment_root,
        MODEL_PATHS["Phase-periodic network"],
    )
    if baseline_runs.shape != phase_runs.shape:
        raise ValueError("The compared arrays have different shapes")

    baseline_mean = baseline_runs.mean(axis=0)
    phase_mean = phase_runs.mean(axis=0)
    order = np.argsort(baseline_mean)
    baseline_sorted = baseline_mean[order]
    phase_sorted = phase_mean[order]
    difference = baseline_sorted - phase_sorted
    if not np.all(difference > 0):
        raise ValueError("The expected paired ordering does not hold for every field")

    source_data_path = args.output_prefix.with_suffix(".csv")
    write_source_data(
        source_data_path,
        order,
        baseline_sorted,
        phase_sorted,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    width_in = 150 / 25.4
    height_in = 82 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)
    ranks = np.arange(1, baseline_sorted.size + 1)

    ax.fill_between(
        ranks,
        phase_sorted,
        baseline_sorted,
        color="#BFD7EA",
        alpha=0.45,
        linewidth=0,
        label="Paired RMSE difference",
    )
    ax.plot(
        ranks,
        baseline_sorted,
        color="#707070",
        linewidth=1.35,
        label="Fourier comparator",
    )
    ax.plot(
        ranks,
        phase_sorted,
        color="#0072B2",
        linewidth=1.6,
        label="Phase-periodic network",
    )

    ax.set_xlabel("Field rank sorted by Fourier comparator RMSE")
    ax.set_ylabel("Field-averaged held-out RMSE")
    ax.set_xlim(1, baseline_sorted.size)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.legend(loc="upper left", ncol=1)
    ax.text(
        0.985,
        0.055,
        "128/128 fields lower\n5 randomized runs per field",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(args.output_prefix.with_suffix(".alignment.json")),
        overlay_svg=str(args.output_prefix.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )

    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    fig.savefig(args.output_prefix.with_suffix(".svg"))
    plt.close(fig)

    provenance = {
        "experiment_root": str(args.experiment_root),
        "seeds": list(SEEDS),
        "models": MODEL_PATHS,
        "run_count_per_field": len(SEEDS),
        "field_count": int(baseline_sorted.size),
        "selection_rule": "All fields included; sorted by field-averaged Fourier comparator RMSE.",
        "exclusions": [],
        "fields_with_lower_phase_periodic_rmse": int(np.sum(difference > 0)),
        "source_data": str(source_data_path),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
