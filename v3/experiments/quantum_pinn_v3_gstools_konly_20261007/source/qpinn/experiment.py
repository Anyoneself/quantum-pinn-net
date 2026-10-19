"""Immutable experiment directories, source snapshots, metrics and paired summaries."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import torch

from .data import Observations, ReferenceCase, generate_case, observe
from .training import evaluate, fit, reconstruction_metrics


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRICS = ("log_k_rmse", "k_relative_l2", "pressure_rmse", "pde_residual_rmse",
                   "downstream_flux_relative_error", "training_seconds")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_observations(path, observations):
    arrays = {}
    for name, value in vars(observations).items():
        if isinstance(value, np.ndarray):
            arrays[name] = value
    for label in ("training", "validation"):
        for name, value in vars(getattr(observations, label)).items():
            arrays[f"{label}_{name}"] = value
    np.savez_compressed(path, **arrays)


def summarize(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["method"], []).append(record)
    summary = {}
    for method, rows in grouped.items():
        summary[method] = {"runs": len(rows)}
        for metric in PRIMARY_METRICS:
            values = np.array([row[metric] for row in rows])
            summary[method][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            }
        # Average initializations within each field before describing field variation.
        field_means = {}
        for field in sorted({row["field_seed"] for row in rows}):
            field_rows = [row for row in rows if row["field_seed"] == field]
            field_means[str(field)] = float(np.mean([row["k_relative_l2"] for row in field_rows]))
        summary[method]["field_mean_k_relative_l2"] = field_means
        summary[method]["uncertainty_note"] = "descriptive run SD; fields and repeated fits are not all independent"
    lookup = {(r["field_seed"], r["seed"], r["method"]): r for r in records}
    pairs = {}
    comparisons = [(f"classical_{s}", f"quantum_{s}") for s in ("plain", "rba", "mc", "rba_mc")]
    comparisons += [(f"{kind}_plain", f"{kind}_{s}") for kind in ("classical", "quantum")
                    for s in ("rba", "mc", "rba_mc")]
    for base, candidate in comparisons:
        differences = []
        for field, seed in sorted({(r["field_seed"], r["seed"]) for r in records}):
            a, b = lookup.get((field, seed, base)), lookup.get((field, seed, candidate))
            if a is not None and b is not None:
                differences.append({"field_seed": field, "seed": seed,
                                    "candidate_minus_baseline": b["k_relative_l2"] - a["k_relative_l2"]})
        if differences:
            pairs[f"{candidate}_vs_{base}"] = {
                "metric": "held-out k_relative_l2; negative favors candidate",
                "runs": differences,
                "mean_difference": float(np.mean([d["candidate_minus_baseline"] for d in differences])),
            }
    return {"methods": summary, "paired_comparisons": pairs}


def write_report(output, summary, manifest):
    lines = [
        "# V3 Darcy inverse reconstruction",
        "",
        "This is a synthetic, dimensionless, per-field reconstruction experiment.",
        "Exact quantum-circuit simulation is not evidence of hardware speedup or quantum advantage.",
        "MC is a paired feature-ablation heuristic; RBA is a classical training aid.",
        "All metrics use raw selected-checkpoint outputs, without interpolation correction.",
        f"Profile: {manifest['profile']}. Status: {manifest['status']}.",
        f"Field generator: {manifest['data_config']['field_generator']}. "
        f"Pressure supervision: {bool(manifest['data_config']['p_train'])}.",
        "Pressure reference values are evaluation-only when pressure supervision is false.",
        "",
        "| Method | Runs | Test k relative L2 | Test pressure RMSE | PDE residual RMSE | Train seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in summary["methods"].items():
        metrics = [row[k]["mean"] for k in ("k_relative_l2", "pressure_rmse", "pde_residual_rmse", "training_seconds")]
        lines.append(f"| {method} | {row['runs']} | " + " | ".join(f"{x:.6g}" for x in metrics) + " |")
    lines.extend([
        "", "See summary.json for paired differences and field-level averages.",
        "A smoke run checks implementation, not convergence, significance, or publication readiness.",
        "Equal epochs and shared observations are controlled; feature-block parameter counts differ.",
        "MC screening overhead is included in training time. There is no matched-wall-time experiment here.",
        "The pressure-network boundary conditions are hard Dirichlet and soft no-flux.",
        "Dataset metadata reports reference grid refinement error; no field-wise normalization is used.",
        "Darcy consistency without pressure observations does not uniquely identify permeability.",
        "GSTools runs generate new fields; they do not reuse V2's normalized/clipped arrays.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_experiment(output, data_config, model_config, train_config, strategy_config,
                   weights, field_seeds, seeds, methods, profile="research"):
    data_config.validate()
    model_config.validate()
    train_config.validate()
    strategy_config.validate(model_config.features)
    for name, value in asdict(weights).items():
        if not np.isfinite(value) or (value < 0 if name == "pressure" else value <= 0):
            raise ValueError("loss weights must be positive except optional zero pressure weight")
    if bool(data_config.p_train) != bool(weights.pressure):
        raise ValueError("pressure observation counts and pressure loss weight must agree")
    from .training import METHODS
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("unknown or empty method selection")
    for values in (field_seeds, seeds, methods):
        if not values or len(values) != len(set(values)):
            raise ValueError("seeds and method selections must be nonempty and unique")
    if min(field_seeds + seeds) < 0:
        raise ValueError("seeds must be nonnegative")
    torch.set_num_threads(train_config.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    sources = list((ROOT / "qpinn").glob("*.py")) + [ROOT / "pyproject.toml"]
    lock = ROOT / "requirements-v3.lock"
    if lock.exists():
        sources.append(lock)
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[str(relative)] = digest(dest)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {
        "schema": 2, "route": "darcy-quantum-pinn-v3", "status": "running", "profile": profile,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_sha256": hashes, "artifact_sha256": {},
        "versions": {name: version(name) for name in (
            ("numpy", "scipy", "torch", "pennylane", "gstools")
            if data_config.field_generator == "gstools" else ("numpy", "scipy", "torch", "pennylane")
        )},
        "python": platform.python_version(), "platform": platform.platform(),
        "data_config": asdict(data_config), "model_config": asdict(model_config),
        "train_config": asdict(train_config), "strategy_config": asdict(strategy_config),
        "loss_weights": asdict(weights), "field_seeds": field_seeds, "seeds": seeds, "methods": methods,
        "expected_runs": len(field_seeds) * len(seeds) * len(methods),
    }
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        for field_seed in field_seeds:
            case = generate_case(data_config, field_seed)
            field_dir = output / f"field_{field_seed}"
            field_dir.mkdir()
            np.savez_compressed(field_dir / "reference.npz", xy=case.xy, log_k=case.log_k, pressure=case.pressure)
            write_json(field_dir / "reference.json", case.metadata)
            for seed in seeds:
                observations = observe(case, data_config, seed)
                seed_dir = field_dir / f"seed_{seed}"
                seed_dir.mkdir()
                save_observations(seed_dir / "observations.npz", observations)
                write_json(seed_dir / "observations.json", observations.metadata)
                for method in methods:
                    run_dir = seed_dir / method
                    run_dir.mkdir()
                    model, info, history, screening = fit(
                        observations.training, observations.validation, method, seed,
                        model_config, train_config, strategy_config, weights,
                    )
                    metrics, arrays = evaluate(model, case, observations)
                    result = {"field_seed": field_seed, **info, **metrics}
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    np.savez_compressed(run_dir / "predictions.npz", **arrays)
                    write_json(run_dir / "history.json", history)
                    write_json(run_dir / "screening.json", screening)
                    write_json(run_dir / "result.json", result)
                    records.append(result)
                    print(f"{len(records)}/{manifest['expected_runs']} field={field_seed} seed={seed} "
                          f"{method} k_rel_l2={metrics['k_relative_l2']:.5f} "
                          f"pde={metrics['pde_residual_rmse']:.5f} "
                          f"best={info['best_epoch']} time={info['training_seconds']:.2f}s", flush=True)
        manifest["status"] = "completed"
        summary = summarize(records)
        write_json(output / "summary.json", summary)
        write_report(output, summary, manifest)
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != output / "manifest.json" and snapshot not in path.parents:
                manifest["artifact_sha256"][str(path.relative_to(output))] = digest(path)
        manifest["completed_runs"] = len(records)
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", completed_runs=len(records), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return summary


def verify_experiment(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("experiment is not completed")
    for prefix, mapping in (("source", manifest["source_sha256"]), ("", manifest["artifact_sha256"])):
        for relative, expected in mapping.items():
            path = (output / prefix / relative).resolve()
            if not path.is_relative_to(output) or digest(path) != expected:
                raise ValueError(f"hash mismatch: {relative}")
    records = []
    for field_seed, seed, method in itertools.product(manifest["field_seeds"], manifest["seeds"], manifest["methods"]):
        field_dir = output / f"field_{field_seed}"
        with np.load(field_dir / "reference.npz", allow_pickle=False) as data:
            case = ReferenceCase(data["xy"], data["log_k"], data["pressure"],
                                 json.loads((field_dir / "reference.json").read_text()))
        seed_dir = field_dir / f"seed_{seed}"
        with np.load(seed_dir / "observations.npz", allow_pickle=False) as data:
            for label in ("k", "p"):
                groups = [data[f"{label}_{part}_indices"] for part in ("train", "validation", "test")]
                indices = np.concatenate(groups)
                if len(indices) != len(case.xy) or not np.array_equal(np.sort(indices), np.arange(len(case.xy))):
                    raise ValueError("invalid observation partition")
            if manifest.get("schema", 1) >= 2 and not manifest["data_config"]["p_train"]:
                for name in ("p_train_indices", "p_validation_indices", "training_p_xy",
                             "validation_p_xy", "training_pressure", "validation_pressure"):
                    if data[name].size:
                        raise ValueError("pressure labels present in permeability-only experiment")
                if manifest["loss_weights"]["pressure"] != 0:
                    raise ValueError("pressure loss enabled without pressure observations")
            # Evaluation requires indices only; no training functions are called here.
            observations = Observations(None, None, *(data[name] for name in (
                "k_train_indices", "k_validation_indices", "k_test_indices",
                "p_train_indices", "p_validation_indices", "p_test_indices")),
                data["evaluation_collocation"], {})
            run_dir = seed_dir / method
            result = json.loads((run_dir / "result.json").read_text())
            history = json.loads((run_dir / "history.json").read_text())
            best = min(history, key=lambda row: row["validation"])
            if len(history) != manifest["train_config"]["epochs"] + 1:
                raise ValueError("incomplete training history")
            if best["epoch"] != result["best_epoch"]:
                raise ValueError("checkpoint was not selected by validation")
            with np.load(run_dir / "predictions.npz", allow_pickle=False) as pred:
                measured = reconstruction_metrics(case, observations, pred["log_k"], pred["pressure"])
                measured["pde_residual_rmse"] = float(np.sqrt(np.mean(pred["physics_residual"]**2)))
                measured["neumann_flux_rmse"] = float(np.sqrt(np.mean(pred["neumann_flux"]**2)))
                kdiff = pred["log_k"][data["k_validation_indices"]] - data["validation_log_k"]
                pdiff = pred["pressure"][data["p_validation_indices"]] - data["validation_pressure"]
                validation = manifest["loss_weights"]["permeability"] * np.mean(kdiff**2)
                if len(pdiff):
                    validation += manifest["loss_weights"]["pressure"] * np.mean(pdiff**2)
                np.testing.assert_allclose(validation, result["best_validation"], rtol=1e-10, atol=1e-12)
                if int(pred["feature_mask"].sum()) != result["active_features_at_best"]:
                    raise ValueError("feature mask mismatch")
                for key, value in measured.items():
                    np.testing.assert_allclose(value, result[key], rtol=1e-10, atol=1e-12, err_msg=key)
            records.append(result)
    if len(records) != manifest["expected_runs"] or len(records) != manifest["completed_runs"]:
        raise ValueError("incomplete experiment matrix")
    if summarize(records) != json.loads((output / "summary.json").read_text()):
        raise ValueError("aggregate summary mismatch")
    return {"verified_runs": len(records), "artifacts": len(manifest["artifact_sha256"]),
            "source_files": len(manifest["source_sha256"]), "status": "passed",
            "scope": "hashes, splits, checkpoint selection, saved-output metrics; not a training rerun"}
