"""配对训练并离线计算NTK；所有新数据限定在指定的新归档目录。"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from core import analyze_checkpoint
from dataset.cases import DataConfig, generate_case, observe
from experiment.runner import ROOT, digest, save_observations, write_json
from metrics.physics import LossWeights
from models.darcy_pinn import ModelConfig
from qpinn.strategies import StrategyConfig
from qpinn.training import TrainConfig, evaluate, fit


def probe_indices(grid, probe_grid):
    """奇数网格比保证探针与参考场单元中心重合，不插值或重生成目标。"""
    if probe_grid < 4 or grid % probe_grid or (grid // probe_grid) % 2 != 1:
        raise ValueError("grid/probe-grid must be an odd integer; probe-grid >= 4")
    ratio = grid // probe_grid
    axis = ratio // 2 + ratio * np.arange(probe_grid)
    return (axis[:, None] * grid + axis[None, :]).ravel()


def source_snapshot(output):
    """冻结实际执行代码和方案；不读取或修改历史实验归档。"""
    sources = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.md"))
    for package in ("qpinn", "models", "metrics", "dataset", "experiment"):
        sources.extend((ROOT / "v3/src" / package).glob("*.py"))
    sources.extend([ROOT / "pyproject.toml", ROOT / "v3/requirements-v3.lock"])
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = digest(destination)
    return hashes


def run(output, field_seeds, seeds, epochs=1000, checkpoints=(0, 100, 300, 1000),
        data_config=None, model_config=None, probe_grid=8, profile="research"):
    """所有方法共享数据和预算；回调只复制权重，不计算核也不读取测试标签。"""
    data_config = DataConfig() if data_config is None else data_config
    model_config = ModelConfig() if model_config is None else model_config
    train_config, weights = TrainConfig(epochs=epochs), LossWeights()
    data_config.validate()
    model_config.validate()
    train_config.validate()
    if data_config.p_train or data_config.p_validation:
        raise ValueError("this NTK protocol is permeability-only")
    for values in (field_seeds, seeds):
        if not values or len(set(values)) != len(values) or min(values) < 0:
            raise ValueError("seeds must be nonnegative, unique and nonempty")
    indices = probe_indices(data_config.grid, probe_grid)
    schedule = sorted({0, epochs, *(e for e in checkpoints if 0 <= e <= epochs)})
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "v3/ntk"):
        raise ValueError("NTK archives must be inside v3/ntk")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = {
        "schema": 1, "route": "v3-finite-width-empirical-ntk", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "profile": profile,
        "data_config": asdict(data_config), "model_config": asdict(model_config),
        "train_config": asdict(train_config), "loss_weights": asdict(weights),
        "strategy_config": asdict(StrategyConfig()),
        "field_seeds": list(field_seeds), "seeds": list(seeds),
        "methods": ["classical_plain", "quantum_plain"],
        "checkpoint_epochs": schedule, "probe_grid": probe_grid,
        "expected_runs": len(field_seeds) * len(seeds) * 2,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "pennylane", "gstools")},
        "source_sha256": source_snapshot(output), "artifact_sha256": {},
        "selection": "minimum validation observation loss, including epoch zero",
        "timing": "training excludes checkpoint callback; offline NTK/evaluation timed separately",
        "limitations": ["unequal feature-block parameter counts", "equal epochs, not equal wall time",
                        "no Fourier/SIREN or entanglement ablation", "six independent fields in default research"],
    }
    write_json(output / "manifest.json", manifest)
    all_records, completed = [], 0
    try:
        for field_seed in field_seeds:
            case = generate_case(data_config, field_seed)
            field_dir = output / f"field_{field_seed}"
            field_dir.mkdir()
            np.savez_compressed(field_dir / "reference.npz", xy=case.xy, log_k=case.log_k,
                                pressure=case.pressure, probe_indices=indices)
            write_json(field_dir / "reference.json", case.metadata)
            for seed in seeds:
                observations = observe(case, data_config, seed)
                seed_dir = field_dir / f"seed_{seed}"
                seed_dir.mkdir()
                save_observations(seed_dir / "observations.npz", observations)
                write_json(seed_dir / "observations.json", observations.metadata)
                # 交替训练次序减少方法与顺序完全重合；仍不构成严格性能基准。
                methods = list(manifest["methods"])
                if (field_seeds.index(field_seed) + seeds.index(seed)) % 2:
                    methods.reverse()
                for method in methods:
                    directory = seed_dir / method
                    directory.mkdir()
                    snapshots, checkpoint_seconds = {}, [0.0]

                    def capture(epoch, model):
                        started = time.perf_counter()
                        snapshots[epoch] = {name: value.detach().cpu().clone()
                                            for name, value in model.state_dict().items()}
                        checkpoint_seconds[0] += time.perf_counter() - started

                    model, info, history, screening = fit(
                        observations.training, observations.validation, method, seed, model_config,
                        train_config, StrategyConfig(), weights, checkpoint_epochs=schedule,
                        checkpoint_callback=capture)
                    if screening:
                        raise AssertionError("plain protocol must not screen features")
                    best_epoch = info["best_epoch"]
                    capture(best_epoch, model)
                    torch.save({"best_epoch": best_epoch, "states": snapshots}, directory / "checkpoints.pt")
                    write_json(directory / "history.json", history)
                    initial = None
                    diagnostic_started = time.perf_counter()
                    for epoch, state in sorted(snapshots.items()):
                        model.load_state_dict(state)
                        arrays, statistics, frequencies, metadata = analyze_checkpoint(
                            model, observations.training, weights, case.xy[indices], case.log_k[indices], initial)
                        if initial is None:
                            initial = {name: arrays[name] for name in ("output_kernel", "loss_kernel")}
                        metrics, predictions = evaluate(model, case, observations)
                        arrays.update({f"prediction_{name}": value for name, value in predictions.items()})
                        snapshot_name = f"epoch_{epoch:05d}"
                        np.savez_compressed(directory / f"{snapshot_name}.npz", **arrays)
                        record = {
                            "field_seed": field_seed, "seed": seed, "method": method, "epoch": epoch,
                            "is_best": epoch == best_epoch,
                            "is_scheduled": epoch in schedule,
                            "validation": history[epoch]["validation"],
                            "metrics": metrics, "ntk": statistics, "frequency": frequencies,
                            "metadata": metadata,
                        }
                        write_json(directory / f"{snapshot_name}.json", record)
                        all_records.append(record)
                        print(f"NTK field={field_seed} seed={seed} {method} epoch={epoch} "
                              f"test_k={metrics['k_relative_l2']:.6f} "
                              f"output_shape_drift={statistics['output']['shape_drift']:.4f}", flush=True)
                    info.update(field_seed=field_seed, checkpoint_seconds=checkpoint_seconds[0],
                                diagnostic_seconds=time.perf_counter() - diagnostic_started)
                    write_json(directory / "training.json", info)
                    completed += 1
                    print(f"COMPLETED {completed}/{manifest['expected_runs']} fits", flush=True)
        write_json(output / "records.json", all_records)
        manifest.update(status="completed", completed_runs=completed, snapshots=len(all_records),
                        completed_utc=datetime.now(timezone.utc).isoformat())
        manifest["artifact_sha256"] = {
            str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json" and "source" not in path.relative_to(output).parts}
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", completed_runs=completed, error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return all_records


def main():
    """命令行只提供明确的规模设置，不提供按测试指标自动搜索的入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-seeds", type=int, nargs="+", default=[411, 422, 433, 444, 455, 466])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 100, 300, 1000])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run(args.output, [411], [101], epochs=2, checkpoints=(0, 1, 2),
            data_config=DataConfig(grid=12, k_train=8, k_validation=4, collocation=4, boundary=2),
            model_config=ModelConfig(features=2, hidden=6, feature_layers=1), probe_grid=4, profile="smoke")
    else:
        run(args.output, args.field_seeds, args.seeds, args.epochs, args.checkpoints)


if __name__ == "__main__":
    main()
