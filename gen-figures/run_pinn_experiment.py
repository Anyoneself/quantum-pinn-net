from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_MODULE_DIR = BASE_DIR / "dataset"
if str(DATASET_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_MODULE_DIR))

from physics_constraints import (  # noqa: E402
    PhysicalConfig,
    evaluate_physical_reasonability,
    expected_log_perm_index,
    summarize_physical_conditions,
)


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


standard_pinn_model = load_module(
    BASE_DIR / "models" / "standard-pinn" / "standard_pinn.py",
    "standard_pinn_model",
)
quantum_pinn_model = load_module(
    BASE_DIR / "models" / "quantum-pinn" / "quantum_pinn.py",
    "quantum_pinn_model",
)
StandardPINN = standard_pinn_model.StandardPINN
QuantumFeaturePINN = quantum_pinn_model.QuantumFeaturePINN
QuantumLossBalancer = quantum_pinn_model.QuantumLossBalancer


SOURCE_MODELS = [
    "gan",
    "unet",
    "unet-attention",
    "unet-quantum",
    "unet-attention-quantum",
]

METRIC_COLUMNS = [
    "Model",
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "Accuracy",
    "R2",
    "SSIM",
    "Physics Residual",
    "Physical Score",
    "Darcy Corr",
    "Pressure Corr",
    "Porosity Corr",
    "Range Score",
    "Composite Score",
]


@dataclass
class Config:
    source_root: str = "/Users/yuanzhi.liu/Desktop/code/experiment/quantum-attention-net"
    output_dir: str = "figures"
    processed_dir: str = "dataset/processed"
    seed: int = 42
    epochs: int = 60
    batch_size: int = 8
    hidden_dim: int = 96
    depth: int = 4
    latent_dim: int = 16
    learning_rate: float = 1e-3
    train_ratio: float = 0.8
    max_samples: int = 0
    device: str = "auto"
    mc_trials: int = 6
    fast: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train baseline PINN and quantum-inspired PINN for permeability-field prediction."
    )
    parser.add_argument("--source-root", default=Config.source_root)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--processed-dir", default=Config.processed_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--train-ratio", type=float, default=Config.train_ratio)
    parser.add_argument("--max-samples", type=int, default=Config.max_samples)
    parser.add_argument("--device", default=Config.device, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mc-trials", type=int, default=Config.mc_trials)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    config = Config(**vars(args))
    if config.fast:
        config.epochs = min(config.epochs, 10)
        config.batch_size = min(config.batch_size, 6)
        config.hidden_dim = min(config.hidden_dim, 64)
        config.max_samples = config.max_samples or 96
        config.mc_trials = min(config.mc_trials, 4)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_source_data(source_root: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []

    for source_idx, model_name in enumerate(SOURCE_MODELS):
        dataset_dir = source_root / "dataset" / "runs" / model_name
        field_path = dataset_dir / "dataset.npy"
        condition_path = dataset_dir / "physical_conditions.npy"
        if not field_path.exists() or not condition_path.exists():
            continue
        model_fields = np.load(field_path).astype(np.float32)
        model_conditions = np.load(condition_path).astype(np.float32)
        count = min(model_fields.shape[0], model_conditions.shape[0])
        fields.append(model_fields[:count, 0])
        conditions.append(model_conditions[:count])
        source_ids.append(np.full(count, source_idx, dtype=np.int64))

    if not fields:
        raise FileNotFoundError(f"No source dataset found under {source_root}/dataset/runs")

    x_fields = np.concatenate(fields, axis=0)
    x_conditions = np.concatenate(conditions, axis=0)
    x_sources = np.concatenate(source_ids, axis=0)

    if max_samples > 0:
        x_fields = x_fields[:max_samples]
        x_conditions = x_conditions[:max_samples]
        x_sources = x_sources[:max_samples]

    return x_fields, x_conditions, x_sources


def make_spatial_masks(fields: np.ndarray, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 2027)
    random_values = rng.random(fields.shape)
    train_mask = random_values < train_ratio
    test_mask = ~train_mask
    return train_mask.astype(np.float32), test_mask.astype(np.float32)


def normalize_conditions(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


class FieldDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        fields: np.ndarray,
        conditions_norm: np.ndarray,
        conditions_raw: np.ndarray,
        expected_mean: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> None:
        self.fields = torch.from_numpy(fields.astype(np.float32))
        self.conditions_norm = torch.from_numpy(conditions_norm.astype(np.float32))
        self.conditions_raw = torch.from_numpy(conditions_raw.astype(np.float32))
        self.expected_mean = torch.from_numpy(expected_mean.astype(np.float32))
        self.sample_index = torch.arange(fields.shape[0], dtype=torch.long)
        self.train_mask = torch.from_numpy(train_mask.astype(np.float32))
        self.test_mask = torch.from_numpy(test_mask.astype(np.float32))

    def __len__(self) -> int:
        return int(self.fields.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "field": self.fields[idx],
            "condition": self.conditions_norm[idx],
            "condition_raw": self.conditions_raw[idx],
            "expected_mean": self.expected_mean[idx],
            "sample_index": self.sample_index[idx],
            "train_mask": self.train_mask[idx],
            "test_mask": self.test_mask[idx],
        }


def make_coordinate_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def model_inputs(coords: torch.Tensor, conditions: torch.Tensor, sample_latent: torch.Tensor) -> torch.Tensor:
    batch_size, point_count = conditions.shape[0], coords.shape[0]
    coords_expand = coords[None, :, :].expand(batch_size, point_count, 2)
    cond_expand = conditions[:, None, :].expand(batch_size, point_count, conditions.shape[-1])
    latent_expand = sample_latent[:, None, :].expand(batch_size, point_count, sample_latent.shape[-1])
    return torch.cat([coords_expand, cond_expand, latent_expand], dim=-1).reshape(batch_size * point_count, -1)


def batch_model_inputs(model: nn.Module, coords: torch.Tensor, conditions: torch.Tensor, sample_index: torch.Tensor) -> torch.Tensor:
    sample_latent = model.sample_embedding(sample_index.to(coords.device))
    return model_inputs(coords, conditions, sample_latent)


def finite_difference_losses(pred_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
    dy = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
    smoothness = dx.square().mean() + dy.square().mean()
    left = pred_grid[:, :, :, 0].mean(dim=-1)
    right = pred_grid[:, :, :, -1].mean(dim=-1)
    top = pred_grid[:, :, 0, :].mean(dim=-1)
    bottom = pred_grid[:, :, -1, :].mean(dim=-1)
    boundary = (left - right).square().mean() + (top - bottom).square().mean()
    return smoothness, boundary


def compute_losses(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    coords: torch.Tensor,
    mask_name: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    fields = batch["field"].to(coords.device)
    conditions = batch["condition"].to(coords.device)
    sample_index = batch["sample_index"].to(coords.device)
    mask = batch[mask_name].to(coords.device)[:, None, :, :]
    expected_mean = batch["expected_mean"].to(coords.device)
    batch_size, grid_size = fields.shape[0], fields.shape[-1]
    inputs = batch_model_inputs(model, coords, conditions, sample_index)
    pred = model(inputs).reshape(batch_size, 1, grid_size, grid_size)
    target = fields[:, None, :, :]
    data_loss = ((pred - target).square() * mask).sum() / (mask.sum() + 1e-8)
    mean_loss = F.mse_loss(pred.mean(dim=(1, 2, 3)), expected_mean)
    smoothness, boundary = finite_difference_losses(pred)
    losses = {
        "data": data_loss,
        "physics": mean_loss,
        "smoothness": smoothness,
        "boundary": boundary,
    }
    return losses, pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coords: torch.Tensor,
    config: Config,
    method: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    fixed_weights = {"data": 1.0, "physics": 0.10, "smoothness": 0.002, "boundary": 0.001}
    balancer = QuantumLossBalancer(["data", "physics", "smoothness", "boundary"]) if method == "quantum-pinn" else None
    history: dict[str, list[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "data_loss": [],
        "physics_loss": [],
        "smoothness_loss": [],
        "boundary_loss": [],
        "loss_weights": [],
        "mc_reinitialized_neurons": [],
    }
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sums = {name: 0.0 for name in fixed_weights}
        total_sum = 0.0
        batch_count = 0
        last_losses_float = dict(fixed_weights)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses, _ = compute_losses(model, batch, coords, "train_mask")
            if balancer is None:
                weights = {name: torch.tensor(weight, device=coords.device) for name, weight in fixed_weights.items()}
            else:
                weights = balancer.weights(losses)
            total = sum(weights[name] * losses[name] for name in fixed_weights)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_count += 1
            total_sum += float(total.detach().cpu())
            last_losses_float = {name: float(losses[name].detach().cpu()) for name in fixed_weights}
            for name in fixed_weights:
                loss_sums[name] += last_losses_float[name]

        scheduler.step()
        if balancer is not None:
            weight_snapshot = balancer.update(last_losses_float)
        else:
            weight_snapshot = dict(fixed_weights)

        reinitialized = 0
        if method == "quantum-pinn" and isinstance(model, QuantumFeaturePINN) and epoch <= max(4, config.epochs // 2):
            reinitialized = monte_carlo_neuron_screen(model, train_loader, coords, config.mc_trials)

        val_loss = evaluate_val_loss(model, val_loader, coords, fixed_weights)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        history["train_loss"].append(total_sum / max(batch_count, 1))
        history["val_loss"].append(val_loss)
        for name in fixed_weights:
            history[f"{name}_loss"].append(loss_sums[name] / max(batch_count, 1))
        history["loss_weights"].append(weight_snapshot)
        history["mc_reinitialized_neurons"].append(reinitialized)

        print(
            f"{method} epoch {epoch:03d}/{config.epochs} "
            f"train={history['train_loss'][-1]:.6f} val={val_loss:.6f} "
            f"physics={history['physics_loss'][-1]:.6f} mc={reinitialized}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val}


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    coords: torch.Tensor,
    fixed_weights: dict[str, float],
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in val_loader:
        losses, _ = compute_losses(model, batch, coords, "test_mask")
        weighted = sum(fixed_weights[name] * losses[name] for name in fixed_weights)
        total += float(weighted.detach().cpu())
        count += 1
    return total / max(count, 1)


def monte_carlo_neuron_screen(
    model: QuantumFeaturePINN,
    loader: DataLoader,
    coords: torch.Tensor,
    trials: int,
) -> int:
    if not hasattr(model, "first_layer"):
        batch = next(iter(loader))
        inputs = batch_model_inputs(model, coords, batch["condition"].to(coords.device), batch["sample_index"].to(coords.device))
        _ = model(inputs[: min(128, inputs.shape[0])])
    if not hasattr(model, "first_layer"):
        return 0

    model.eval()
    utilities = []
    with torch.no_grad():
        iterator = iter(loader)
        for _ in range(max(trials, 1)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            inputs = batch_model_inputs(model, coords, batch["condition"].to(coords.device), batch["sample_index"].to(coords.device))
            features = torch.cat([inputs, model.quantum_features(inputs)], dim=-1)
            activation = model.first_layer(features)
            utility = activation.abs().mean(dim=0)
            utilities.append(utility)
    score = torch.stack(utilities).mean(dim=0)
    threshold = float(torch.quantile(score, 0.05).detach().cpu())
    model.train()
    return model.reset_low_utility_neurons(score, threshold)


@torch.no_grad()
def predict_fields(
    model: nn.Module,
    conditions_norm: np.ndarray,
    sample_indices: np.ndarray,
    grid_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    coords = make_coordinate_grid(grid_size, device)
    preds: list[np.ndarray] = []
    for start in range(0, conditions_norm.shape[0], 16):
        cond = torch.from_numpy(conditions_norm[start : start + 16].astype(np.float32)).to(device)
        sample_index = torch.from_numpy(sample_indices[start : start + 16].astype(np.int64)).to(device)
        inputs = batch_model_inputs(model, coords, cond, sample_index)
        pred = model(inputs).reshape(cond.shape[0], grid_size, grid_size)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def simple_ssim(real: np.ndarray, pred: np.ndarray) -> float:
    values = []
    c1 = 0.01**2
    c2 = 0.03**2
    for idx in range(real.shape[0]):
        x = real[idx].astype(np.float64)
        y = pred[idx].astype(np.float64)
        mu_x = x.mean()
        mu_y = y.mean()
        var_x = x.var()
        var_y = y.var()
        cov = ((x - mu_x) * (y - mu_y)).mean()
        values.append(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))
    return float(np.mean(values))


def compute_metrics(
    name: str,
    real: np.ndarray,
    pred: np.ndarray,
    conditions: np.ndarray,
    expected_mean: np.ndarray,
    physical_config: PhysicalConfig,
    test_mask: np.ndarray,
) -> dict[str, float | str]:
    err = pred - real
    mask = test_mask.astype(bool)
    err_eval = err[mask]
    real_eval = real[mask]
    pred_eval = pred[mask]
    mse = float(np.mean(err_eval**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err_eval)))
    mape = float(np.mean(np.abs(err_eval) / (np.abs(real_eval) + 1e-4)))
    accuracy = float(1.0 - mae / (float(real_eval.max() - real_eval.min()) + 1e-8))
    ss_tot = float(np.sum((real_eval - real_eval.mean()) ** 2))
    ss_res = float(np.sum((pred_eval - real_eval) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    ssim_value = simple_ssim(real, pred)
    pred_mean = pred.reshape(pred.shape[0], -1).mean(axis=1)
    physics_residual = float(np.mean(np.abs(pred_mean - expected_mean)))
    physical = evaluate_physical_reasonability(
        pred,
        conditions,
        physical_config,
        reference_min=float(real.min()),
        reference_max=float(real.max()),
    )
    composite = float(np.mean([accuracy, max(r2, 0.0), ssim_value, 1.0 - min(physics_residual, 1.0), physical["physical_score"]]))
    return {
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Accuracy": accuracy,
        "R2": r2,
        "SSIM": ssim_value,
        "Physics Residual": physics_residual,
        "Physical Score": physical["physical_score"],
        "Darcy Corr": physical["darcy_corr"],
        "Pressure Corr": physical["pressure_corr"],
        "Porosity Corr": physical["porosity_corr"],
        "Range Score": physical["range_score"],
        "Composite Score": composite,
    }


def expected_normalized_mean(fields: np.ndarray, conditions: np.ndarray, physical_config: PhysicalConfig) -> tuple[np.ndarray, dict[str, float]]:
    expected = expected_log_perm_index(conditions, physical_config)
    actual = fields.reshape(fields.shape[0], -1).mean(axis=1)
    slope, intercept = np.polyfit(expected.astype(np.float64), actual.astype(np.float64), deg=1)
    fitted = slope * expected + intercept
    return fitted.astype(np.float32), {"slope": float(slope), "intercept": float(intercept)}


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_loss(histories: dict[str, dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(7.0, 4.2), dpi=180)
    for name, history in histories.items():
        label = "Baseline PINN" if name == "standard-pinn" else "Quantum-PINN"
        plt.plot(history["train_loss"], label=f"{label} train", linewidth=2)
        plt.plot(history["val_loss"], label=f"{label} val", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted loss")
    plt.title("Training convergence")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric_bars(rows: list[dict[str, float | str]], path: Path) -> None:
    metrics = ["RMSE", "MAE", "Physics Residual", "Physical Score", "Composite Score"]
    labels = [str(row["Model"]) for row in rows]
    x = np.arange(len(metrics))
    width = 0.34
    plt.figure(figsize=(8.0, 4.6), dpi=180)
    for idx, row in enumerate(rows):
        values = [float(row[m]) for m in metrics]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=labels[idx])
    plt.xticks(x, metrics, rotation=18, ha="right")
    plt.ylabel("Metric value")
    plt.title("Quality and physics metrics")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(real: np.ndarray, train_mask: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    sample_count = min(4, real.shape[0])
    methods = ["Target", "Observed", "Baseline pred.", "Quantum pred.", "Held-out gain"]
    fig, axes = plt.subplots(sample_count, len(methods), figsize=(12.2, 2.5 * sample_count), dpi=180)
    if sample_count == 1:
        axes = axes[None, :]
    vmin = float(real[:sample_count].min())
    vmax = float(real[:sample_count].max())
    observed_cmap = plt.get_cmap("viridis").copy()
    observed_cmap.set_bad(color="#f2f2f2")
    for row in range(sample_count):
        mask = train_mask[row].astype(bool)
        baseline_raw = predictions["standard-pinn"][row]
        quantum_raw = predictions["quantum-pinn"][row]
        baseline = np.where(mask, real[row], baseline_raw)
        quantum = np.where(mask, real[row], quantum_raw)
        observed = np.ma.masked_where(train_mask[row] < 0.5, real[row])
        improvement = np.abs(baseline_raw - real[row]) - np.abs(quantum_raw - real[row])
        improvement = np.where(mask, 0.0, improvement)
        images = [real[row], observed, baseline, quantum, improvement]
        for col, image in enumerate(images):
            cmap = observed_cmap if col == 1 else ("viridis" if col < 4 else "coolwarm")
            if col < 4:
                im = axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                limit = float(np.max(np.abs(improvement))) + 1e-8
                im = axes[row, col].imshow(image, cmap=cmap, vmin=-limit, vmax=limit)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(methods[col], fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    fig.suptitle("Permeability field prediction comparison", y=0.995, fontsize=11)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_weight_dynamics(history: dict[str, Any], path: Path) -> None:
    weights = history.get("loss_weights", [])
    if not weights:
        return
    names = ["data", "physics", "smoothness", "boundary"]
    plt.figure(figsize=(7.0, 4.2), dpi=180)
    for name in names:
        plt.plot([snapshot[name] for snapshot in weights], label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Quantum loss weight")
    plt.title("Dynamic quantum-inspired PINN loss weights")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_run_artifacts(
    run_dir: Path,
    model: nn.Module,
    prediction: np.ndarray,
    target: np.ndarray,
    history: dict[str, Any],
    metrics: dict[str, float | str],
    config: Config,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "predicted_log_perm.npy", prediction)
    np.save(run_dir / "target_log_perm.npy", target)
    torch.save(model.state_dict(), run_dir / "model.pt")
    summary = {
        "config": asdict(config),
        "history": history,
        "metrics": metrics,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = pick_device(config.device)
    output_dir = (BASE_DIR / config.output_dir).resolve()
    processed_dir = (BASE_DIR / config.processed_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fields, conditions, sources = load_source_data(Path(config.source_root), config.max_samples)
    train_mask, test_mask = make_spatial_masks(fields, config.train_ratio, config.seed)
    conditions_norm, _, cond_stats = normalize_conditions(conditions, conditions)
    sample_indices = np.arange(fields.shape[0], dtype=np.int64)
    physical_config = PhysicalConfig()
    expected_mean, expected_fit = expected_normalized_mean(fields, conditions, physical_config)

    np.save(processed_dir / "fields.npy", fields)
    np.save(processed_dir / "conditions.npy", conditions)
    np.save(processed_dir / "source_ids.npy", sources)
    np.save(processed_dir / "train_point_mask.npy", train_mask)
    np.save(processed_dir / "test_point_mask.npy", test_mask)
    (processed_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "source_root": config.source_root,
                "source_models": SOURCE_MODELS,
                "sample_count": int(fields.shape[0]),
                "train_point_count": int(train_mask.sum()),
                "test_point_count": int(test_mask.sum()),
                "train_point_ratio": float(train_mask.mean()),
                "field_shape": list(fields.shape[1:]),
                "sample_latent": {
                    "type": "learnable_embedding",
                    "latent_dim": config.latent_dim,
                },
                "condition_stats": cond_stats,
                "physical_condition_summary": summarize_physical_conditions(conditions),
                "expected_mean_fit": expected_fit,
                "evaluation_protocol": "spatial holdout prediction inside each GSTools permeability field",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = FieldDataset(
        fields,
        conditions_norm,
        conditions,
        expected_mean,
        train_mask,
        test_mask,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    grid_size = int(fields.shape[-1])
    coords = make_coordinate_grid(grid_size, device)
    input_dim = 2 + conditions_norm.shape[1] + config.latent_dim
    models: dict[str, nn.Module] = {
        "standard-pinn": StandardPINN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            sample_count=fields.shape[0],
            latent_dim=config.latent_dim,
        ).to(device),
        "quantum-pinn": QuantumFeaturePINN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            sample_count=fields.shape[0],
            latent_dim=config.latent_dim,
        ).to(device),
    }

    histories: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for method, model in models.items():
        print(f"Training {method} on {device}", flush=True)
        train_result = train_model(model, train_loader, val_loader, coords, config, method)
        histories[method] = train_result["history"]
        prediction = predict_fields(model, conditions_norm, sample_indices, grid_size, device)
        predictions[method] = prediction
        display_name = "Baseline PINN" if method == "standard-pinn" else "Quantum-PINN"
        metrics = compute_metrics(
            display_name,
            fields,
            prediction,
            conditions,
            expected_mean,
            physical_config,
            test_mask,
        )
        metrics_rows.append(metrics)
        save_run_artifacts(
            output_dir / "runs" / method,
            model,
            prediction,
            fields,
            histories[method],
            metrics,
            config,
        )

    save_csv(metrics_rows, output_dir / "metrics_summary.csv")
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    plot_loss(histories, output_dir / "loss_curve_same_epoch.png")
    plot_metric_bars(metrics_rows, output_dir / "quality_physics_metrics_bars.png")
    plot_comparison(fields, train_mask, predictions, output_dir / "prediction_comparison.png")
    plot_weight_dynamics(histories["quantum-pinn"], output_dir / "quantum_loss_weight_dynamics.png")

    improvements = {
        "rmse_reduction_percent": 100.0
        * (float(metrics_rows[0]["RMSE"]) - float(metrics_rows[1]["RMSE"]))
        / (float(metrics_rows[0]["RMSE"]) + 1e-8),
        "mae_reduction_percent": 100.0
        * (float(metrics_rows[0]["MAE"]) - float(metrics_rows[1]["MAE"]))
        / (float(metrics_rows[0]["MAE"]) + 1e-8),
        "physics_residual_reduction_percent": 100.0
        * (float(metrics_rows[0]["Physics Residual"]) - float(metrics_rows[1]["Physics Residual"]))
        / (float(metrics_rows[0]["Physics Residual"]) + 1e-8),
        "composite_score_gain_percent": 100.0
        * (float(metrics_rows[1]["Composite Score"]) - float(metrics_rows[0]["Composite Score"]))
        / (abs(float(metrics_rows[0]["Composite Score"])) + 1e-8),
    }
    (output_dir / "improvement_summary.json").write_text(json.dumps(improvements, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics_rows, "improvements": improvements}, indent=2), flush=True)


if __name__ == "__main__":
    main()
