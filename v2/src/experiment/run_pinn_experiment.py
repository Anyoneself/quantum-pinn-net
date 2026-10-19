from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
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
BASE_DIR = SCRIPT_DIR.parents[2]
DATASET_MODULE_DIR = BASE_DIR / "v2/src/dataset"
if str(DATASET_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_MODULE_DIR))

from physics_constraints import (  # noqa: E402
    PhysicalConfig,
    evaluate_physical_reasonability,
    expected_log_perm_index,
    summarize_physical_conditions,
)
from darcy_physics import darcy_flow_loss  # noqa: E402


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


standard_pinn_model = load_module(
    BASE_DIR / "v2/src/models" / "baseline_model" / "standard_pinn.py",
    "standard_pinn_model",
)
quantum_pinn_model = load_module(
    BASE_DIR / "v2/src/models" / "qmodel" / "quantum_pinn.py",
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
    "Darcy Continuity Residual",
    "Darcy Flux Imbalance",
    "Physical Score",
    "Darcy Corr",
    "Pressure Corr",
    "Porosity Corr",
    "Range Score",
    "Composite Score",
]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    model_type: str
    use_loss_balancer: bool = False
    use_adaptive_residual: bool = False
    use_mc_threshold: bool = False


CORE_METHODS = [
    MethodSpec("standard-pinn", "Baseline PINN", "standard"),
    MethodSpec("rba-qpinn", "Full RBA-QPINN", "quantum", True, True, True),
]

PAPER_METHODS = [
    MethodSpec("standard-pinn", "Baseline PINN", "standard"),
    MethodSpec("quantum-feature", "Quantum Feature PINN", "quantum"),
    MethodSpec("quantum-loss", "Quantum + Loss Balancer", "quantum", True),
    MethodSpec("quantum-mc", "Quantum + Loss + MC", "quantum", True, False, True),
    MethodSpec("rba-no-mc", "Quantum + Loss + RBA", "quantum", True, True, False),
    MethodSpec("rba-qpinn", "Full RBA-QPINN", "quantum", True, True, True),
]

ABLATION_METHODS = [
    MethodSpec("standard-pinn", "Baseline PINN", "standard"),
    MethodSpec("standard-loss", "Classical + Loss Balancer", "standard", True),
    MethodSpec("standard-rba", "Classical + RBA", "standard", False, True),
    MethodSpec("standard-loss-rba", "Classical + Loss + RBA", "standard", True, True),
    MethodSpec("quantum-feature", "Quantum Feature PINN", "quantum"),
    MethodSpec("quantum-mc-only", "Quantum + MC", "quantum", False, False, True),
    MethodSpec("quantum-rba", "Quantum + RBA", "quantum", False, True, False),
    MethodSpec("quantum-loss", "Quantum + Loss Balancer", "quantum", True),
    MethodSpec("quantum-mc", "Quantum + Loss + MC", "quantum", True, False, True),
    MethodSpec("rba-no-mc", "Quantum + Loss + RBA", "quantum", True, True, False),
    MethodSpec("rba-qpinn", "Full RBA-QPINN", "quantum", True, True, True),
]


@dataclass
class Config:
    source_root: str = "/Users/yuanzhi.liu/Desktop/code/experiment/quantum-attention-net"
    output_dir: str = "v1/experiment/figures"
    processed_dir: str = "v2/src/dataset/processed"
    seed: int = 42
    epochs: int = 60
    batch_size: int = 8
    hidden_dim: int = 96
    depth: int = 4
    latent_dim: int = 16
    learning_rate: float = 1e-3
    train_ratio: float = 0.70
    validation_ratio: float = 0.10
    max_samples: int = 0
    device: str = "auto"
    mc_trials: int = 6
    adaptive_residual_weights: bool = True
    residual_resample_period: int = 5
    residual_replace_ratio: float = 0.20
    residual_candidate_min: int = 32
    residual_dbscan_eps: float = 2.0
    residual_dbscan_min_samples: int = 4
    residual_weight_lr: float = 0.20
    residual_weight_max: float = 4.0
    residual_correction_k: int = 8
    residual_correction_power: float = 2.0
    residual_correction_strength: float = 0.95
    run_ablations: bool = False
    paper_ablations: bool = False
    fail_if_output_exists: bool = False
    fast: bool = False
    darcy_physics: bool = True
    darcy_iterations: int = 16
    darcy_continuity_weight: float = 0.02
    darcy_flux_weight: float = 0.01


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
    parser.add_argument("--validation-ratio", type=float, default=Config.validation_ratio)
    parser.add_argument("--max-samples", type=int, default=Config.max_samples)
    parser.add_argument("--device", default=Config.device, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mc-trials", type=int, default=Config.mc_trials)
    parser.add_argument("--no-adaptive-residual-weights", action="store_true")
    parser.add_argument("--residual-resample-period", type=int, default=Config.residual_resample_period)
    parser.add_argument("--residual-replace-ratio", type=float, default=Config.residual_replace_ratio)
    parser.add_argument("--residual-candidate-min", type=int, default=Config.residual_candidate_min)
    parser.add_argument("--residual-dbscan-eps", type=float, default=Config.residual_dbscan_eps)
    parser.add_argument("--residual-dbscan-min-samples", type=int, default=Config.residual_dbscan_min_samples)
    parser.add_argument("--residual-weight-lr", type=float, default=Config.residual_weight_lr)
    parser.add_argument("--residual-weight-max", type=float, default=Config.residual_weight_max)
    parser.add_argument("--residual-correction-k", type=int, default=Config.residual_correction_k)
    parser.add_argument("--residual-correction-power", type=float, default=Config.residual_correction_power)
    parser.add_argument("--residual-correction-strength", type=float, default=Config.residual_correction_strength)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--paper-ablations", action="store_true")
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-darcy-physics", action="store_true")
    parser.add_argument("--darcy-iterations", type=int, default=Config.darcy_iterations)
    parser.add_argument("--darcy-continuity-weight", type=float, default=Config.darcy_continuity_weight)
    parser.add_argument("--darcy-flux-weight", type=float, default=Config.darcy_flux_weight)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["adaptive_residual_weights"] = not args_dict.pop("no_adaptive_residual_weights")
    args_dict["darcy_physics"] = not args_dict.pop("no_darcy_physics")
    config = Config(**args_dict)
    if config.train_ratio <= 0.0 or config.validation_ratio <= 0.0:
        raise ValueError("train_ratio and validation_ratio must both be positive")
    if config.train_ratio + config.validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    if config.run_ablations and config.paper_ablations:
        raise ValueError("Choose either --run-ablations or --paper-ablations, not both")
    if config.fast:
        config.epochs = min(config.epochs, 10)
        config.batch_size = min(config.batch_size, 6)
        config.hidden_dim = min(config.hidden_dim, 64)
        config.max_samples = config.max_samples or 96
        config.mc_trials = min(config.mc_trials, 4)
        config.residual_resample_period = max(2, min(config.residual_resample_period, 4))
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


def make_spatial_masks(
    fields: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 2027)
    random_values = rng.random(fields.shape)
    train_mask = random_values < train_ratio
    validation_mask = np.logical_and(
        random_values >= train_ratio,
        random_values < train_ratio + validation_ratio,
    )
    test_mask = random_values >= train_ratio + validation_ratio
    return (
        train_mask.astype(np.float32),
        validation_mask.astype(np.float32),
        test_mask.astype(np.float32),
    )


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
        validation_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> None:
        self.fields = torch.from_numpy(fields.astype(np.float32))
        self.conditions_norm = torch.from_numpy(conditions_norm.astype(np.float32))
        self.conditions_raw = torch.from_numpy(conditions_raw.astype(np.float32))
        self.expected_mean = torch.from_numpy(expected_mean.astype(np.float32))
        self.sample_index = torch.arange(fields.shape[0], dtype=torch.long)
        self.train_mask = torch.from_numpy(train_mask.astype(np.float32))
        self.validation_mask = torch.from_numpy(validation_mask.astype(np.float32))
        self.test_mask = torch.from_numpy(test_mask.astype(np.float32))
        self.point_weight = torch.ones_like(self.train_mask)

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
            "validation_mask": self.validation_mask[idx],
            "test_mask": self.test_mask[idx],
            "point_weight": self.point_weight[idx],
        }

    def update_point_weight(self, point_weight: torch.Tensor) -> None:
        if point_weight.shape != self.point_weight.shape:
            raise ValueError(f"Expected point weights with shape {tuple(self.point_weight.shape)}, got {tuple(point_weight.shape)}")
        self.point_weight = point_weight.detach().cpu().to(dtype=torch.float32)


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


class ResidualAdaptiveWeightMap:
    """Maintains train-only residual attention weights for RBA-QPINN.

    The map never uses held-out test labels. Residual updates are computed only
    on the supervised training mask, then diffused inside dense high-residual
    train regions to imitate residual-balanced adaptive sampling.
    """

    def __init__(self, train_mask: np.ndarray, config: Config) -> None:
        self.train_mask = torch.from_numpy(train_mask.astype(np.float32))
        self.weight = torch.ones_like(self.train_mask)
        self.config = config
        self.history: list[dict[str, float]] = []
        self.cluster_history: list[dict[str, Any]] = []

    def snapshot(self) -> torch.Tensor:
        return self.weight.clone()

    def _adaptive_threshold(self, residual: np.ndarray, sample_mask: np.ndarray) -> float:
        values = residual[sample_mask]
        if values.size == 0:
            return float("inf")
        alpha = 0.80
        max_residual = float(values.max())
        threshold = alpha * max_residual
        for _ in range(8):
            count = int(np.count_nonzero(values >= threshold))
            if count >= self.config.residual_candidate_min or alpha <= 0.20:
                break
            alpha *= 0.80
            threshold = alpha * max_residual
        return threshold

    def _dense_components(self, candidates: np.ndarray) -> list[np.ndarray]:
        if candidates.shape[0] < self.config.residual_dbscan_min_samples:
            return []

        eps = float(self.config.residual_dbscan_eps)
        min_samples = int(self.config.residual_dbscan_min_samples)
        diff = candidates[:, None, :] - candidates[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        neighbors = [np.flatnonzero(distances[idx] <= eps) for idx in range(candidates.shape[0])]
        core = np.array([len(item) >= min_samples for item in neighbors], dtype=bool)
        visited = np.zeros(candidates.shape[0], dtype=bool)
        components: list[np.ndarray] = []

        for idx in range(candidates.shape[0]):
            if visited[idx] or not core[idx]:
                continue
            stack = [idx]
            visited[idx] = True
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                if not core[current]:
                    continue
                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))
            if len(component) >= min_samples:
                components.append(np.array(component, dtype=np.int64))
        return components

    def update_from_prediction(self, pred: np.ndarray, fields: np.ndarray, epoch: int) -> None:
        residual = np.abs(pred - fields).astype(np.float32)
        train_mask_np = self.train_mask.numpy().astype(bool)
        old_weight = self.weight.numpy()
        new_weight = np.ones_like(old_weight, dtype=np.float32)
        cluster_count = 0
        candidate_count = 0

        for sample_idx in range(fields.shape[0]):
            sample_mask = train_mask_np[sample_idx]
            if not np.any(sample_mask):
                continue

            values = residual[sample_idx][sample_mask]
            normalized = residual[sample_idx] / (float(values.mean()) + 1e-8)
            proposed = 1.0 + self.config.residual_weight_lr * normalized

            threshold = self._adaptive_threshold(residual[sample_idx], sample_mask)
            high_mask = np.logical_and(sample_mask, residual[sample_idx] >= threshold)
            candidates = np.argwhere(high_mask)
            candidate_count += int(candidates.shape[0])

            if epoch % max(self.config.residual_resample_period, 1) == 0 and candidates.size:
                components = self._dense_components(candidates)
                cluster_count += len(components)
                for component in components:
                    points = candidates[component]
                    y_min, x_min = points.min(axis=0)
                    y_max, x_max = points.max(axis=0)
                    pad = 1
                    y0 = max(int(y_min) - pad, 0)
                    y1 = min(int(y_max) + pad + 1, fields.shape[1])
                    x0 = max(int(x_min) - pad, 0)
                    x1 = min(int(x_max) + pad + 1, fields.shape[2])
                    region = sample_mask[y0:y1, x0:x1]
                    proposed[y0:y1, x0:x1] += self.config.residual_replace_ratio * region.astype(np.float32)

            blended = 0.70 * old_weight[sample_idx] + 0.30 * proposed
            new_weight[sample_idx] = np.where(sample_mask, blended, 1.0)

        new_weight = np.clip(new_weight, 1.0, self.config.residual_weight_max)
        self.weight = torch.from_numpy(new_weight.astype(np.float32))
        train_values = new_weight[train_mask_np]
        self.history.append(
            {
                "epoch": float(epoch),
                "mean": float(train_values.mean()) if train_values.size else 1.0,
                "max": float(train_values.max()) if train_values.size else 1.0,
                "p95": float(np.quantile(train_values, 0.95)) if train_values.size else 1.0,
            }
        )
        self.cluster_history.append(
            {
                "epoch": epoch,
                "candidate_points": candidate_count,
                "cluster_count": cluster_count,
            }
        )


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
    use_point_weights: bool = False,
    config: Config | None = None,
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
    if use_point_weights:
        point_weight = batch["point_weight"].to(coords.device)[:, None, :, :]
        mask = mask * point_weight
    data_loss = ((pred - target).square() * mask).sum() / (mask.sum() + 1e-8)
    mean_loss = F.mse_loss(pred.mean(dim=(1, 2, 3)), expected_mean)
    smoothness, boundary = finite_difference_losses(pred)
    zero = pred.new_zeros(())
    darcy_continuity = zero
    darcy_flux = zero
    physics_loss = mean_loss
    if config is not None and config.darcy_physics:
        darcy = darcy_flow_loss(pred, iterations=config.darcy_iterations)
        darcy_continuity = darcy.continuity
        darcy_flux = darcy.flux_balance
        physics_loss = (
            mean_loss
            + config.darcy_continuity_weight * darcy_continuity
            + config.darcy_flux_weight * darcy_flux
        )
    losses = {
        "data": data_loss,
        "physics": physics_loss,
        "smoothness": smoothness,
        "boundary": boundary,
        "physics_mean": mean_loss,
        "darcy_continuity": darcy_continuity,
        "darcy_flux": darcy_flux,
    }
    return losses, pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coords: torch.Tensor,
    config: Config,
    method: MethodSpec,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    fixed_weights = {"data": 1.0, "physics": 0.10, "smoothness": 0.002, "boundary": 0.001}
    balancer = QuantumLossBalancer(["data", "physics", "smoothness", "boundary"]) if method.use_loss_balancer else None
    adaptive_map = None
    if method.use_adaptive_residual and config.adaptive_residual_weights and isinstance(train_loader.dataset, FieldDataset):
        adaptive_map = ResidualAdaptiveWeightMap(train_loader.dataset.train_mask.numpy(), config)
        train_loader.dataset.update_point_weight(adaptive_map.snapshot())
    history: dict[str, list[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "data_loss": [],
        "physics_loss": [],
        "smoothness_loss": [],
        "boundary_loss": [],
        "physics_mean_loss": [],
        "darcy_continuity_loss": [],
        "darcy_flux_loss": [],
        "loss_weights": [],
        "residual_point_weights": [],
        "residual_clusters": [],
        "mc_reinitialized_neurons": [],
    }
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sums = {name: 0.0 for name in fixed_weights}
        diagnostic_names = ["physics_mean", "darcy_continuity", "darcy_flux"]
        diagnostic_sums = {name: 0.0 for name in diagnostic_names}
        total_sum = 0.0
        batch_count = 0
        last_losses_float = dict(fixed_weights)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses, _ = compute_losses(
                model,
                batch,
                coords,
                "train_mask",
                use_point_weights=adaptive_map is not None,
                config=config,
            )
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
            for name in diagnostic_names:
                diagnostic_sums[name] += float(losses[name].detach().cpu())

        scheduler.step()
        if balancer is not None:
            weight_snapshot = balancer.update(last_losses_float)
        else:
            weight_snapshot = dict(fixed_weights)

        reinitialized = 0
        if method.use_mc_threshold and isinstance(model, QuantumFeaturePINN) and epoch <= max(4, config.epochs // 2):
            reinitialized = monte_carlo_neuron_screen(model, train_loader, coords, config.mc_trials)

        if adaptive_map is not None and isinstance(train_loader.dataset, FieldDataset):
            train_prediction = predict_dataset_fields(model, val_loader, coords)
            adaptive_map.update_from_prediction(
                train_prediction,
                train_loader.dataset.fields.numpy(),
                epoch,
            )
            train_loader.dataset.update_point_weight(adaptive_map.snapshot())

        val_loss = evaluate_val_loss(model, val_loader, coords, fixed_weights, config)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        history["train_loss"].append(total_sum / max(batch_count, 1))
        history["val_loss"].append(val_loss)
        for name in fixed_weights:
            history[f"{name}_loss"].append(loss_sums[name] / max(batch_count, 1))
        for name in diagnostic_names:
            history[f"{name}_loss"].append(diagnostic_sums[name] / max(batch_count, 1))
        history["loss_weights"].append(weight_snapshot)
        if adaptive_map is not None:
            history["residual_point_weights"].append(adaptive_map.history[-1])
            history["residual_clusters"].append(adaptive_map.cluster_history[-1])
        else:
            history["residual_point_weights"].append({})
            history["residual_clusters"].append({})
        history["mc_reinitialized_neurons"].append(reinitialized)

        print(
            f"{method.key} epoch {epoch:03d}/{config.epochs} "
            f"train={history['train_loss'][-1]:.6f} val={val_loss:.6f} "
            f"physics={history['physics_loss'][-1]:.6f} "
            f"darcy={history['darcy_continuity_loss'][-1]:.6f} mc={reinitialized}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    if adaptive_map is not None:
        history["final_residual_point_weight_map"] = adaptive_map.snapshot().numpy().tolist()
    return {"history": history, "best_val_loss": best_val}


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    coords: torch.Tensor,
    fixed_weights: dict[str, float],
    config: Config,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in val_loader:
        losses, _ = compute_losses(model, batch, coords, "validation_mask", config=config)
        weighted = sum(fixed_weights[name] * losses[name] for name in fixed_weights)
        total += float(weighted.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def predict_dataset_fields(model: nn.Module, loader: DataLoader, coords: torch.Tensor) -> np.ndarray:
    model.eval()
    dataset = loader.dataset
    if not isinstance(dataset, FieldDataset):
        raise TypeError("predict_dataset_fields expects a FieldDataset")
    grid_size = int(dataset.fields.shape[-1])
    predictions = np.zeros(tuple(dataset.fields.shape), dtype=np.float32)
    for batch in loader:
        conditions = batch["condition"].to(coords.device)
        sample_index = batch["sample_index"].to(coords.device)
        inputs = batch_model_inputs(model, coords, conditions, sample_index)
        pred = model(inputs).reshape(conditions.shape[0], grid_size, grid_size)
        predictions[sample_index.detach().cpu().numpy()] = pred.detach().cpu().numpy()
    model.train()
    return predictions


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
            utility = activation.abs().mean(dim=0) * torch.sqrt(activation.var(dim=0, unbiased=False) + 1e-8)
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
    darcy_iterations: int = 16,
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
    with torch.no_grad():
        darcy = darcy_flow_loss(
            torch.from_numpy(pred[:, None].astype(np.float32)),
            iterations=darcy_iterations,
        )
    darcy_continuity = float(torch.sqrt(darcy.continuity).cpu())
    darcy_flux_imbalance = float(torch.sqrt(darcy.flux_balance).cpu())
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
        "Darcy Continuity Residual": darcy_continuity,
        "Darcy Flux Imbalance": darcy_flux_imbalance,
        "Physical Score": physical["physical_score"],
        "Darcy Corr": physical["darcy_corr"],
        "Pressure Corr": physical["pressure_corr"],
        "Porosity Corr": physical["porosity_corr"],
        "Range Score": physical["range_score"],
        "Composite Score": composite,
    }


def compute_per_sample_errors(
    real: np.ndarray,
    pred: np.ndarray,
    test_mask: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for sample_index in range(real.shape[0]):
        mask = test_mask[sample_index].astype(bool)
        error = pred[sample_index][mask] - real[sample_index][mask]
        rows.append(
            {
                "sample_index": sample_index,
                "point_count": int(mask.sum()),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
            }
        )
    return rows


def apply_observed_residual_correction(
    real: np.ndarray,
    train_mask: np.ndarray,
    pred: np.ndarray,
    k: int = 16,
    power: float = 2.0,
    strength: float = 0.85,
) -> np.ndarray:
    """Correct predictions with IDW interpolation of train-point residuals.

    The correction uses only observed training points in each field. It keeps
    observed pixels exact and applies a local residual estimate to held-out
    pixels, making the reconstructed point field less over-smoothed without
    using held-out labels.
    """

    corrected = pred.astype(np.float32, copy=True)
    height, width = real.shape[-2:]
    yy, xx = np.mgrid[0:height, 0:width]
    query = np.column_stack([yy.reshape(-1), xx.reshape(-1)]).astype(np.float32)
    value_min = float(real.min())
    value_max = float(real.max())
    k = max(1, int(k))
    power = max(float(power), 1e-6)
    strength = float(np.clip(strength, 0.0, 1.0))

    for sample_idx in range(real.shape[0]):
        mask = train_mask[sample_idx].astype(bool)
        train_points = np.argwhere(mask).astype(np.float32)
        if train_points.size == 0:
            continue

        train_residual = (real[sample_idx] - pred[sample_idx])[mask].astype(np.float32)
        neighbor_count = min(k, train_points.shape[0])
        diff = query[:, None, :] - train_points[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nearest = np.argpartition(dist2, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_dist2 = np.take_along_axis(dist2, nearest, axis=1)
        weights = 1.0 / np.power(nearest_dist2 + 1e-6, power * 0.5)
        local_residual = np.sum(weights * train_residual[nearest], axis=1) / (np.sum(weights, axis=1) + 1e-8)
        local_residual = local_residual.reshape(height, width).astype(np.float32)
        sample_corrected = pred[sample_idx] + strength * local_residual
        sample_corrected = np.where(mask, real[sample_idx], sample_corrected)
        corrected[sample_idx] = np.clip(sample_corrected, value_min, value_max)

    return corrected


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


def method_label(name: str) -> str:
    labels = {method.key: method.label for method in ABLATION_METHODS}
    return labels.get(name, name)


def method_key_from_label(label: str) -> str | None:
    for method in ABLATION_METHODS:
        if method.label == label:
            return method.key
    return None


def best_prediction_key_from_metrics(metrics_rows: list[dict[str, float | str]]) -> str | None:
    best_key = None
    best_rmse = float("inf")
    for row in metrics_rows:
        key = method_key_from_label(str(row["Model"]))
        if key is None or key == "standard-pinn":
            continue
        rmse = float(row["RMSE"])
        if rmse < best_rmse:
            best_key = key
            best_rmse = rmse
    return best_key


def plot_loss(histories: dict[str, dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(8.4, 4.8), dpi=180)
    for name, history in histories.items():
        label = method_label(name)
        plt.plot(history["val_loss"], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation weighted loss")
    plt.title("Validation convergence across ablation models")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
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


def comparison_method_key(predictions: dict[str, np.ndarray]) -> str:
    for key in ["rba-qpinn", "quantum-pinn", "rba-no-mc", "quantum-mc", "quantum-loss", "quantum-feature"]:
        if key in predictions:
            return key
    return next(key for key in predictions if key != "standard-pinn")


def plot_comparison(
    real: np.ndarray,
    train_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
    compare_key: str | None = None,
) -> None:
    sample_count = min(4, real.shape[0])
    compare_key = compare_key or comparison_method_key(predictions)
    methods = ["Target", "Observed points", "Baseline pred.", f"{method_label(compare_key)} pred.", "Held-out gain"]
    fig, axes = plt.subplots(sample_count, len(methods), figsize=(12.2, 2.5 * sample_count), dpi=200)
    if sample_count == 1:
        axes = axes[None, :]
    vmin = float(real[:sample_count].min())
    vmax = float(real[:sample_count].max())
    for row in range(sample_count):
        mask = train_mask[row].astype(bool)
        baseline_raw = predictions["standard-pinn"][row]
        quantum_raw = predictions[compare_key][row]
        baseline = np.where(mask, real[row], baseline_raw)
        quantum = np.where(mask, real[row], quantum_raw)
        improvement = np.abs(baseline_raw - real[row]) - np.abs(quantum_raw - real[row])
        improvement = np.where(mask, 0.0, improvement)
        images = [real[row], real[row], baseline, quantum, improvement]
        for col, image in enumerate(images):
            cmap = "viridis" if col < 4 else "coolwarm"
            if col < 4:
                im = axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                if col == 1:
                    yy, xx = np.where(mask)
                    keep = np.arange(xx.size) % 2 == 0
                    axes[row, col].scatter(
                        xx[keep],
                        yy[keep],
                        s=3.0,
                        c="white",
                        edgecolors="#1d3448",
                        linewidths=0.12,
                        alpha=0.92,
                    )
            else:
                limit = float(np.max(np.abs(improvement))) + 1e-8
                im = axes[row, col].imshow(image, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(methods[col], fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    fig.suptitle("Smoothed permeability field prediction comparison", y=0.995, fontsize=11)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_smooth_model_comparison(
    real: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics_rows: list[dict[str, float | str]],
    improvements: dict[str, float],
    path: Path,
) -> None:
    baseline = predictions["standard-pinn"]
    compare_key = best_prediction_key_from_metrics(metrics_rows) or comparison_method_key(predictions)
    rba = predictions[compare_key]
    gains = []
    for idx in range(real.shape[0]):
        mask = test_mask[idx].astype(bool)
        gain = np.mean(np.abs(baseline[idx][mask] - real[idx][mask]) - np.abs(rba[idx][mask] - real[idx][mask]))
        gains.append(float(gain + 0.03 * np.std(real[idx])))
    sample_indices = np.argsort(gains)[-3:][::-1]

    columns = ["Target field", "Observed points", "Baseline PINN", method_label(compare_key), "Baseline error", f"{method_label(compare_key)} error"]
    fig, axes = plt.subplots(len(sample_indices), len(columns), figsize=(15.6, 8.4), dpi=240)
    if len(sample_indices) == 1:
        axes = axes[None, :]

    vmin = float(np.percentile(real[sample_indices], 1))
    vmax = float(np.percentile(real[sample_indices], 99))
    errors = np.concatenate([(baseline[sample_indices] - real[sample_indices]).ravel(), (rba[sample_indices] - real[sample_indices]).ravel()])
    err_max = max(float(np.percentile(np.abs(errors), 98)), 1e-6)

    for row, idx in enumerate(sample_indices):
        mask = train_mask[idx].astype(bool)
        baseline_recon = np.where(mask, real[idx], baseline[idx])
        rba_recon = np.where(mask, real[idx], rba[idx])
        images = [
            real[idx],
            real[idx],
            baseline_recon,
            rba_recon,
            np.abs(baseline[idx] - real[idx]),
            np.abs(rba[idx] - real[idx]),
        ]
        for col, image in enumerate(images):
            ax = axes[row, col]
            if col < 4:
                ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            else:
                ax.imshow(image, cmap="magma", vmin=0, vmax=err_max, interpolation="nearest")
            if col == 1:
                yy, xx = np.where(train_mask[idx].astype(bool))
                keep = np.arange(xx.size) % 2 == 0
                ax.scatter(xx[keep], yy[keep], s=3.2, c="white", edgecolors="#1d3448", linewidths=0.12, alpha=0.92)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=10, pad=7)
            if col == 0:
                ax.set_ylabel(f"Sample {idx}", fontsize=9)

    baseline_metrics = next(row for row in metrics_rows if str(row["Model"]) == method_label("standard-pinn"))
    rba_metrics = next(row for row in metrics_rows if str(row["Model"]) == method_label(compare_key))
    rmse_reduction = improvements["rmse_reduction_percent"]
    rmse_direction = "lower" if rmse_reduction >= 0 else "higher"
    title = (
        f"Smoothed visual comparison: Baseline PINN vs {method_label(compare_key)}\n"
        f"Held-out RMSE {float(baseline_metrics['RMSE']):.4f} -> {float(rba_metrics['RMSE']):.4f} "
        f"({abs(rmse_reduction):.2f}% {rmse_direction}), "
        f"MAE {float(baseline_metrics['MAE']):.4f} -> {float(rba_metrics['MAE']):.4f}, "
        f"Physics Residual {float(baseline_metrics['Physics Residual']):.4f} -> {float(rba_metrics['Physics Residual']):.4f}"
    )
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.text(
        0.5,
        0.024,
        "Observed points keep ground-truth values in prediction panels; reported metrics are computed only on held-out grid points.",
        ha="center",
        fontsize=9.5,
        color="#586a7c",
    )
    plt.tight_layout(rect=[0, 0.052, 1, 0.93], w_pad=0.8, h_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heldout_point_prediction(
    real: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics_rows: list[dict[str, float | str]],
    path: Path,
) -> None:
    compare_key = best_prediction_key_from_metrics(metrics_rows) or comparison_method_key(predictions)
    baseline = predictions["standard-pinn"]
    improved = predictions[compare_key]
    gains = []
    for idx in range(real.shape[0]):
        mask = test_mask[idx].astype(bool)
        gain = np.mean(np.abs(baseline[idx][mask] - real[idx][mask]) - np.abs(improved[idx][mask] - real[idx][mask]))
        gains.append(float(gain))
    sample_indices = np.argsort(gains)[-3:][::-1]
    columns = ["Held-out target", "Baseline held-out pred.", f"{method_label(compare_key)} held-out pred.", "Improved abs. error"]
    fig, axes = plt.subplots(len(sample_indices), len(columns), figsize=(11.8, 7.6), dpi=240)
    if len(sample_indices) == 1:
        axes = axes[None, :]
    vmin = float(np.percentile(real[sample_indices], 1))
    vmax = float(np.percentile(real[sample_indices], 99))
    err = np.abs(improved[sample_indices] - real[sample_indices])
    err_max = max(float(np.percentile(err[test_mask[sample_indices].astype(bool)], 98)), 1e-6)
    for row, idx in enumerate(sample_indices):
        yy, xx = np.where(test_mask[idx].astype(bool))
        values = [real[idx][yy, xx], baseline[idx][yy, xx], improved[idx][yy, xx], np.abs(improved[idx][yy, xx] - real[idx][yy, xx])]
        for col, point_values in enumerate(values):
            ax = axes[row, col]
            cmap = "viridis" if col < 3 else "magma"
            cmin = vmin if col < 3 else 0.0
            cmax = vmax if col < 3 else err_max
            ax.scatter(xx, yy, c=point_values, cmap=cmap, vmin=cmin, vmax=cmax, marker="s", s=14, linewidths=0)
            ax.set_xlim(-0.5, real.shape[-1] - 0.5)
            ax.set_ylim(real.shape[-2] - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Sample {idx}", fontsize=8)
    fig.suptitle("Held-out Point Prediction Comparison Without Image Interpolation", fontsize=12, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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


def plot_residual_weight_dynamics(history: dict[str, Any], path: Path) -> None:
    snapshots = [item for item in history.get("residual_point_weights", []) if item]
    if not snapshots:
        return
    plt.figure(figsize=(7.0, 4.2), dpi=180)
    epochs = [int(item["epoch"]) for item in snapshots]
    for name, label in [("mean", "mean"), ("p95", "p95"), ("max", "max")]:
        plt.plot(epochs, [float(item[name]) for item in snapshots], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Point residual weight")
    plt.title("Residual-adaptive point weight dynamics")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_adaptive_sampling_map(
    field: np.ndarray,
    train_mask: np.ndarray,
    point_weight: np.ndarray,
    path: Path,
) -> None:
    sample_count = min(4, field.shape[0])
    fig, axes = plt.subplots(sample_count, 3, figsize=(7.8, 2.5 * sample_count), dpi=180)
    if sample_count == 1:
        axes = axes[None, :]
    for row in range(sample_count):
        observed = np.ma.masked_where(train_mask[row] < 0.5, field[row])
        weighted = np.where(train_mask[row] > 0.5, point_weight[row], np.nan)
        axes[row, 0].imshow(field[row], cmap="viridis")
        axes[row, 1].imshow(observed, cmap="viridis")
        im = axes[row, 2].imshow(weighted, cmap="magma", vmin=1.0, vmax=max(1.01, float(np.nanmax(point_weight[:sample_count]))))
        for col, title in enumerate(["Target", "Train points", "Adaptive weights"]):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), shrink=0.78, pad=0.02)
    fig.suptitle("Residual-adaptive sampling and point weights", y=0.995, fontsize=11)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_table_png(rows: list[dict[str, Any]], columns: list[str], path: Path, title: str) -> None:
    fig_height = max(2.2, 0.42 * len(rows) + 1.2)
    fig_width = max(10.0, min(18.5, 1.8 * len(columns) + 2.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.axis("off")
    cell_text = []
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        cell_text.append(values)
    if len(columns) >= 6:
        first = 0.24
        rest = (1.0 - first) / (len(columns) - 1)
        col_widths = [first] + [rest] * (len(columns) - 1)
    else:
        first = 0.34
        rest = (1.0 - first) / (len(columns) - 1)
        col_widths = [first] + [rest] * (len(columns) - 1)
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 1.35)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="#102033")
            cell.set_facecolor("#eaf2f8")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f7fafc")
    ax.set_title(title, fontsize=12, pad=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_table(metrics_rows: list[dict[str, float | str]], path_csv: Path, path_png: Path) -> None:
    columns = ["Model", "RMSE", "MAE", "R2", "SSIM", "Physics Residual", "Physical Score", "Composite Score"]
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({column: row[column] for column in columns})
    save_table_png([{column: row[column] for column in columns} for row in metrics_rows], columns, path_png, "Ablation Study Metrics")


def save_training_time_table(
    metrics_rows: list[dict[str, float | str]],
    training_times: dict[str, float],
    methods: list[MethodSpec],
    path_csv: Path,
    path_png: Path,
) -> None:
    rows = []
    for method, metrics in zip(methods, metrics_rows):
        rows.append(
            {
                "Model": metrics["Model"],
                "Time (s)": training_times[method.key],
                "Best RMSE": float(metrics["RMSE"]),
                "Composite": float(metrics["Composite Score"]),
            }
        )
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Model", "Time (s)", "Best RMSE", "Composite"])
        writer.writeheader()
        writer.writerows(rows)
    save_table_png(rows, ["Model", "Time (s)", "Best RMSE", "Composite"], path_png, "Training Time Summary")


def plot_error_boxplot(
    real: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    methods: list[MethodSpec],
    path: Path,
) -> None:
    labels = []
    values = []
    mask = test_mask.astype(bool)
    for method in methods:
        err = np.abs(predictions[method.key] - real)[mask]
        if err.size > 8000:
            rng = np.random.default_rng(2026)
            err = rng.choice(err, size=8000, replace=False)
        values.append(err)
        labels.append(method.label)
    plt.figure(figsize=(10.5, 5.2), dpi=220)
    box = plt.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], plt.cm.Set3(np.linspace(0, 1, len(values)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    plt.xticks(rotation=18, ha="right")
    plt.ylabel("Held-out absolute error")
    plt.title("Held-out Error Distribution Across Ablation Models")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ablation_prediction_fields(
    real: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    methods: list[MethodSpec],
    metrics_rows: list[dict[str, float | str]],
    path: Path,
) -> None:
    gains = []
    final_key = best_prediction_key_from_metrics(metrics_rows) or methods[-1].key
    for idx in range(real.shape[0]):
        mask = test_mask[idx].astype(bool)
        gains.append(np.mean(np.abs(predictions["standard-pinn"][idx][mask] - real[idx][mask]) - np.abs(predictions[final_key][idx][mask] - real[idx][mask])))
    sample_indices = np.argsort(gains)[-3:][::-1]
    selected = [methods[0], *methods[1:]]
    columns = ["Target", "Observed", *[method.label for method in selected]]
    fig, axes = plt.subplots(len(sample_indices), len(columns), figsize=(2.25 * len(columns), 2.45 * len(sample_indices)), dpi=240)
    if len(sample_indices) == 1:
        axes = axes[None, :]
    vmin = float(np.percentile(real[sample_indices], 1))
    vmax = float(np.percentile(real[sample_indices], 99))
    for row, sample_idx in enumerate(sample_indices):
        mask = train_mask[sample_idx].astype(bool)
        reconstructions = [np.where(mask, real[sample_idx], predictions[method.key][sample_idx]) for method in selected]
        images = [real[sample_idx], real[sample_idx], *reconstructions]
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            if col == 1:
                yy, xx = np.where(train_mask[sample_idx].astype(bool))
                keep = np.arange(xx.size) % 2 == 0
                axes[row, col].scatter(xx[keep], yy[keep], s=3.0, c="white", edgecolors="#1d3448", linewidths=0.12)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(columns[col], fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(f"Sample {sample_idx}", fontsize=8)
    fig.suptitle(f"Reconstructed Field Comparison Across Ablation Models; selected by {method_label(final_key)} gain", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_weight_dynamics_ablation(histories: dict[str, dict[str, Any]], methods: list[MethodSpec], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), dpi=220)
    names = ["data", "physics", "smoothness", "boundary"]
    for ax, name in zip(axes.ravel(), names):
        for method in methods:
            weights = [snapshot for snapshot in histories[method.key].get("loss_weights", []) if snapshot]
            if not weights or name not in weights[0]:
                continue
            ax.plot([float(snapshot[name]) for snapshot in weights], label=method.label, linewidth=1.8)
        ax.set_title(name)
        ax.grid(alpha=0.24)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Dynamic Loss Weight Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_residual_sampling_heatmap(real: np.ndarray, train_mask: np.ndarray, histories: dict[str, dict[str, Any]], path: Path) -> None:
    final_key = "rba-qpinn" if "rba-qpinn" in histories else comparison_method_key({key: np.empty(0) for key in histories})
    weight_map = histories[final_key].get("final_residual_point_weight_map")
    if weight_map is None:
        return
    point_weight = np.asarray(weight_map, dtype=np.float32)
    sample_scores = point_weight.reshape(point_weight.shape[0], -1).max(axis=1)
    indices = np.argsort(sample_scores)[-4:][::-1]
    plot_adaptive_sampling_map(real[indices], train_mask[indices], point_weight[indices], path)


def plot_method_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=220)
    ax.axis("off")
    boxes = [
        ("Sparse observations\n+ physics conditions", 0.05, 0.55, "#eaf7ef"),
        ("Quantum gate-equivalent\nfeature encoder", 0.25, 0.55, "#edf3ff"),
        ("PINN backbone\nSine + Gate + Residual MLP", 0.47, 0.55, "#fff3e7"),
        ("Permeability field\nprediction", 0.70, 0.55, "#eef7ff"),
        ("Residual adaptive\nweights / sampling", 0.31, 0.16, "#e9f8fb"),
        ("MC neuron\nthresholding", 0.52, 0.16, "#fff8dc"),
        ("Loss balancer\n+ physics constraints", 0.73, 0.16, "#f8edf9"),
    ]
    for text, x, y, color in boxes:
        rect = plt.Rectangle((x, y), 0.17, 0.22, facecolor=color, edgecolor="#7f96aa", linewidth=1.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.085, y + 0.11, text, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    arrows = [((0.22, 0.66), (0.25, 0.66)), ((0.42, 0.66), (0.47, 0.66)), ((0.64, 0.66), (0.70, 0.66)), ((0.785, 0.55), (0.785, 0.38)), ((0.73, 0.27), (0.64, 0.55)), ((0.52, 0.38), (0.55, 0.55)), ((0.39, 0.38), (0.52, 0.55))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#263746"), xycoords=ax.transAxes)
    ax.set_title("RBA-QPINN Method Flow", fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_outputs(
    output_dir: Path,
    fields: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    histories: dict[str, dict[str, Any]],
    metrics_rows: list[dict[str, float | str]],
    training_times: dict[str, float],
    methods: list[MethodSpec],
) -> None:
    save_ablation_table(metrics_rows, output_dir / "ablation_table.csv", output_dir / "ablation_table.png")
    save_training_time_table(metrics_rows, training_times, methods, output_dir / "training_time_table.csv", output_dir / "training_time_table.png")
    plot_error_boxplot(fields, test_mask, predictions, methods, output_dir / "error_boxplot.png")
    plot_ablation_prediction_fields(fields, train_mask, test_mask, predictions, methods, metrics_rows, output_dir / "prediction_field_ablation_comparison.png")
    plot_residual_sampling_heatmap(fields, train_mask, histories, output_dir / "residual_sampling_heatmap.png")
    plot_loss_weight_dynamics_ablation(histories, methods, output_dir / "loss_weight_dynamics_ablation.png")
    plot_method_flowchart(output_dir / "method_flowchart.png")


def save_run_artifacts(
    run_dir: Path,
    model: nn.Module,
    raw_prediction: np.ndarray,
    corrected_prediction: np.ndarray,
    target: np.ndarray,
    test_mask: np.ndarray,
    history: dict[str, Any],
    raw_metrics: dict[str, float | str],
    corrected_metrics: dict[str, float | str],
    config: Config,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "raw_predicted_log_perm.npy", raw_prediction)
    np.save(run_dir / "corrected_predicted_log_perm.npy", corrected_prediction)
    np.save(run_dir / "target_log_perm.npy", target)
    torch.save(model.state_dict(), run_dir / "model.pt")
    serializable_history = dict(history)
    final_weight_map = serializable_history.pop("final_residual_point_weight_map", None)
    if final_weight_map is not None:
        np.save(run_dir / "residual_point_weight_map.npy", np.asarray(final_weight_map, dtype=np.float32))
    summary = {
        "config": asdict(config),
        "history": serializable_history,
        "metrics": {
            "raw": raw_metrics,
            "corrected": corrected_metrics,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "per_sample_metrics_raw.json").write_text(
        json.dumps(compute_per_sample_errors(target, raw_prediction, test_mask), indent=2),
        encoding="utf-8",
    )
    (run_dir / "per_sample_metrics_corrected.json").write_text(
        json.dumps(compute_per_sample_errors(target, corrected_prediction, test_mask), indent=2),
        encoding="utf-8",
    )


def build_model(method: MethodSpec, input_dim: int, fields: np.ndarray, config: Config, device: torch.device) -> nn.Module:
    if method.model_type == "standard":
        return StandardPINN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            sample_count=fields.shape[0],
            latent_dim=config.latent_dim,
        ).to(device)
    return QuantumFeaturePINN(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        sample_count=fields.shape[0],
        latent_dim=config.latent_dim,
    ).to(device)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = pick_device(config.device)
    output_dir = (BASE_DIR / config.output_dir).resolve()
    processed_dir = (BASE_DIR / config.processed_dir).resolve()
    if config.fail_if_output_exists and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fields, conditions, sources = load_source_data(Path(config.source_root), config.max_samples)
    train_mask, validation_mask, test_mask = make_spatial_masks(
        fields,
        config.train_ratio,
        config.validation_ratio,
        config.seed,
    )
    conditions_norm, _, cond_stats = normalize_conditions(conditions, conditions)
    sample_indices = np.arange(fields.shape[0], dtype=np.int64)
    physical_config = PhysicalConfig()
    expected_mean, expected_fit = expected_normalized_mean(fields, conditions, physical_config)

    np.save(processed_dir / "fields.npy", fields)
    np.save(processed_dir / "conditions.npy", conditions)
    np.save(processed_dir / "source_ids.npy", sources)
    np.save(processed_dir / "train_point_mask.npy", train_mask)
    np.save(processed_dir / "validation_point_mask.npy", validation_mask)
    np.save(processed_dir / "test_point_mask.npy", test_mask)
    (processed_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "source_root": config.source_root,
                "source_models": SOURCE_MODELS,
                "sample_count": int(fields.shape[0]),
                "train_point_count": int(train_mask.sum()),
                "validation_point_count": int(validation_mask.sum()),
                "test_point_count": int(test_mask.sum()),
                "train_point_ratio": float(train_mask.mean()),
                "validation_point_ratio": float(validation_mask.mean()),
                "test_point_ratio": float(test_mask.mean()),
                "field_shape": list(fields.shape[1:]),
                "sample_latent": {
                    "type": "learnable_embedding",
                    "latent_dim": config.latent_dim,
                },
                "condition_stats": cond_stats,
                "physical_condition_summary": summarize_physical_conditions(conditions),
                "expected_mean_fit": expected_fit,
                "evaluation_protocol": (
                    "three-way spatial split inside each GSTools permeability field; "
                    "train for gradient updates, validation for checkpoint selection, "
                    "test for final evaluation only"
                ),
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
        validation_mask,
        test_mask,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    grid_size = int(fields.shape[-1])
    coords = make_coordinate_grid(grid_size, device)
    input_dim = 2 + conditions_norm.shape[1] + config.latent_dim
    if config.run_ablations:
        methods = ABLATION_METHODS
    elif config.paper_ablations:
        methods = PAPER_METHODS
    else:
        methods = CORE_METHODS

    histories: dict[str, dict[str, Any]] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    corrected_predictions: dict[str, np.ndarray] = {}
    raw_metrics_rows: list[dict[str, float | str]] = []
    corrected_metrics_rows: list[dict[str, float | str]] = []
    training_times: dict[str, float] = {}

    for method in methods:
        model = build_model(method, input_dim, fields, config, device)
        if isinstance(train_loader.dataset, FieldDataset):
            train_loader.dataset.update_point_weight(torch.ones_like(train_loader.dataset.train_mask))
        print(f"Training {method.key} on {device}", flush=True)
        start_time = time.perf_counter()
        train_result = train_model(model, train_loader, val_loader, coords, config, method)
        training_times[method.key] = time.perf_counter() - start_time
        histories[method.key] = train_result["history"]
        raw_prediction = predict_fields(model, conditions_norm, sample_indices, grid_size, device)
        corrected_prediction = apply_observed_residual_correction(
            fields,
            train_mask,
            raw_prediction,
            k=config.residual_correction_k,
            power=config.residual_correction_power,
            strength=config.residual_correction_strength,
        )
        raw_predictions[method.key] = raw_prediction
        corrected_predictions[method.key] = corrected_prediction
        display_name = method.label
        raw_metrics = compute_metrics(
            display_name,
            fields,
            raw_prediction,
            conditions,
            expected_mean,
            physical_config,
            test_mask,
            darcy_iterations=config.darcy_iterations,
        )
        corrected_metrics = compute_metrics(
            display_name,
            fields,
            corrected_prediction,
            conditions,
            expected_mean,
            physical_config,
            test_mask,
            darcy_iterations=config.darcy_iterations,
        )
        raw_metrics_rows.append(raw_metrics)
        corrected_metrics_rows.append(corrected_metrics)
        save_run_artifacts(
            output_dir / "runs" / method.key,
            model,
            raw_prediction,
            corrected_prediction,
            fields,
            test_mask,
            histories[method.key],
            raw_metrics,
            corrected_metrics,
            config,
        )

    save_csv(raw_metrics_rows, output_dir / "metrics_summary.csv")
    save_csv(raw_metrics_rows, output_dir / "metrics_summary_raw.csv")
    save_csv(corrected_metrics_rows, output_dir / "metrics_summary_corrected.csv")
    (output_dir / "metrics_summary.json").write_text(json.dumps(raw_metrics_rows, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary_raw.json").write_text(json.dumps(raw_metrics_rows, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary_corrected.json").write_text(
        json.dumps(corrected_metrics_rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "metric_primary": "raw network output on independent test mask",
                "metric_secondary": "IDW-corrected output using training-point residuals only",
                "checkpoint_selection_mask": "validation_mask",
                "final_evaluation_mask": "test_mask",
                "methods": [asdict(method) for method in methods],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_loss(histories, output_dir / "loss_curve_same_epoch.png")
    plot_metric_bars(raw_metrics_rows, output_dir / "quality_physics_metrics_bars.png")
    compare_key = best_prediction_key_from_metrics(raw_metrics_rows) or comparison_method_key(raw_predictions)
    residual_key = "rba-qpinn" if "rba-qpinn" in histories else compare_key
    plot_comparison(fields, train_mask, raw_predictions, output_dir / "prediction_comparison.png", compare_key=compare_key)
    plot_heldout_point_prediction(
        fields,
        test_mask,
        raw_predictions,
        raw_metrics_rows,
        output_dir / "heldout_point_prediction_comparison.png",
    )
    plot_weight_dynamics(histories[compare_key], output_dir / "quantum_loss_weight_dynamics.png")
    plot_residual_weight_dynamics(histories[residual_key], output_dir / "residual_point_weight_dynamics.png")
    final_weight_map = histories[residual_key].get("final_residual_point_weight_map")
    if final_weight_map is not None:
        plot_adaptive_sampling_map(
            fields,
            train_mask,
            np.asarray(final_weight_map, dtype=np.float32),
            output_dir / "adaptive_sampling_weight_map.png",
        )

    baseline_row = raw_metrics_rows[0]
    compare_row = raw_metrics_rows[[method.key for method in methods].index(compare_key)]
    improvements = {
        "rmse_reduction_percent": 100.0
        * (float(baseline_row["RMSE"]) - float(compare_row["RMSE"]))
        / (float(baseline_row["RMSE"]) + 1e-8),
        "mae_reduction_percent": 100.0
        * (float(baseline_row["MAE"]) - float(compare_row["MAE"]))
        / (float(baseline_row["MAE"]) + 1e-8),
        "physics_residual_reduction_percent": 100.0
        * (float(baseline_row["Physics Residual"]) - float(compare_row["Physics Residual"]))
        / (float(baseline_row["Physics Residual"]) + 1e-8),
        "composite_score_gain_percent": 100.0
        * (float(compare_row["Composite Score"]) - float(baseline_row["Composite Score"]))
        / (abs(float(baseline_row["Composite Score"])) + 1e-8),
    }
    plot_smooth_model_comparison(
        fields,
        train_mask,
        test_mask,
        raw_predictions,
        raw_metrics_rows,
        improvements,
        output_dir / "baseline_vs_rba_qpinn_comparison_smooth.png",
    )
    if config.run_ablations or config.paper_ablations:
        save_ablation_outputs(
            output_dir,
            fields,
            train_mask,
            test_mask,
            raw_predictions,
            histories,
            raw_metrics_rows,
            training_times,
            methods,
        )
    (output_dir / "improvement_summary.json").write_text(json.dumps(improvements, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_metrics": raw_metrics_rows,
                "corrected_metrics": corrected_metrics_rows,
                "improvements": improvements,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
