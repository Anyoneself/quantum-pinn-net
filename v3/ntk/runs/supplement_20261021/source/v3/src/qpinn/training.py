"""Train without access to reference fields; evaluate only after checkpoint selection."""

from dataclasses import asdict, dataclass
import copy
import time

import numpy as np
import torch

from dataset.cases import solve_darcy
from models.baseline_model import BaselineDarcyPINN
from models.qmodel import QDarcyPINN, QuantumFeatures
from metrics.physics import components, darcy_residual, flux, objective, tensor, validation_score
from .strategies import ResidualAttention, screen_features


METHODS = tuple(f"{kind}_{strategy}" for kind in ("classical", "quantum")
                for strategy in ("plain", "rba", "mc", "rba_mc"))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    learning_rate: float = 0.001
    gradient_clip: float = 5.0
    threads: int = 1

    def validate(self):
        if self.epochs < 1 or self.threads < 1:
            raise ValueError("epochs and threads must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")


def fit(training, validation, method, seed, model_config, train_config, strategy_config, weights,
        *, checkpoint_epochs=(), checkpoint_callback=None):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    train_config.validate()
    strategy_config.validate(model_config.features)
    kind, strategy = method.split("_", 1)
    model = (BaselineDarcyPINN if kind == "classical" else QDarcyPINN)(model_config, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    attention = ResidualAttention(len(training.collocation), strategy_config)
    mc_rng = np.random.default_rng(seed + 5009)
    ones = torch.ones(len(training.collocation), dtype=torch.float64)
    history, screening = [], []
    best_state = copy.deepcopy(model.state_dict())
    initial_validation = validation_score(model, validation, weights)
    if not np.isfinite(initial_validation):
        raise FloatingPointError("non-finite initial validation loss")
    best_score, best_epoch = initial_validation, 0
    history.append({"epoch": 0, "validation": initial_validation,
                    "active_features": int(model.permeability.feature_mask.sum())})
    # 可选回调仅保存中间权重；NTK等昂贵诊断在训练结束后离线执行。
    checkpoint_seconds = 0.0
    if checkpoint_callback is not None and 0 in checkpoint_epochs:
        checkpoint_callback(0, model)
    screening_seconds = 0.0
    started = time.perf_counter()
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        errors = components(model, training)
        enabled = epoch > strategy_config.warmup
        point_weights = attention.update(errors["physics"]) if "rba" in strategy and enabled else ones
        total, losses = objective(errors, weights, point_weights)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        optimizer.step()
        if "mc" in strategy and enabled and (epoch - strategy_config.warmup) % strategy_config.mc_every == 0:
            start = time.perf_counter()
            event = screen_features(model, training, validation, weights, point_weights,
                                    strategy_config, mc_rng)
            screening_seconds += time.perf_counter() - start
            screening.append({"epoch": epoch, **event})
        score = validation_score(model, validation, weights)
        if not np.isfinite(score):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch, "validation": score, "training_total_pre_step": float(total.detach()),
            "losses_pre_step": {name: float(value.detach()) for name, value in losses.items()},
            "physics_unweighted_mse_pre_step": float(errors["physics"].detach().square().mean()),
            "gradient_norm_pre_clip": float(gradient_norm),
            "rba_weight_min": float(point_weights.min()), "rba_weight_max": float(point_weights.max()),
            "active_features": int(model.permeability.feature_mask.sum()),
        })
        if checkpoint_callback is not None and epoch in checkpoint_epochs:
            checkpoint_started = time.perf_counter()
            checkpoint_callback(epoch, model)
            checkpoint_seconds += time.perf_counter() - checkpoint_started
    elapsed = time.perf_counter() - started - checkpoint_seconds
    model.load_state_dict(best_state)
    model.eval()
    feature_block = model.permeability.features
    info = {
        "method": method, "seed": seed, "epochs": train_config.epochs, "best_epoch": best_epoch,
        "initial_validation": initial_validation, "best_validation": best_score,
        "training_seconds": elapsed, "screening_seconds": screening_seconds,
        "parameter_counts": model.parameter_counts(),
        "active_features_at_best": int(model.permeability.feature_mask.sum()),
        "quantum_forward_calls_training": feature_block.forward_calls if isinstance(feature_block, QuantumFeatures) else 0,
        "quantum_sample_evaluations_training": feature_block.sample_evaluations if isinstance(feature_block, QuantumFeatures) else 0,
        "model_config": asdict(model_config), "train_config": asdict(train_config),
        "strategy_config": asdict(strategy_config), "loss_weights": asdict(weights),
        "selection_rule": "minimum noisy validation observation loss, including epoch 0",
        "observed_variables": ["log_k", "pressure"] if len(training.pressure) else ["log_k"],
        "pressure_supervision": bool(len(training.pressure)),
        "quantum_backend": "default.qubit, exact expectations, torch backprop" if kind == "quantum" else None,
        "training_view_has_test_labels": False,
    }
    return model, info, history, screening


def reconstruction_metrics(case, observations, log_k, pressure):
    ki, pi = observations.k_test_indices, observations.p_test_indices
    true_k, predicted_k = np.exp(case.log_k), np.exp(log_k)
    grid = int(round(np.sqrt(len(log_k))))
    ref_flow = solve_darcy(true_k.reshape(grid, grid))
    pred_flow = solve_darcy(predicted_k.reshape(grid, grid))
    return {
        "log_k_rmse": float(np.sqrt(np.mean((log_k[ki] - case.log_k[ki])**2))),
        "k_rmse": float(np.sqrt(np.mean((predicted_k[ki] - true_k[ki])**2))),
        "k_relative_l2": float(np.linalg.norm(predicted_k[ki] - true_k[ki]) / np.linalg.norm(true_k[ki])),
        "pressure_rmse": float(np.sqrt(np.mean((pressure[pi] - case.pressure[pi])**2))),
        "downstream_pressure_rmse": float(np.sqrt(np.mean((pred_flow["pressure"] - ref_flow["pressure"])**2))),
        "downstream_flux_relative_error": abs(pred_flow["inlet"] - ref_flow["inlet"]) / abs(ref_flow["inlet"]),
        "downstream_solver_residual": pred_flow["solver_residual"],
        "downstream_flux_imbalance": pred_flow["balance"],
        "test_k_points": int(len(ki)), "test_p_points": int(len(pi)),
    }


def evaluate(model, case, observations):
    started = time.perf_counter()
    with torch.no_grad():
        log_k = model.permeability(tensor(case.xy)).numpy()
        pressure = model.pressure(tensor(case.xy)).numpy()
    prediction_seconds = time.perf_counter() - started
    residual = darcy_residual(model, tensor(observations.evaluation_collocation)).detach().numpy()
    axis = (np.arange(64) + 0.5) / 64
    left = tensor(np.column_stack((np.zeros_like(axis), axis)))
    right = tensor(np.column_stack((np.ones_like(axis), axis)))
    _, _, _, ql = flux(model, left)
    _, _, _, qr = flux(model, right)
    inlet, outlet = float(ql[:, 0].detach().mean()), float(qr[:, 0].detach().mean())
    boundary_xy = tensor(np.concatenate((np.column_stack((axis, np.zeros_like(axis))),
                                        np.column_stack((axis, np.ones_like(axis))))))
    _, _, _, qb = flux(model, boundary_xy)
    boundary_flux = qb[:, 1].detach().numpy()
    metrics = reconstruction_metrics(case, observations, log_k, pressure)
    metrics.update({
        "pde_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "neumann_flux_rmse": float(np.sqrt(np.mean(boundary_flux**2))),
        "network_inlet_flux": inlet, "network_outlet_flux": outlet,
        "network_flux_imbalance": abs(inlet - outlet) / max(abs(inlet), 1e-12),
        "network_flux_relative_error": abs(inlet - case.metadata["reference_inlet_fine"]) / abs(case.metadata["reference_inlet_fine"]),
        "prediction_seconds": prediction_seconds,
    })
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite evaluation metric")
    arrays = {"log_k": log_k, "pressure": pressure, "physics_residual": residual,
              "neumann_flux": boundary_flux, "network_boundary_qx": np.array([inlet, outlet]),
              "feature_mask": model.permeability.feature_mask.detach().numpy()}
    return metrics, arrays
