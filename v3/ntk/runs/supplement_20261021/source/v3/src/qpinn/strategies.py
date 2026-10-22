"""Classical training aids; neither RBA nor Monte Carlo screening is a quantum algorithm."""

from dataclasses import dataclass

import numpy as np
import torch

from dataset.cases import TrainingView
from metrics.physics import components, objective, validation_score


@dataclass(frozen=True)
class StrategyConfig:
    warmup: int = 20
    rba_decay: float = 0.9
    rba_gain: float = 4.0
    mc_every: int = 20
    mc_trials: int = 4
    mc_batch: int = 32
    mc_fraction: float = 0.25
    mc_threshold: float = 0.01
    min_features: int = 2

    def validate(self, features):
        if self.warmup < 0 or not 0 <= self.rba_decay < 1 or self.rba_gain < 0:
            raise ValueError("invalid RBA settings")
        if self.mc_every < 1 or self.mc_trials < 2 or self.mc_batch < 2:
            raise ValueError("MC needs a positive period, >=2 trials and >=2 batch points")
        if not 0 < self.mc_fraction <= 1 or not 1 <= self.min_features <= features:
            raise ValueError("invalid feature screening limits")
        if not np.isfinite(self.mc_threshold) or self.mc_threshold < 0:
            raise ValueError("mc_threshold must be finite and nonnegative")


class ResidualAttention:
    """Bounded EMA residual attention with a unit floor and mean-normalized loss."""

    def __init__(self, count, config):
        self.config = config
        self.scores = torch.zeros(count, dtype=torch.float64)

    @torch.no_grad()
    def update(self, residual):
        magnitude = residual.detach().abs()
        normalized = magnitude / magnitude.max().clamp_min(1e-12)
        self.scores.mul_(self.config.rba_decay).add_(normalized, alpha=1 - self.config.rba_decay)
        return self.weights()

    def weights(self):
        return 1 + self.config.rba_gain * self.scores


def subset(training, rng, count):
    ki = rng.choice(len(training.log_k), min(count, len(training.log_k)), replace=False)
    pi = rng.choice(len(training.pressure), min(count, len(training.pressure)), replace=False)
    ci = rng.choice(len(training.collocation), min(count, len(training.collocation)), replace=False)
    bi = rng.choice(len(training.boundary_xy), min(count, len(training.boundary_xy)), replace=False)
    batch = TrainingView(training.k_xy[ki], training.log_k[ki], training.p_xy[pi],
                         training.pressure[pi], training.collocation[ci], training.boundary_xy[bi])
    return batch, ci


def screen_features(model, training, validation, weights, point_weights, config, rng):
    """Paired ablation trials; only validation observations may accept a proposed mask."""
    mask = model.permeability.feature_mask.detach().clone()
    active = torch.nonzero(mask > 0, as_tuple=False).flatten().tolist()
    if len(active) <= config.min_features:
        return {"accepted": False, "reason": "minimum_features", "active": active}
    deltas = []
    for _ in range(config.mc_trials):
        batch, indices = subset(training, rng, config.mc_batch)
        frozen_weights = point_weights[indices].detach().clone()
        baseline = float(objective(components(model, batch, mask), weights, frozen_weights)[0].detach())
        trial = []
        for feature in active:
            candidate = mask.clone()
            candidate[feature] = 0
            loss = float(objective(components(model, batch, candidate), weights, frozen_weights)[0].detach())
            trial.append((loss - baseline) / max(abs(baseline), 1e-12))
        deltas.append(trial)
    values = np.asarray(deltas)
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite Monte Carlo utility")
    means = values.mean(0)
    # A conservative heuristic, not a formal independent-sample confidence interval.
    upper = means + 2 * values.std(0, ddof=1) / np.sqrt(config.mc_trials)
    count = min(max(1, int(len(active) * config.mc_fraction)), len(active) - config.min_features)
    candidates = [i for i in np.argsort(means)[:count] if upper[i] <= config.mc_threshold]
    proposed = mask.clone()
    removed = [active[i] for i in candidates]
    proposed[removed] = 0
    before = validation_score(model, validation, weights, mask)
    after = validation_score(model, validation, weights, proposed)
    accepted = bool(removed and np.isfinite(after) and after <= before)
    if accepted:
        model.permeability.feature_mask.copy_(proposed)
    return {
        "accepted": accepted, "candidate_features": removed, "active_before": active,
        "relative_loss_changes": values.tolist(), "mean": means.tolist(),
        "upper_heuristic": upper.tolist(), "threshold": config.mc_threshold,
        "validation_before": before, "validation_after": after,
        "active_after": torch.nonzero(model.permeability.feature_mask > 0).flatten().tolist(),
    }
