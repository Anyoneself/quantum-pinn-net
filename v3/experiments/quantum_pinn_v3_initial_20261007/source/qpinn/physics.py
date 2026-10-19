"""Strong-form Darcy residual, differentiated through k and p, including the QNode."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossWeights:
    permeability: float = 10.0
    pressure: float = 10.0
    physics: float = 1.0
    boundary: float = 1.0


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def gradient(value, xy):
    return torch.autograd.grad(value.sum(), xy, create_graph=True)[0]


def flux(model, coordinates, mask=None):
    xy = coordinates.detach().clone().requires_grad_(True)
    p = model.pressure(xy)
    log_k = model.permeability(xy, mask)
    q = -torch.exp(log_k)[:, None] * gradient(p, xy)
    return xy, p, log_k, q


def darcy_residual(model, coordinates, mask=None):
    xy, _, _, q = flux(model, coordinates, mask)
    return gradient(q[:, 0], xy)[:, 0] + gradient(q[:, 1], xy)[:, 1]


def components(model, training, mask=None):
    k_error = model.permeability(tensor(training.k_xy), mask) - tensor(training.log_k)
    p_error = model.pressure(tensor(training.p_xy)) - tensor(training.pressure)
    residual = darcy_residual(model, tensor(training.collocation), mask)
    _, _, _, boundary_q = flux(model, tensor(training.boundary_xy), mask)
    return {"permeability": k_error, "pressure": p_error,
            "physics": residual, "boundary": boundary_q[:, 1]}


def objective(errors, weights, point_weights):
    losses = {name: error.square().mean() for name, error in errors.items()}
    losses["physics"] = (point_weights * errors["physics"].square()).sum() / point_weights.sum()
    total = sum(getattr(weights, name) * value for name, value in losses.items())
    return total, losses


@torch.no_grad()
def validation_score(model, validation, weights, mask=None):
    k_error = model.permeability(tensor(validation.k_xy), mask) - tensor(validation.log_k)
    p_error = model.pressure(tensor(validation.p_xy)) - tensor(validation.pressure)
    return float(weights.permeability * k_error.square().mean()
                 + weights.pressure * p_error.square().mean())
