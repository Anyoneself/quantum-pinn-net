from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DarcyLossTerms:
    """Dimensionless steady Darcy-flow diagnostics for a predicted field."""

    continuity: torch.Tensor
    flux_balance: torch.Tensor
    pressure: torch.Tensor


def normalized_log_perm_to_positive_permeability(log_perm: torch.Tensor) -> torch.Tensor:
    """Map normalized log-permeability to a strictly positive relative field.

    The dataset does not retain a permeability unit or the affine transform
    used before normalization.  Consequently this function returns relative,
    rather than dimensional, permeability.  Subtracting the sample mean makes
    the PDE residual constrain spatial structure without duplicating the
    separate Darcy/geomechanics mean constraint.
    """

    centered = log_perm - log_perm.mean(dim=(-2, -1), keepdim=True)
    return torch.exp(torch.clamp(centered, min=-6.0, max=6.0))


def _harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 2.0 * a * b / (a + b + eps)


def solve_dimensionless_pressure(
    permeability: torch.Tensor,
    iterations: int = 16,
    relaxation: float = 0.8,
) -> torch.Tensor:
    """Solve div(k grad(p))=0 with left/right Dirichlet and top/bottom no-flow BCs.

    The domain is the unit square, p=1 at x=0, p=0 at x=1, and
    dp/dn=0 on y=0 and y=1.  Weighted Jacobi iterations keep the operation
    differentiable with respect to ``permeability``.
    """

    if permeability.ndim != 4 or permeability.shape[1] != 1:
        raise ValueError("permeability must have shape (batch, 1, height, width)")
    if permeability.shape[-2] < 3 or permeability.shape[-1] < 3:
        raise ValueError("Darcy pressure solve requires at least a 3 x 3 grid")

    batch, _, height, width = permeability.shape
    x = torch.linspace(1.0, 0.0, width, device=permeability.device, dtype=permeability.dtype)
    pressure = x.view(1, 1, 1, width).expand(batch, 1, height, width)

    k_center = permeability[:, :, 1:-1, 1:-1]
    t_west = _harmonic_mean(k_center, permeability[:, :, 1:-1, :-2])
    t_east = _harmonic_mean(k_center, permeability[:, :, 1:-1, 2:])
    t_north = _harmonic_mean(k_center, permeability[:, :, :-2, 1:-1])
    t_south = _harmonic_mean(k_center, permeability[:, :, 2:, 1:-1])
    denominator = t_west + t_east + t_north + t_south + 1e-8
    omega = float(max(0.0, min(relaxation, 1.0)))

    for _ in range(max(int(iterations), 1)):
        jacobi = (
            t_west * pressure[:, :, 1:-1, :-2]
            + t_east * pressure[:, :, 1:-1, 2:]
            + t_north * pressure[:, :, :-2, 1:-1]
            + t_south * pressure[:, :, 2:, 1:-1]
        ) / denominator
        interior = (1.0 - omega) * pressure[:, :, 1:-1, 1:-1] + omega * jacobi
        middle = torch.cat(
            [
                torch.ones_like(pressure[:, :, 1:-1, :1]),
                interior,
                torch.zeros_like(pressure[:, :, 1:-1, -1:]),
            ],
            dim=-1,
        )
        pressure = torch.cat([middle[:, :, :1, :], middle, middle[:, :, -1:, :]], dim=-2)

    return pressure


def darcy_flow_loss(
    predicted_log_perm: torch.Tensor,
    iterations: int = 16,
) -> DarcyLossTerms:
    """Return normalized mass-conservation and boundary-flux losses.

    Face fluxes use Darcy's law q=-k grad(p) with harmonic transmissibility.
    Residuals are normalized by the mean flux magnitude so that their scale is
    stable for normalized synthetic fields.
    """

    permeability = normalized_log_perm_to_positive_permeability(predicted_log_perm)
    pressure = solve_dimensionless_pressure(permeability, iterations=iterations)
    height, width = permeability.shape[-2:]
    dx = 1.0 / max(width - 1, 1)
    dy = 1.0 / max(height - 1, 1)

    t_x = _harmonic_mean(permeability[:, :, :, :-1], permeability[:, :, :, 1:])
    t_y = _harmonic_mean(permeability[:, :, :-1, :], permeability[:, :, 1:, :])
    q_x = -t_x * (pressure[:, :, :, 1:] - pressure[:, :, :, :-1]) / dx
    q_y = -t_y * (pressure[:, :, 1:, :] - pressure[:, :, :-1, :]) / dy

    divergence = (
        (q_x[:, :, 1:-1, 1:] - q_x[:, :, 1:-1, :-1]) / dx
        + (q_y[:, :, 1:, 1:-1] - q_y[:, :, :-1, 1:-1]) / dy
    )
    flux_scale = q_x.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    continuity = (divergence / flux_scale).square().mean()

    inlet = q_x[:, :, :, 0].mean(dim=-1)
    outlet = q_x[:, :, :, -1].mean(dim=-1)
    boundary_scale = 0.5 * (inlet.abs() + outlet.abs()) + 1e-6
    flux_balance = ((inlet - outlet) / boundary_scale).square().mean()
    return DarcyLossTerms(continuity=continuity, flux_balance=flux_balance, pressure=pressure)
