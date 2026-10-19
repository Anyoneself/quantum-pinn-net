"""Traceable dimensionless Darcy fields and observation-only training views."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class DataConfig:
    grid: int = 24
    refinement: int = 3
    modes: int = 4
    log_amplitude: float = 0.6
    k_train: int = 48
    k_validation: int = 24
    p_train: int = 48
    p_validation: int = 24
    collocation: int = 128
    boundary: int = 32
    noise: float = 0.0

    def validate(self):
        if self.grid < 4 or self.refinement < 1 or self.refinement % 2 != 1:
            raise ValueError("grid >= 4 and odd, positive refinement are required")
        if self.modes < 1 or not np.isfinite(self.log_amplitude) or self.log_amplitude < 0:
            raise ValueError("modes must be positive and log_amplitude finite and nonnegative")
        for counts in ((self.k_train, self.k_validation), (self.p_train, self.p_validation)):
            if min(counts) < 2 or sum(counts) >= self.grid**2:
                raise ValueError("each observation split needs >=2 points and a nonempty test set")
        if self.collocation < 2 or self.boundary < 2:
            raise ValueError("collocation and boundary counts must be >=2")
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError("noise must be finite and nonnegative")


def cell_coordinates(grid):
    axis = (np.arange(grid) + 0.5) / grid
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel()))


def field_parameters(seed, modes, amplitude):
    rng = np.random.default_rng(seed)
    frequencies = np.array([(i, j) for i in range(1, modes + 1) for j in range(1, modes + 1)])
    coefficients = rng.normal(size=len(frequencies)) / np.sum(frequencies**2, axis=1)
    coefficients *= amplitude / max(np.sum(np.abs(coefficients)), 1e-15)
    return {
        "seed": int(seed),
        "frequencies": frequencies.tolist(),
        "coefficients": coefficients.tolist(),
        "phases": rng.uniform(0, 2 * np.pi, (len(frequencies), 2)).tolist(),
        "log_mean": float(rng.uniform(-0.15, 0.15)),
        "amplitude_bound": amplitude,
        "formula": "log(k/k_ref)=log_mean+sum(c*sin(2*pi*i*x+a)*sin(2*pi*j*y+b))",
        "normalization": "none; k_ref=1, mu=1, L_ref=1, delta_p_ref=1",
    }


def log_field(xy, parameters):
    frequencies = np.asarray(parameters["frequencies"])
    phases = np.asarray(parameters["phases"])
    angles = 2 * np.pi * xy[:, None, :] * frequencies[None, :, :] + phases[None, :, :]
    return parameters["log_mean"] + np.prod(np.sin(angles), axis=2) @ np.asarray(
        parameters["coefficients"]
    )


def solve_darcy(permeability):
    """Independent cell-centred FVM, p(left)=1, p(right)=0, no top/bottom flux."""
    k = np.asarray(permeability, dtype=np.float64)
    if k.ndim != 2 or min(k.shape) < 2 or not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("permeability must be a finite positive 2D array")
    h, w = k.shape
    ids = np.arange(h * w).reshape(h, w)
    gx = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]) * w / h
    gy = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) * h / w
    a = np.concatenate((ids[:, :-1].ravel(), ids[:-1].ravel()))
    b = np.concatenate((ids[:, 1:].ravel(), ids[1:].ravel()))
    g = np.concatenate((gx.ravel(), gy.ravel()))
    diagonal = np.bincount(np.concatenate((a, b)), weights=np.tile(g, 2), minlength=h * w)
    left, right = ids[:, 0], ids[:, -1]
    gl, gr = 2 * k[:, 0] * w / h, 2 * k[:, -1] * w / h
    diagonal[left] += gl
    diagonal[right] += gr
    matrix = coo_matrix(
        (np.concatenate((-g, -g, diagonal)),
         (np.concatenate((a, b, ids.ravel())), np.concatenate((b, a, ids.ravel())))),
        shape=(h * w, h * w),
    ).tocsr()
    rhs = np.zeros(h * w)
    rhs[left] = gl
    p = spsolve(matrix, rhs)
    inlet, outlet = float(np.sum(gl * (1 - p[left]))), float(np.sum(gr * p[right]))
    return {
        "pressure": p.reshape(h, w),
        "inlet": inlet,
        "outlet": outlet,
        "balance": abs(inlet - outlet) / max(abs(inlet), 1e-15),
        "solver_residual": float(np.max(np.abs(matrix @ p - rhs))),
    }


@dataclass
class ReferenceCase:
    xy: np.ndarray
    log_k: np.ndarray
    pressure: np.ndarray
    metadata: dict


def generate_case(config, seed):
    config.validate()
    parameters = field_parameters(seed, config.modes, config.log_amplitude)
    n, r = config.grid, config.refinement
    xy = cell_coordinates(n)
    log_k = log_field(xy, parameters)
    fine_k = np.exp(log_field(cell_coordinates(n * r), parameters)).reshape(n * r, n * r)
    fine = solve_darcy(fine_k)
    # Odd refinement puts each coarse cell centre exactly on a fine cell centre.
    pressure = fine["pressure"][r // 2::r, r // 2::r].ravel()
    coarse = solve_darcy(np.exp(log_k).reshape(n, n))
    metadata = {
        "data_config": asdict(config),
        "generator": parameters,
        "equation": "q=-k*grad(p), div(q)=0",
        "boundary_conditions": {"left_p": 1, "right_p": 0, "top_qn": 0, "bottom_qn": 0},
        "fine_grid": n * r,
        "reference_inlet_fine": fine["inlet"],
        "reference_inlet_coarse": coarse["inlet"],
        "fine_solver_residual": fine["solver_residual"],
        "fine_flux_balance": fine["balance"],
        "coarse_fine_pressure_rmse": float(np.sqrt(np.mean((coarse["pressure"].ravel() - pressure)**2))),
        "coarse_fine_flux_relative_difference": abs(coarse["inlet"] - fine["inlet"]) / abs(fine["inlet"]),
        "permeability_contrast": float(np.exp(log_k.max() - log_k.min())),
        "task": "per-field inverse reconstruction, not zero-shot unseen-field prediction",
    }
    return ReferenceCase(xy, log_k, pressure, metadata)


@dataclass
class TrainingView:
    k_xy: np.ndarray
    log_k: np.ndarray
    p_xy: np.ndarray
    pressure: np.ndarray
    collocation: np.ndarray
    boundary_xy: np.ndarray


@dataclass
class ValidationView:
    k_xy: np.ndarray
    log_k: np.ndarray
    p_xy: np.ndarray
    pressure: np.ndarray


@dataclass
class Observations:
    training: TrainingView
    validation: ValidationView
    k_train_indices: np.ndarray
    k_validation_indices: np.ndarray
    k_test_indices: np.ndarray
    p_train_indices: np.ndarray
    p_validation_indices: np.ndarray
    p_test_indices: np.ndarray
    evaluation_collocation: np.ndarray
    metadata: dict


def observe(case, config, seed):
    config.validate()
    rng = np.random.default_rng(seed)
    n = len(case.xy)
    k_order, p_order = rng.permutation(n), rng.permutation(n)
    kt = k_order[:config.k_train]
    kv = k_order[config.k_train:config.k_train + config.k_validation]
    ke = k_order[config.k_train + config.k_validation:]
    pt = p_order[:config.p_train]
    pv = p_order[config.p_train:config.p_train + config.p_validation]
    pe = p_order[config.p_train + config.p_validation:]
    k_scale = float(np.std(case.log_k[kt]))
    p_scale = float(np.std(case.pressure[pt]))
    k_values = case.log_k[np.concatenate((kt, kv))].copy()
    p_values = case.pressure[np.concatenate((pt, pv))].copy()
    k_values += rng.normal(0, config.noise * k_scale, len(k_values))
    p_values += rng.normal(0, config.noise * p_scale, len(p_values))
    collocation = rng.uniform(0, 1, (config.collocation, 2))
    x = rng.uniform(0, 1, config.boundary)
    boundary = np.column_stack((np.tile(x, 2), np.repeat([0.0, 1.0], len(x))))
    evaluation = rng.uniform(0, 1, (max(128, config.collocation), 2))
    return Observations(
        TrainingView(case.xy[kt].copy(), k_values[:len(kt)],
                     case.xy[pt].copy(), p_values[:len(pt)], collocation, boundary),
        ValidationView(case.xy[kv].copy(), k_values[len(kt):],
                       case.xy[pv].copy(), p_values[len(pt):]),
        kt, kv, ke, pt, pv, pe, evaluation,
        {"seed": seed, "noise_fraction": config.noise,
         "noise_log_k_scale_training_only": k_scale, "noise_pressure_scale_training_only": p_scale},
    )
