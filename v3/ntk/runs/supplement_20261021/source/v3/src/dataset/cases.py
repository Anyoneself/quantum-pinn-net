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
    p_train: int = 0
    p_validation: int = 0
    collocation: int = 128
    boundary: int = 32
    noise: float = 0.0
    field_generator: str = "gstools"
    log_variance: float = 0.4
    correlation_length: float = 0.25
    anisotropy: float = 0.6
    angle: float = 0.0
    log_mean: float = 0.0
    gstools_modes: int = 1000

    def validate(self):
        if self.grid < 4 or self.refinement < 1 or self.refinement % 2 != 1:
            raise ValueError("grid >= 4 and odd, positive refinement are required")
        if self.modes < 1 or not np.isfinite(self.log_amplitude) or self.log_amplitude < 0:
            raise ValueError("modes must be positive and log_amplitude finite and nonnegative")
        if min(self.k_train, self.k_validation) < 2 or self.k_train + self.k_validation >= self.grid**2:
            raise ValueError("permeability splits need >=2 points and a nonempty test set")
        if (self.p_train, self.p_validation) != (0, 0):
            if min(self.p_train, self.p_validation) < 2 or self.p_train + self.p_validation >= self.grid**2:
                raise ValueError("pressure counts must both be zero, or >=2 with a nonempty test set")
        if self.collocation < 2 or self.boundary < 2:
            raise ValueError("collocation and boundary counts must be >=2")
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError("noise must be finite and nonnegative")
        if self.field_generator not in ("gstools", "trigonometric"):
            raise ValueError("field_generator must be gstools or trigonometric")
        for name in ("log_variance", "correlation_length", "anisotropy"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.angle) or not np.isfinite(self.log_mean):
            raise ValueError("angle and log_mean must be finite")
        if self.gstools_modes < 1:
            raise ValueError("gstools_modes must be positive")


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


def gstools_log_field(xy, config, seed):
    """Evaluate one seeded GSTools realization without fitting label statistics."""
    import gstools as gs

    model = gs.Gaussian(dim=2, var=config.log_variance,
                        len_scale=config.correlation_length,
                        anis=config.anisotropy, angles=config.angle)
    field = gs.SRF(model, mean=config.log_mean, seed=seed,
                   generator="RandMeth", mode_no=config.gstools_modes)
    values = np.asarray(field((xy[:, 0], xy[:, 1]), mesh_type="unstructured", store=False))
    metadata = {
        "library": "gstools", "library_version": gs.__version__,
        "covariance": "Gaussian", "dimension": 2, "generator": "RandMeth",
        "seed": int(seed), "mode_no": config.gstools_modes,
        "log_variance": config.log_variance, "correlation_length": config.correlation_length,
        "anisotropy": config.anisotropy, "angle_radians": config.angle,
        "log_mean": config.log_mean,
        "formula": "log(k/k_ref)=GSTools.SRF(Gaussian covariance, mean=log_mean)",
        "normalization": "none; natural log, k_ref=1, mu=1, L_ref=1, delta_p_ref=1",
        "v2_arrays_reused": False,
    }
    return values, metadata


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
    n, r = config.grid, config.refinement
    xy = cell_coordinates(n)
    fine_xy = cell_coordinates(n * r)
    if config.field_generator == "gstools":
        fine_log_k, parameters = gstools_log_field(fine_xy, config, seed)
        # Both grids sample the same realization, never independently generated fields.
        log_k = fine_log_k.reshape(n*r, n*r)[r//2::r, r//2::r].ravel().copy()
    else:
        parameters = field_parameters(seed, config.modes, config.log_amplitude)
        parameters["library"] = "legacy-trigonometric"
        log_k = log_field(xy, parameters)
        fine_log_k = log_field(fine_xy, parameters)
    fine_k = np.exp(fine_log_k).reshape(n * r, n * r)
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
        "observed_variables": ["log_k", "pressure"] if config.p_train else ["log_k"],
        "reference_pressure_role": "observation_and_evaluation" if config.p_train else "evaluation_only",
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
    k_values = case.log_k[np.concatenate((kt, kv))].copy()
    # With no pressure observations, neither training nor validation reads its values.
    p_scale = float(np.std(case.pressure[pt])) if len(pt) else 0.0
    p_values = (case.pressure[np.concatenate((pt, pv))].copy()
                if len(pt) else np.empty(0, dtype=np.float64))
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
         "observed_variables": ["log_k", "pressure"] if len(pt) else ["log_k"],
         "pressure_reference_role": "observation_and_evaluation" if len(pt) else "evaluation_only",
         "noise_log_k_scale_training_only": k_scale, "noise_pressure_scale_training_only": p_scale},
    )
