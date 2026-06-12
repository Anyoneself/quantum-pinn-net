from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CONDITION_COLUMNS = [
    "pressure_mpa",
    "pressure_gradient_mpa_per_m",
    "viscosity_cp",
    "porosity",
    "flow_rate_m3_day",
]


@dataclass
class PhysicalConfig:
    enabled: bool = True
    pressure_mpa: float = 30.0
    pressure_gradient_mpa_per_m: float = 0.012
    viscosity_cp: float = 1.0
    porosity: float = 0.22
    flow_rate_m3_day: float = 120.0
    reference_pressure_mpa: float = 30.0
    reference_pressure_gradient_mpa_per_m: float = 0.012
    reference_viscosity_cp: float = 1.0
    reference_porosity: float = 0.22
    reference_flow_rate_m3_day: float = 120.0
    compressibility_1_mpa: float = 0.015
    variability: float = 0.15


def physical_config_from_object(obj: Any) -> PhysicalConfig:
    return PhysicalConfig(
        enabled=bool(getattr(obj, "physics_enabled", True)),
        pressure_mpa=float(getattr(obj, "pressure_mpa", 30.0)),
        pressure_gradient_mpa_per_m=float(getattr(obj, "pressure_gradient_mpa_per_m", 0.012)),
        viscosity_cp=float(getattr(obj, "viscosity_cp", 1.0)),
        porosity=float(getattr(obj, "porosity", 0.22)),
        flow_rate_m3_day=float(getattr(obj, "flow_rate_m3_day", 120.0)),
        reference_pressure_mpa=float(getattr(obj, "reference_pressure_mpa", 30.0)),
        reference_pressure_gradient_mpa_per_m=float(getattr(obj, "reference_pressure_gradient_mpa_per_m", 0.012)),
        reference_viscosity_cp=float(getattr(obj, "reference_viscosity_cp", 1.0)),
        reference_porosity=float(getattr(obj, "reference_porosity", 0.22)),
        reference_flow_rate_m3_day=float(getattr(obj, "reference_flow_rate_m3_day", 120.0)),
        compressibility_1_mpa=float(getattr(obj, "compressibility_1_mpa", 0.015)),
        variability=float(getattr(obj, "physics_variability", 0.15)),
    )


def physical_config_summary(config: PhysicalConfig) -> dict[str, float | bool]:
    return asdict(config)


def sample_physical_conditions(sample_count: int, seed: int, config: PhysicalConfig) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919)
    spread = max(config.variability, 0.0)

    pressure = rng.normal(config.pressure_mpa, max(config.pressure_mpa * spread, 1e-6), sample_count)
    pressure_gradient = rng.normal(
        config.pressure_gradient_mpa_per_m,
        max(config.pressure_gradient_mpa_per_m * spread, 1e-8),
        sample_count,
    )
    viscosity = rng.normal(config.viscosity_cp, max(config.viscosity_cp * spread, 1e-6), sample_count)
    porosity = rng.normal(config.porosity, max(config.porosity * spread * 0.5, 1e-6), sample_count)
    flow_rate = rng.normal(config.flow_rate_m3_day, max(config.flow_rate_m3_day * spread, 1e-6), sample_count)

    conditions = np.stack(
        [
            np.clip(pressure, 1.0, None),
            np.clip(pressure_gradient, 1e-5, None),
            np.clip(viscosity, 0.05, None),
            np.clip(porosity, 0.03, 0.45),
            np.clip(flow_rate, 1.0, None),
        ],
        axis=1,
    )
    return conditions.astype(np.float32)


def expected_log_perm_index(conditions: np.ndarray, config: PhysicalConfig) -> np.ndarray:
    pressure = conditions[:, 0]
    pressure_gradient = conditions[:, 1]
    viscosity = conditions[:, 2]
    porosity = conditions[:, 3]
    flow_rate = conditions[:, 4]

    darcy_drive = (
        np.log(flow_rate / max(config.reference_flow_rate_m3_day, 1e-8))
        + np.log(viscosity / max(config.reference_viscosity_cp, 1e-8))
        - np.log(pressure_gradient / max(config.reference_pressure_gradient_mpa_per_m, 1e-8))
    )
    porosity_term = 2.8 * (porosity - config.reference_porosity) / max(config.reference_porosity, 1e-8)
    compaction_term = -config.compressibility_1_mpa * (pressure - config.reference_pressure_mpa)
    return (darcy_drive + porosity_term + compaction_term).astype(np.float32)


def apply_physical_constraints(
    log_fields: np.ndarray,
    conditions: np.ndarray,
    config: PhysicalConfig,
) -> np.ndarray:
    if not config.enabled:
        return log_fields.astype(np.float32)

    adjusted = log_fields.astype(np.float32).copy()
    expected = expected_log_perm_index(conditions, config)
    height, width = adjusted.shape[-2:]
    flow_ratio = conditions[:, 4] / max(config.reference_flow_rate_m3_day, 1e-8)
    pressure_ratio = conditions[:, 0] / max(config.reference_pressure_mpa, 1e-8)
    gradient_ratio = conditions[:, 1] / max(config.reference_pressure_gradient_mpa_per_m, 1e-8)
    porosity_ratio = conditions[:, 3] / max(config.reference_porosity, 1e-8)

    x_ramp = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
    y_ramp = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
    radial = np.sqrt(x_ramp**2 + y_ramp**2).astype(np.float32)
    radial = radial - float(radial.mean())

    for idx in range(adjusted.shape[0]):
        field = adjusted[idx]
        mean = float(field.mean())
        heterogeneity_scale = np.clip(1.0 + 0.85 * (porosity_ratio[idx] - 1.0), 0.55, 1.75)
        compaction_smoothing = np.clip(0.22 * (pressure_ratio[idx] - 1.0), -0.12, 0.35)
        directional_drive = np.clip(0.38 * np.tanh(flow_ratio[idx] - 1.0), -0.28, 0.28)
        gradient_drive = np.clip(0.32 * np.tanh(gradient_ratio[idx] - 1.0), -0.22, 0.22)
        pressure_bowl = np.clip(0.36 * np.tanh(pressure_ratio[idx] - 1.0), -0.18, 0.28)

        field = mean + (field - mean) * heterogeneity_scale
        field = (1.0 - compaction_smoothing) * field + compaction_smoothing * mean
        field = field + 0.70 * expected[idx]
        field = field + directional_drive * (0.7 * x_ramp + 0.3 * y_ramp)
        field = field - gradient_drive * (0.35 * x_ramp - 0.65 * y_ramp)
        field = field - pressure_bowl * radial
        adjusted[idx] = field

    return adjusted.astype(np.float32)


def summarize_physical_conditions(conditions: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col_idx, name in enumerate(CONDITION_COLUMNS):
        values = conditions[:, col_idx]
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_physical_reasonability(
    log_fields: np.ndarray,
    conditions: np.ndarray,
    config: PhysicalConfig,
    reference_min: float | None = None,
    reference_max: float | None = None,
) -> dict[str, float]:
    if conditions.size == 0:
        return {
            "physical_score": 0.0,
            "darcy_corr": 0.0,
            "pressure_corr": 0.0,
            "porosity_corr": 0.0,
            "range_score": 0.0,
        }

    mean_log_perm = log_fields.reshape(log_fields.shape[0], -1).mean(axis=1)
    expected = expected_log_perm_index(conditions, config)
    darcy_corr = _safe_corr(mean_log_perm, expected)
    pressure_corr = _safe_corr(mean_log_perm, conditions[:, 0])
    porosity_corr = _safe_corr(mean_log_perm, conditions[:, 3])

    if reference_min is None:
        reference_min = float(log_fields.min())
    if reference_max is None:
        reference_max = float(log_fields.max())
    out_of_range = np.logical_or(log_fields < reference_min, log_fields > reference_max)
    range_score = 1.0 - float(out_of_range.mean())

    darcy_score = np.clip((darcy_corr + 1.0) * 0.5, 0.0, 1.0)
    pressure_score = np.clip((-pressure_corr + 1.0) * 0.5, 0.0, 1.0)
    porosity_score = np.clip((porosity_corr + 1.0) * 0.5, 0.0, 1.0)
    physical_score = float(np.mean([darcy_score, pressure_score, porosity_score, range_score]))

    return {
        "physical_score": physical_score,
        "darcy_corr": darcy_corr,
        "pressure_corr": pressure_corr,
        "porosity_corr": porosity_corr,
        "range_score": float(range_score),
    }
