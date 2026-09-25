from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from core.analysis.barrier import BarrierMetrics
from core.analysis.path_metrics import PathMetrics
from core.analysis.validation import ValidationReport
from core.ga.evolution import Evaluation


def geometry_penalty(candidate, cfg) -> float:
    rf_width = candidate.outer_m - candidate.inner_m
    width_violation = np.maximum(cfg.min_rf_width_m - rf_width, 0.0)

    inner_curvature = np.diff(candidate.inner_m, n=2)
    outer_curvature = np.diff(candidate.outer_m, n=2)

    curvature = np.sqrt(
        np.mean((inner_curvature / 20e-6) ** 2)
        + np.mean((outer_curvature / 25e-6) ** 2)
    )

    return float(
        np.sum((width_violation / cfg.min_rf_width_m) ** 2)
        + 0.08 * curvature
    )


def _safe_frequency_metrics(
    frequencies_low_hz: np.ndarray,
    frequencies_high_hz: np.ndarray,
    *,
    target_frequency_hz: float,
) -> tuple[float, float, float, float]:
    if len(frequencies_low_hz) == 0 or len(frequencies_high_hz) == 0:
        return np.inf, np.inf, np.inf, np.inf

    if not (
        np.all(np.isfinite(frequencies_low_hz))
        and np.all(np.isfinite(frequencies_high_hz))
    ):
        return np.inf, np.inf, np.inf, np.inf

    transverse_mean = 0.5 * (
        np.asarray(frequencies_low_hz, dtype=float)
        + np.asarray(frequencies_high_hz, dtype=float)
    )

    frequency_error_hz = float(
        np.max(np.abs(transverse_mean - target_frequency_hz))
    )
    anisotropy_max = float(
        np.max(
            np.asarray(frequencies_high_hz, dtype=float)
            / np.maximum(np.asarray(frequencies_low_hz, dtype=float), 1.0)
        )
    )
    frequency_min_hz = float(np.min(frequencies_low_hz))
    frequency_max_hz = float(np.max(frequencies_high_hz))
    return (
        frequency_error_hz,
        anisotropy_max,
        frequency_min_hz,
        frequency_max_hz,
    )


def build_evaluation(
    *,
    candidate,
    cfg,
    path_metrics: PathMetrics,
    barrier_metrics: BarrierMetrics,
    validation_report: ValidationReport,
    frequencies_low_hz: np.ndarray,
    frequencies_high_hz: np.ndarray,
) -> Evaluation:
    geometry_penalty_value = geometry_penalty(candidate, cfg)

    (
        frequency_error_hz,
        anisotropy_max,
        frequency_min_hz,
        frequency_max_hz,
    ) = _safe_frequency_metrics(
        frequencies_low_hz,
        frequencies_high_hz,
        target_frequency_hz=cfg.target_frequency_hz,
    )

    invalid = bool(
        (not validation_report.valid)
        or (not barrier_metrics.valid)
        or (not path_metrics.valid)
        or (not np.isfinite(path_metrics.height_peak_deviation_m))
        or (not np.isfinite(path_metrics.maximum_lateral_offset_m))
        or (not np.isfinite(barrier_metrics.barrier_height_ev))
        or (not np.isfinite(frequency_error_hz))
        or (path_metrics.maximum_transverse_residual_v_m > 5e3)
        or (frequency_min_hz < 0.15e6)
    )

    invalid_penalty = 1000.0 if invalid else 0.0

    objectives = np.asarray(
        [
            path_metrics.height_peak_deviation_m / 3e-6 + invalid_penalty,
            path_metrics.maximum_lateral_offset_m / 3e-6 + invalid_penalty,
            max(barrier_metrics.barrier_height_ev, 0.0) / 0.100 + invalid_penalty,
            frequency_error_hz / 0.5e6
            + 0.15 * max(anisotropy_max - 3.0, 0.0),
            geometry_penalty_value + invalid_penalty,
        ],
        dtype=np.float64,
    )

    metrics: dict[str, Any] = {
        "valid": not invalid,
        "invalid": invalid,
        "validation_ok": validation_report.valid,
        "validation_messages": tuple(validation_report.messages),
        "height_peak_m": float(path_metrics.height_peak_deviation_m),
        "height_rms_m": float(path_metrics.height_rms_deviation_m),
        "height_peak_to_peak_m": float(path_metrics.height_peak_to_peak_m),
        "arm_reference_height_m": float(path_metrics.arm_reference_height_m),
        "maximum_slope": float(path_metrics.maximum_slope),
        "maximum_curvature_m_inv": float(path_metrics.maximum_curvature_m_inv),
        "lateral_peak_m": float(path_metrics.maximum_lateral_offset_m),
        "field_residual_max_v_m": float(
            path_metrics.maximum_transverse_residual_v_m
        ),
        "full_field_norm_max_v_m": float(
            path_metrics.maximum_full_field_norm_v_m
        ),
        "barrier_ev": float(barrier_metrics.barrier_height_ev),
        "barrier_reference_ev": float(barrier_metrics.reference_energy_ev),
        "barrier_maximum_ev": float(barrier_metrics.maximum_energy_ev),
        "barrier_rms_above_reference_ev": float(
            barrier_metrics.rms_energy_above_reference_ev
        ),
        "frequency_error_hz": float(frequency_error_hz),
        "anisotropy_max": float(anisotropy_max),
        "frequency_min_hz": float(frequency_min_hz),
        "frequency_max_hz": float(frequency_max_hz),
        "geometry_penalty": float(geometry_penalty_value),
        "n_points": int(path_metrics.n_points),
        "n_converged": int(path_metrics.n_converged),
    }

    return Evaluation(objectives=objectives, metrics=metrics)