from __future__ import annotations

import numpy as np

from core.optimization.junction_evaluator import JunctionEvaluation


def coarse_objectives(evaluation: JunctionEvaluation) -> np.ndarray:
    return np.asarray(evaluation.objectives, dtype=np.float64)


def coarse_score(evaluation: JunctionEvaluation) -> float:
    return float(np.sum(np.asarray(evaluation.objectives, dtype=np.float64)))


def is_coarse_feasible(
    evaluation: JunctionEvaluation,
    *,
    max_height_peak_m: float = 10e-6,
    max_lateral_peak_m: float = 10e-6,
    max_frequency_error_hz: float = 0.75e6,
    max_geometry_penalty: float = 1.0,
) -> bool:
    metrics = evaluation.metrics
    if not evaluation.valid:
        return False

    return bool(
        metrics["height_peak_m"] <= max_height_peak_m
        and metrics["lateral_peak_m"] <= max_lateral_peak_m
        and metrics["frequency_error_hz"] <= max_frequency_error_hz
        and metrics["geometry_penalty"] <= max_geometry_penalty
    )