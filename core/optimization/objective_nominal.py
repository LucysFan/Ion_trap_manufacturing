from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.optimization.genome import Genome
from core.optimization.junction_evaluator import (
    JunctionEvaluation,
    JunctionEvaluator,
)


@dataclass(frozen=True)
class NominalWeights:
    height_peak: float = 1.0
    lateral_peak: float = 1.0
    barrier: float = 1.0
    frequency_error: float = 1.0
    anisotropy: float = 0.5
    geometry_penalty: float = 1.0


@dataclass(frozen=True)
class NominalScales:
    height_peak_m: float = 3e-6
    lateral_peak_m: float = 3e-6
    barrier_ev: float = 0.100
    frequency_error_hz: float = 0.5e6
    anisotropy_excess: float = 1.0
    geometry_penalty: float = 1.0


@dataclass
class NominalObjectiveResult:
    genome: Genome
    objectives: np.ndarray
    scalar_cost: float
    metrics: dict[str, float]
    valid: bool
    failure_reason: str | None
    parts: dict[str, float]


def _safe_float(value: float) -> float:
    value = float(value)
    if np.isnan(value):
        return float("inf")
    return value


def _component_costs(
    evaluation: JunctionEvaluation,
    *,
    weights: NominalWeights,
    scales: NominalScales,
) -> dict[str, float]:
    metrics = evaluation.metrics

    height_peak = _safe_float(metrics["height_peak_m"])
    lateral_peak = _safe_float(metrics["lateral_peak_m"])
    barrier_ev = max(_safe_float(metrics["barrier_ev"]), 0.0)
    frequency_error_hz = _safe_float(metrics["frequency_error_hz"])
    anisotropy_max = _safe_float(metrics["anisotropy_max"])
    geometry_penalty = _safe_float(metrics["geometry_penalty"])

    anisotropy_excess = max(anisotropy_max - 3.0, 0.0)

    parts = {
        "height_peak": weights.height_peak * (height_peak / scales.height_peak_m),
        "lateral_peak": weights.lateral_peak * (lateral_peak / scales.lateral_peak_m),
        "barrier": weights.barrier * (barrier_ev / scales.barrier_ev),
        "frequency_error": weights.frequency_error
        * (frequency_error_hz / scales.frequency_error_hz),
        "anisotropy": weights.anisotropy
        * (anisotropy_excess / scales.anisotropy_excess),
        "geometry_penalty": weights.geometry_penalty
        * (geometry_penalty / scales.geometry_penalty),
    }

    invalid = (not evaluation.valid) or any(not np.isfinite(v) for v in parts.values())
    if invalid:
        for key in parts:
            parts[key] = float(parts[key]) + 1000.0

    return {key: float(value) for key, value in parts.items()}


def objective_nominal(
    genome: Genome,
    evaluator: JunctionEvaluator,
    *,
    weights: NominalWeights | None = None,
    scales: NominalScales | None = None,
) -> NominalObjectiveResult:
    weights = weights or NominalWeights()
    scales = scales or NominalScales()

    evaluation = evaluator.evaluate(genome)
    parts = _component_costs(
        evaluation,
        weights=weights,
        scales=scales,
    )

    scalar_cost = float(sum(parts.values()))

    return NominalObjectiveResult(
        genome=evaluation.genome,
        objectives=np.asarray(evaluation.objectives, dtype=np.float64),
        scalar_cost=scalar_cost,
        metrics=dict(evaluation.metrics),
        valid=bool(evaluation.valid),
        failure_reason=evaluation.failure_reason,
        parts=parts,
    )


def objective_nominal_population(
    genomes: Iterable[Genome],
    evaluator: JunctionEvaluator,
    *,
    weights: NominalWeights | None = None,
    scales: NominalScales | None = None,
) -> list[NominalObjectiveResult]:
    weights = weights or NominalWeights()
    scales = scales or NominalScales()

    evaluations = evaluator.evaluate_population(list(genomes))
    results: list[NominalObjectiveResult] = []

    for evaluation in evaluations:
        parts = _component_costs(
            evaluation,
            weights=weights,
            scales=scales,
        )
        scalar_cost = float(sum(parts.values()))

        results.append(
            NominalObjectiveResult(
                genome=evaluation.genome,
                objectives=np.asarray(evaluation.objectives, dtype=np.float64),
                scalar_cost=scalar_cost,
                metrics=dict(evaluation.metrics),
                valid=bool(evaluation.valid),
                failure_reason=evaluation.failure_reason,
                parts=parts,
            )
        )

    return results


def scalarize_nominal_objectives(
    objectives: np.ndarray,
    *,
    objective_weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]

    if values.ndim != 2:
        raise ValueError("objectives must be a 1D or 2D array")

    n_obj = values.shape[1]
    if objective_weights is None:
        weights = np.ones(n_obj, dtype=np.float64)
    else:
        weights = np.asarray(objective_weights, dtype=np.float64).reshape(-1)
        if weights.size != n_obj:
            raise ValueError(
                f"objective_weights must have length {n_obj}, got {weights.size}"
            )

    return values @ weights


def best_nominal_result(
    genomes: Iterable[Genome],
    evaluator: JunctionEvaluator,
    *,
    weights: NominalWeights | None = None,
    scales: NominalScales | None = None,
    objective_weights: np.ndarray | None = None,
) -> NominalObjectiveResult:
    results = objective_nominal_population(
        genomes,
        evaluator,
        weights=weights,
        scales=scales,
    )
    if not results:
        raise ValueError("No genomes provided")

    objective_matrix = np.stack([result.objectives for result in results], axis=0)
    scores = scalarize_nominal_objectives(
        objective_matrix,
        objective_weights=objective_weights,
    )
    best_index = int(np.argmin(scores))
    return results[best_index]