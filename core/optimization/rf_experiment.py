from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.optimization.genome import Genome
from core.optimization.junction_evaluator import (
    EvaluatorConfig,
    JunctionEvaluation,
    JunctionEvaluator,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260923
    population_size: int = 32
    offspring_size: int = 32
    generations: int = 20
    checkpoint_every: int = 5
    plot_every: int = 1


@dataclass
class GenerationRecord:
    generation: int
    pareto_size: int
    valid_count: int
    population_size: int

    best_height_peak_m: float
    median_height_peak_m: float

    best_lateral_peak_m: float
    median_lateral_peak_m: float

    best_barrier_ev: float
    median_barrier_ev: float

    best_frequency_error_hz: float
    median_frequency_error_hz: float

    elapsed_s: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    evaluator_config: EvaluatorConfig
    final_population: list[Genome]
    final_evaluations: list[JunctionEvaluation]
    history: list[GenerationRecord]
    selected_index: int

    def selected_genome(self) -> Genome:
        return self.final_population[self.selected_index]

    def selected_evaluation(self) -> JunctionEvaluation:
        return self.final_evaluations[self.selected_index]


def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def metric_is_valid(metrics: dict[str, object]) -> bool:
    if "valid" in metrics:
        return bool(metrics["valid"])
    if "invalid" in metrics:
        return not bool(metrics["invalid"])
    return False


def objective_matrix(
    evaluations: list[JunctionEvaluation],
) -> np.ndarray:
    return np.stack(
        [evaluation.objectives for evaluation in evaluations],
        axis=0,
    )


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    count = objectives.shape[0]
    dominates: list[list[int]] = [[] for _ in range(count)]
    dominated_count = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(count):
        for j in range(count):
            if i == j:
                continue

            less_equal = np.all(objectives[i] <= objectives[j])
            strictly_less = np.any(objectives[i] < objectives[j])
            greater_equal = np.all(objectives[j] <= objectives[i])
            strictly_greater = np.any(objectives[j] < objectives[i])

            if less_equal and strictly_less:
                dominates[i].append(j)
            elif greater_equal and strictly_greater:
                dominated_count[i] += 1

        if dominated_count[i] == 0:
            fronts[0].append(i)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for i in fronts[front_index]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    return fronts


def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    n_points, n_obj = front_objectives.shape
    if n_points == 0:
        return np.zeros(0, dtype=float)
    if n_points <= 2:
        return np.full(n_points, np.inf, dtype=float)

    distance = np.zeros(n_points, dtype=float)

    for obj in range(n_obj):
        values = front_objectives[:, obj]
        order = np.argsort(values)
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf

        vmin = values[order[0]]
        vmax = values[order[-1]]
        span = vmax - vmin
        if span <= 1e-18:
            continue

        for k in range(1, n_points - 1):
            left = values[order[k - 1]]
            right = values[order[k + 1]]
            distance[order[k]] += (right - left) / span

    return distance


def ranks_and_crowding(
    objectives: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives)
    ranks = np.full(objectives.shape[0], fill_value=-1, dtype=int)
    crowding = np.zeros(objectives.shape[0], dtype=float)

    for rank, front in enumerate(fronts):
        if not front:
            continue
        ranks[np.asarray(front, dtype=int)] = rank
        crowding[np.asarray(front, dtype=int)] = crowding_distance(
            objectives[np.asarray(front, dtype=int)]
        )

    return fronts, ranks, crowding


def generation_statistics(
    generation: int,
    evaluations: list[JunctionEvaluation],
    elapsed_s: float,
) -> GenerationRecord:
    objectives = objective_matrix(evaluations)
    fronts = non_dominated_sort(objectives)

    valid_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.valid or metric_is_valid(evaluation.metrics)
    ]

    def array_from_metric(key: str) -> np.ndarray:
        return np.asarray(
            [
                finite_or_nan(evaluation.metrics.get(key))
                for evaluation in valid_evaluations
            ],
            dtype=float,
        )

    if valid_evaluations:
        height_values = array_from_metric("height_peak_m")
        lateral_values = array_from_metric("lateral_peak_m")
        barrier_values = array_from_metric("barrier_ev")
        frequency_values = array_from_metric("frequency_error_hz")

        best_height = float(np.nanmin(height_values))
        median_height = float(np.nanmedian(height_values))

        best_lateral = float(np.nanmin(lateral_values))
        median_lateral = float(np.nanmedian(lateral_values))

        best_barrier = float(np.nanmin(barrier_values))
        median_barrier = float(np.nanmedian(barrier_values))

        best_frequency = float(np.nanmin(frequency_values))
        median_frequency = float(np.nanmedian(frequency_values))
    else:
        best_height = float("nan")
        median_height = float("nan")
        best_lateral = float("nan")
        median_lateral = float("nan")
        best_barrier = float("nan")
        median_barrier = float("nan")
        best_frequency = float("nan")
        median_frequency = float("nan")

    return GenerationRecord(
        generation=int(generation),
        pareto_size=int(len(fronts[0]) if fronts else 0),
        valid_count=int(len(valid_evaluations)),
        population_size=int(len(evaluations)),
        best_height_peak_m=best_height,
        median_height_peak_m=median_height,
        best_lateral_peak_m=best_lateral,
        median_lateral_peak_m=median_lateral,
        best_barrier_ev=best_barrier,
        median_barrier_ev=median_barrier,
        best_frequency_error_hz=best_frequency,
        median_frequency_error_hz=median_frequency,
        elapsed_s=float(elapsed_s),
    )


def choose_compromise_index(
    evaluations: list[JunctionEvaluation],
    *,
    weights: np.ndarray | None = None,
) -> int:
    objectives = objective_matrix(evaluations)
    fronts = non_dominated_sort(objectives)

    candidate_indices = fronts[0] if fronts else list(range(len(evaluations)))

    valid_indices = [
        index
        for index in candidate_indices
        if evaluations[index].valid or metric_is_valid(evaluations[index].metrics)
    ]
    if valid_indices:
        candidate_indices = valid_indices

    selected = objectives[np.asarray(candidate_indices, dtype=int)]

    minimum = np.nanmin(selected, axis=0)
    maximum = np.nanmax(selected, axis=0)
    span = maximum - minimum
    span = np.where(span > 1e-12, span, 1.0)

    normalized = (selected - minimum) / span

    if weights is None:
        weights = np.ones(selected.shape[1], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (selected.shape[1],):
            raise ValueError("weights shape does not match objective count")

    scores = np.nansum(normalized * weights[None, :], axis=1)
    local_index = int(np.nanargmin(scores))
    return int(candidate_indices[local_index])


def genome_record(
    genome: Genome,
    evaluation: JunctionEvaluation,
    rank: int,
) -> dict[str, object]:
    return {
        "rank": int(rank),
        "valid": bool(evaluation.valid),
        "failure_reason": evaluation.failure_reason,
        "topology": int(genome.topology),
        "topology_size_m": float(genome.topology_size_m),
        "topology_width_m": float(genome.topology_width_m),
        "inner_control_m": genome.inner_control_m.tolist(),
        "outer_control_m": genome.outer_control_m.tolist(),
        "feature_kind": genome.feature_kind.tolist(),
        "feature_operation": genome.feature_operation.tolist(),
        "feature_radius_m": genome.feature_radius_m.tolist(),
        "feature_theta_rad": genome.feature_theta_rad.tolist(),
        "feature_p1_m": genome.feature_p1_m.tolist(),
        "feature_p2_m": genome.feature_p2_m.tolist(),
        "feature_angle_rad": genome.feature_angle_rad.tolist(),
        "objectives": evaluation.objectives.tolist(),
        **evaluation.metrics,
    }


def write_history_csv(
    path: Path,
    history: list[GenerationRecord],
) -> None:
    if not history:
        return

    rows = [record.as_dict() for record in history]
    fieldnames = list(rows[0].keys())

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = [str(row.get(field, "")) for field in fieldnames]
            handle.write(",".join(values) + "\n")

    temporary_path.replace(path)


def write_results(
    output_dir: Path,
    population: list[Genome],
    evaluations: list[JunctionEvaluation],
    generation: int,
    *,
    experiment_config: ExperimentConfig | None = None,
    evaluator_config: EvaluatorConfig | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    objectives = objective_matrix(evaluations)
    fronts, ranks, _ = ranks_and_crowding(objectives)

    records = [
        genome_record(genome, evaluation, int(ranks[i]))
        for i, (genome, evaluation) in enumerate(zip(population, evaluations))
    ]

    fieldnames = list(
        dict.fromkeys(key for record in records for key in record.keys())
    )

    final_population_csv = output_dir / "final_population.csv"
    with final_population_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for record in records:
            values = [
                str(record.get(field, "")).replace(",", ";")
                for field in fieldnames
            ]
            handle.write(",".join(values) + "\n")

    pareto_records = [records[index] for index in fronts[0]] if fronts else []
    pareto_csv = output_dir / "pareto_front.csv"
    with pareto_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for record in pareto_records:
            values = [
                str(record.get(field, "")).replace(",", ";")
                for field in fieldnames
            ]
            handle.write(",".join(values) + "\n")

    payload: dict[str, Any] = {
        "generation": int(generation),
        "population": records,
        "pareto_indices": fronts[0] if fronts else [],
    }

    if experiment_config is not None:
        payload["experiment_config"] = asdict(experiment_config)
    if evaluator_config is not None:
        payload["evaluator_config"] = asdict(evaluator_config)

    (output_dir / "checkpoint.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "GenerationRecord",
    "choose_compromise_index",
    "crowding_distance",
    "finite_or_nan",
    "generation_statistics",
    "genome_record",
    "metric_is_valid",
    "non_dominated_sort",
    "objective_matrix",
    "ranks_and_crowding",
    "write_history_csv",
    "write_results",
]