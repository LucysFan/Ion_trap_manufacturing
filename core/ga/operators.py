from __future__ import annotations

import numpy as np

from core.ga.candidates import Candidate, Evaluation, N_CTRL, N_FEATURES, TOPOLOGIES
from core.ga.nsga2 import crowding_distance, non_dominated_sort


def repair_candidate(candidate: Candidate, cfg) -> Candidate:
    candidate.inner_m = np.clip(candidate.inner_m, 5e-6, 180e-6)
    candidate.outer_m = np.clip(candidate.outer_m, 30e-6, 360e-6)
    candidate.outer_m = np.maximum(
        candidate.outer_m,
        candidate.inner_m + cfg.min_rf_width_m,
    )

    candidate.inner_m[-1] = cfg.base_inner_m
    candidate.outer_m[-1] = cfg.base_outer_m

    candidate.topology = int(
        np.clip(candidate.topology, 0, len(TOPOLOGIES) - 1)
    )
    candidate.topology_size_m = float(
        np.clip(candidate.topology_size_m, 12e-6, 170e-6)
    )
    candidate.topology_width_m = float(
        np.clip(candidate.topology_width_m, cfg.min_feature_m, 70e-6)
    )

    candidate.feature_kind = np.clip(
        candidate.feature_kind,
        0,
        2,
    ).astype(np.int8)
    candidate.feature_operation = np.where(
        candidate.feature_operation >= 0,
        1,
        -1,
    ).astype(np.int8)
    candidate.feature_radius_m = np.clip(
        candidate.feature_radius_m,
        20e-6,
        320e-6,
    )
    candidate.feature_theta_rad = np.mod(
        candidate.feature_theta_rad,
        np.pi / 4.0,
    )
    candidate.feature_p1_m = np.clip(
        candidate.feature_p1_m,
        cfg.min_feature_m,
        100e-6,
    )
    candidate.feature_p2_m = np.clip(
        candidate.feature_p2_m,
        cfg.min_feature_m,
        100e-6,
    )
    candidate.feature_angle_rad = np.mod(
        candidate.feature_angle_rad + np.pi,
        2.0 * np.pi,
    ) - np.pi
    return candidate


def crossover(
    first: Candidate,
    second: Candidate,
    cfg,
    rng: np.random.Generator,
) -> Candidate:
    child = first.copy()

    alpha = rng.uniform(-0.20, 1.20, N_CTRL)
    child.inner_m = (
        alpha * first.inner_m + (1.0 - alpha) * second.inner_m
    )

    alpha = rng.uniform(-0.20, 1.20, N_CTRL)
    child.outer_m = (
        alpha * first.outer_m + (1.0 - alpha) * second.outer_m
    )

    if rng.random() < 0.5:
        child.topology = second.topology

    blend = float(rng.uniform(-0.20, 1.20))
    child.topology_size_m = (
        blend * first.topology_size_m
        + (1.0 - blend) * second.topology_size_m
    )

    blend = float(rng.uniform(-0.20, 1.20))
    child.topology_width_m = (
        blend * first.topology_width_m
        + (1.0 - blend) * second.topology_width_m
    )

    for index in range(N_FEATURES):
        if rng.random() < 0.5:
            child.feature_kind[index] = second.feature_kind[index]
            child.feature_operation[index] = second.feature_operation[index]

        for attribute in (
            "feature_radius_m",
            "feature_theta_rad",
            "feature_p1_m",
            "feature_p2_m",
            "feature_angle_rad",
        ):
            first_array = getattr(first, attribute)
            second_array = getattr(second, attribute)
            weight = float(rng.uniform(-0.20, 1.20))
            getattr(child, attribute)[index] = (
                weight * first_array[index]
                + (1.0 - weight) * second_array[index]
            )

    return repair_candidate(child, cfg)


def mutate(
    candidate: Candidate,
    cfg,
    rng: np.random.Generator,
    progress: float,
) -> Candidate:
    scale = 1.0 - 0.75 * progress

    for array, sigma in (
        (candidate.inner_m, 18e-6),
        (candidate.outer_m, 24e-6),
    ):
        selected = rng.random(array.shape) < cfg.mutation_probability
        array[selected] += rng.normal(
            0.0,
            sigma * scale,
            np.sum(selected),
        )

    if rng.random() < cfg.topology_mutation_probability:
        candidate.topology = int(rng.integers(0, len(TOPOLOGIES)))

    if rng.random() < cfg.mutation_probability:
        candidate.topology_size_m += float(
            rng.normal(0.0, 18e-6 * scale)
        )

    if rng.random() < cfg.mutation_probability:
        candidate.topology_width_m += float(
            rng.normal(0.0, 10e-6 * scale)
        )

    for index in range(N_FEATURES):
        if rng.random() < cfg.topology_mutation_probability:
            candidate.feature_kind[index] = int(rng.integers(0, 3))
        if rng.random() < cfg.topology_mutation_probability:
            candidate.feature_operation[index] *= -1
        if rng.random() < cfg.mutation_probability:
            candidate.feature_radius_m[index] += float(
                rng.normal(0.0, 25e-6 * scale)
            )
        if rng.random() < cfg.mutation_probability:
            candidate.feature_theta_rad[index] += float(
                rng.normal(0.0, 0.12 * scale)
            )
        if rng.random() < cfg.mutation_probability:
            candidate.feature_p1_m[index] += float(
                rng.normal(0.0, 12e-6 * scale)
            )
        if rng.random() < cfg.mutation_probability:
            candidate.feature_p2_m[index] += float(
                rng.normal(0.0, 12e-6 * scale)
            )
        if rng.random() < cfg.mutation_probability:
            candidate.feature_angle_rad[index] += float(
                rng.normal(0.0, 0.25 * scale)
            )

    return repair_candidate(candidate, cfg)


def select_survivors(
    population: list[Candidate],
    evaluations: list[Evaluation],
    count: int,
) -> tuple[list[Candidate], list[Evaluation]]:
    objectives = np.stack(
        [evaluation.objectives for evaluation in evaluations]
    )
    fronts = non_dominated_sort(objectives)
    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= count:
            selected.extend(front)
        else:
            distances = crowding_distance(front, objectives)
            ordered = sorted(
                front,
                key=lambda index: distances[index],
                reverse=True,
            )
            selected.extend(ordered[: count - len(selected)])
            break

    return (
        [population[index] for index in selected],
        [evaluations[index] for index in selected],
    )


def choose_compromise_candidate(
    evaluations: list[Evaluation],
) -> int:
    objective_matrix = np.stack(
        [evaluation.objectives for evaluation in evaluations]
    )
    fronts = non_dominated_sort(objective_matrix)

    candidate_indices = fronts[0] if fronts else list(
        range(len(evaluations))
    )

    valid_indices = [
        index
        for index in candidate_indices
        if bool(
            evaluations[index].metrics.get(
                "valid",
                not evaluations[index].metrics.get("invalid", True),
            )
        )
    ]

    if valid_indices:
        candidate_indices = valid_indices

    selected_objectives = objective_matrix[candidate_indices]
    minimum = np.nanmin(selected_objectives, axis=0)
    maximum = np.nanmax(selected_objectives, axis=0)
    span = maximum - minimum
    span = np.where(span > 1e-12, span, 1.0)

    normalized = (selected_objectives - minimum) / span
    scores = np.nansum(normalized, axis=1)
    local_index = int(np.nanargmin(scores))

    return int(candidate_indices[local_index])


__all__ = [
    "choose_compromise_candidate",
    "crossover",
    "mutate",
    "repair_candidate",
    "select_survivors",
]