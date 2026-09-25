from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.optimization.constraints import repair_genome
from core.optimization.genome import (
    FEATURE_NONE,
    FEATURE_OP_ADD_RF,
    FEATURE_OP_REMOVE_RF,
    Genome,
    N_CTRL,
    N_FEATURES,
    TOPOLOGY_NAMES,
)
from core.optimization.junction_evaluator import JunctionEvaluation


@dataclass(frozen=True)
class Nsga2Config:
    crossover_probability: float = 0.9
    mutation_probability: float = 0.20
    mutation_eta: float = 20.0
    topology_mutation_probability: float = 0.08
    feature_toggle_probability: float = 0.10
    feature_parameter_mutation_probability: float = 0.18
    tournament_size: int = 2

    min_rf_width_m: float = 18e-6
    fix_terminal_controls: bool = True
    base_inner_m: float = 37.35e-6
    base_outer_m: float = 216.45e-6


def objective_matrix(
    evaluations: list[JunctionEvaluation],
) -> np.ndarray:
    return np.stack(
        [evaluation.objectives for evaluation in evaluations],
        axis=0,
    )


def dominates(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float = 1e-12,
) -> bool:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return bool(
        np.all(left <= right + atol)
        and np.any(left < right - atol)
    )


def nondominated_sort(
    objectives: np.ndarray,
) -> list[list[int]]:
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("objectives must have shape (population, n_objectives)")

    count = objectives.shape[0]
    dominates_set: list[list[int]] = [[] for _ in range(count)]
    domination_count = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominates_set[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    level = 0
    while level < len(fronts) and fronts[level]:
        next_front: list[int] = []
        for p in fronts[level]:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        level += 1

    return fronts


def crowding_distance(
    objectives: np.ndarray,
    indices: list[int] | np.ndarray,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return np.zeros(0, dtype=float)

    selected = np.asarray(objectives, dtype=float)[indices]
    n_points, n_objectives = selected.shape
    distances = np.zeros(n_points, dtype=float)

    if n_points <= 2:
        distances[:] = np.inf
        return distances

    for objective_index in range(n_objectives):
        order = np.argsort(selected[:, objective_index])
        ordered = selected[order, objective_index]
        span = ordered[-1] - ordered[0]

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if span <= 1e-15:
            continue

        for rank in range(1, n_points - 1):
            distances[order[rank]] += (
                ordered[rank + 1] - ordered[rank - 1]
            ) / span

    return distances


def rank_and_crowding(
    objectives: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    objectives = np.asarray(objectives, dtype=float)
    fronts = nondominated_sort(objectives)

    rank = np.full(objectives.shape[0], -1, dtype=int)
    crowding = np.zeros(objectives.shape[0], dtype=float)

    for front_rank, front in enumerate(fronts):
        front_indices = np.asarray(front, dtype=int)
        front_distances = crowding_distance(objectives, front_indices)
        rank[front_indices] = front_rank
        crowding[front_indices] = front_distances

    return fronts, rank, crowding


def tournament_select(
    rank: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
    *,
    tournament_size: int = 2,
) -> int:
    if tournament_size < 2:
        raise ValueError("tournament_size must be at least 2")

    competitors = rng.integers(0, len(rank), size=tournament_size)
    best = int(competitors[0])

    for challenger in competitors[1:]:
        challenger = int(challenger)
        if rank[challenger] < rank[best]:
            best = challenger
        elif rank[challenger] == rank[best]:
            if crowding[challenger] > crowding[best]:
                best = challenger
            elif crowding[challenger] == crowding[best]:
                if rng.random() < 0.5:
                    best = challenger

    return best


def select_survivors(
    population: list[Genome],
    evaluations: list[JunctionEvaluation],
    count: int,
) -> tuple[list[Genome], list[JunctionEvaluation], np.ndarray]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(population) != len(evaluations):
        raise ValueError("population and evaluations must have the same length")

    objectives = objective_matrix(evaluations)
    fronts = nondominated_sort(objectives)

    selected_indices: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= count:
            selected_indices.extend(front)
            continue

        distances = crowding_distance(objectives, front)
        order = np.argsort(-distances)
        selected_indices.extend([front[i] for i in order[: count - len(selected_indices)]])
        break

    indices = np.asarray(selected_indices, dtype=int)
    survivors = [population[i] for i in indices]
    survivor_evaluations = [evaluations[i] for i in indices]
    return survivors, survivor_evaluations, indices


def simulated_binary_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    *,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent_a = np.asarray(parent_a, dtype=float)
    parent_b = np.asarray(parent_b, dtype=float)

    if parent_a.shape != parent_b.shape:
        raise ValueError("parent vectors must have the same shape")

    child_a = parent_a.copy()
    child_b = parent_b.copy()

    for i in range(parent_a.size):
        if rng.random() > 0.5:
            continue

        x1 = parent_a[i]
        x2 = parent_b[i]
        if abs(x1 - x2) <= 1e-18:
            continue

        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        child_a[i] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
        child_b[i] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)

    return child_a, child_b


def polynomial_mutation(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    eta: float,
    probability: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("values, lower, upper must have matching shapes")

    for i in range(values.size):
        if rng.random() >= probability:
            continue

        lo = lower[i]
        hi = upper[i]
        if hi <= lo:
            values[i] = lo
            continue

        x = np.clip(values[i], lo, hi)
        delta1 = (x - lo) / (hi - lo)
        delta2 = (hi - x) / (hi - lo)
        u = rng.random()
        mut_pow = 1.0 / (eta + 1.0)

        if u < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
            delta_q = 1.0 - val ** mut_pow

        x = x + delta_q * (hi - lo)
        values[i] = np.clip(x, lo, hi)

    return values


def _continuous_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate(
        [
            np.full(N_CTRL, Genome.INNER_MIN_M, dtype=float),
            np.full(N_CTRL, Genome.OUTER_MIN_M, dtype=float),
            np.asarray([0.0], dtype=float),
            np.asarray([Genome.TOPOLOGY_SIZE_MIN_M], dtype=float),
            np.asarray([Genome.TOPOLOGY_WIDTH_MIN_M], dtype=float),
            np.full(N_FEATURES, 0.0, dtype=float),
            np.full(N_FEATURES, -1.0, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_RADIUS_MIN_M, dtype=float),
            np.full(N_FEATURES, 0.0, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_P_MIN_M, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_P_MIN_M, dtype=float),
            np.full(N_FEATURES, -np.pi, dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(N_CTRL, Genome.INNER_MAX_M, dtype=float),
            np.full(N_CTRL, Genome.OUTER_MAX_M, dtype=float),
            np.asarray([float(len(TOPOLOGY_NAMES) - 1)], dtype=float),
            np.asarray([Genome.TOPOLOGY_SIZE_MAX_M], dtype=float),
            np.asarray([Genome.TOPOLOGY_WIDTH_MAX_M], dtype=float),
            np.full(N_FEATURES, 2.0, dtype=float),
            np.full(N_FEATURES, 1.0, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_RADIUS_MAX_M, dtype=float),
            np.full(N_FEATURES, np.pi / 4.0, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_P_MAX_M, dtype=float),
            np.full(N_FEATURES, Genome.FEATURE_P_MAX_M, dtype=float),
            np.full(N_FEATURES, np.pi, dtype=float),
        ]
    )
    return lower, upper


def mutate_genome(
    genome: Genome,
    rng: np.random.Generator,
    *,
    config: Nsga2Config,
) -> Genome:
    lower, upper = _continuous_bounds()
    vector = genome.to_vector()

    mutated = polynomial_mutation(
        vector,
        rng,
        eta=config.mutation_eta,
        probability=config.mutation_probability,
        lower=lower,
        upper=upper,
    )

    child = Genome.from_vector(mutated)

    if rng.random() < config.topology_mutation_probability:
        child.topology = int(rng.integers(0, len(TOPOLOGY_NAMES)))

    for index in range(N_FEATURES):
        if rng.random() < config.feature_toggle_probability:
            if child.feature_kind[index] == FEATURE_NONE:
                child.feature_kind[index] = int(rng.integers(1, 3))
                child.feature_operation[index] = int(
                    rng.choice(
                        np.asarray(
                            [FEATURE_OP_REMOVE_RF, FEATURE_OP_ADD_RF],
                            dtype=np.int8,
                        )
                    )
                )
            else:
                child.feature_kind[index] = FEATURE_NONE

        if child.feature_kind[index] != FEATURE_NONE:
            if rng.random() < config.feature_parameter_mutation_probability:
                child.feature_radius_m[index] = float(
                    rng.uniform(
                        Genome.FEATURE_RADIUS_MIN_M,
                        Genome.FEATURE_RADIUS_MAX_M,
                    )
                )
            if rng.random() < config.feature_parameter_mutation_probability:
                child.feature_theta_rad[index] = float(
                    rng.uniform(0.0, np.pi / 4.0)
                )
            if rng.random() < config.feature_parameter_mutation_probability:
                child.feature_p1_m[index] = float(
                    rng.uniform(
                        Genome.FEATURE_P_MIN_M,
                        Genome.FEATURE_P_MAX_M,
                    )
                )
            if rng.random() < config.feature_parameter_mutation_probability:
                child.feature_p2_m[index] = float(
                    rng.uniform(
                        Genome.FEATURE_P_MIN_M,
                        Genome.FEATURE_P_MAX_M,
                    )
                )
            if rng.random() < config.feature_parameter_mutation_probability:
                child.feature_angle_rad[index] = float(
                    rng.uniform(-np.pi, np.pi)
                )

    return repair_genome(
        child,
        min_rf_width_m=config.min_rf_width_m,
        fix_terminal_controls=config.fix_terminal_controls,
        base_inner_m=config.base_inner_m,
        base_outer_m=config.base_outer_m,
    )


def crossover_genomes(
    parent_a: Genome,
    parent_b: Genome,
    rng: np.random.Generator,
    *,
    config: Nsga2Config,
) -> tuple[Genome, Genome]:
    vector_a = parent_a.to_vector()
    vector_b = parent_b.to_vector()

    if rng.random() < config.crossover_probability:
        child_a_vec, child_b_vec = simulated_binary_crossover(
            vector_a,
            vector_b,
            rng,
            eta=config.mutation_eta,
        )
    else:
        child_a_vec = vector_a.copy()
        child_b_vec = vector_b.copy()

    child_a = Genome.from_vector(child_a_vec)
    child_b = Genome.from_vector(child_b_vec)

    child_a = repair_genome(
        child_a,
        min_rf_width_m=config.min_rf_width_m,
        fix_terminal_controls=config.fix_terminal_controls,
        base_inner_m=config.base_inner_m,
        base_outer_m=config.base_outer_m,
    )
    child_b = repair_genome(
        child_b,
        min_rf_width_m=config.min_rf_width_m,
        fix_terminal_controls=config.fix_terminal_controls,
        base_inner_m=config.base_inner_m,
        base_outer_m=config.base_outer_m,
    )

    return child_a, child_b


def make_offspring(
    population: list[Genome],
    evaluations: list[JunctionEvaluation],
    rng: np.random.Generator,
    *,
    offspring_size: int,
    config: Nsga2Config,
) -> list[Genome]:
    if len(population) == 0:
        raise ValueError("population must not be empty")
    if len(population) != len(evaluations):
        raise ValueError("population and evaluations must have the same length")
    if offspring_size <= 0:
        raise ValueError("offspring_size must be positive")

    objectives = objective_matrix(evaluations)
    _, rank, crowding = rank_and_crowding(objectives)

    offspring: list[Genome] = []
    while len(offspring) < offspring_size:
        i = tournament_select(
            rank,
            crowding,
            rng,
            tournament_size=config.tournament_size,
        )
        j = tournament_select(
            rank,
            crowding,
            rng,
            tournament_size=config.tournament_size,
        )

        parent_a = population[i]
        parent_b = population[j]

        child_a, child_b = crossover_genomes(
            parent_a,
            parent_b,
            rng,
            config=config,
        )

        child_a = mutate_genome(child_a, rng, config=config)
        offspring.append(child_a)

        if len(offspring) < offspring_size:
            child_b = mutate_genome(child_b, rng, config=config)
            offspring.append(child_b)

    return offspring


def step_nsga2(
    population: list[Genome],
    evaluations: list[JunctionEvaluation],
    offspring: list[Genome],
    offspring_evaluations: list[JunctionEvaluation],
    *,
    survivor_count: int,
) -> tuple[list[Genome], list[JunctionEvaluation], np.ndarray]:
    merged_population = list(population) + list(offspring)
    merged_evaluations = list(evaluations) + list(offspring_evaluations)

    return select_survivors(
        merged_population,
        merged_evaluations,
        survivor_count,
    )


def initialize_population(
    rng: np.random.Generator,
    *,
    size: int,
    initializer: Callable[[np.random.Generator], Genome],
    config: Nsga2Config,
) -> list[Genome]:
    if size <= 0:
        raise ValueError("size must be positive")

    population: list[Genome] = []
    for _ in range(size):
        genome = initializer(rng)
        genome = repair_genome(
            genome,
            min_rf_width_m=config.min_rf_width_m,
            fix_terminal_controls=config.fix_terminal_controls,
            base_inner_m=config.base_inner_m,
            base_outer_m=config.base_outer_m,
        )
        population.append(genome)

    return population


__all__ = [
    "Nsga2Config",
    "crossover_genomes",
    "crowding_distance",
    "dominates",
    "initialize_population",
    "make_offspring",
    "mutate_genome",
    "nondominated_sort",
    "objective_matrix",
    "rank_and_crowding",
    "select_survivors",
    "simulated_binary_crossover",
    "step_nsga2",
    "tournament_select",
]