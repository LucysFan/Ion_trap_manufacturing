from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    count = len(objectives)
    dominates_set = [[] for _ in range(count)]
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
    while fronts[level]:
        next_front: list[int] = []
        for p in fronts[level]:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        level += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(
    front: list[int],
    objectives: np.ndarray,
) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: np.inf for index in front}

    for objective_index in range(objectives.shape[1]):
        ordered = sorted(
            front,
            key=lambda index: objectives[index, objective_index],
        )
        distance[ordered[0]] = np.inf
        distance[ordered[-1]] = np.inf

        low = objectives[ordered[0], objective_index]
        high = objectives[ordered[-1], objective_index]
        if high <= low:
            continue

        for position in range(1, len(ordered) - 1):
            previous_value = objectives[
                ordered[position - 1], objective_index
            ]
            next_value = objectives[
                ordered[position + 1], objective_index
            ]
            distance[ordered[position]] += (
                next_value - previous_value
            ) / (high - low)

    return distance


def rank_and_crowding(
    objectives: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives)
    rank = np.empty(len(objectives), dtype=int)
    crowding = np.zeros(len(objectives), dtype=float)

    for front_rank, front in enumerate(fronts):
        distances = crowding_distance(front, objectives)
        for index in front:
            rank[index] = front_rank
            crowding[index] = distances[index]

    return fronts, rank, crowding


def tournament(
    rank: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> int:
    first, second = rng.integers(0, len(rank), size=2)
    if rank[first] < rank[second]:
        return int(first)
    if rank[second] < rank[first]:
        return int(second)
    return int(
        first if crowding[first] >= crowding[second] else second
    )


__all__ = [
    "crowding_distance",
    "dominates",
    "non_dominated_sort",
    "rank_and_crowding",
    "tournament",
]