from __future__ import annotations

import numpy as np


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


def nondominated_indices(
    objectives: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim != 2:
        raise ValueError("Objectives must have shape (population, n_objectives).")

    if valid is None:
        valid = np.all(np.isfinite(objectives), axis=1)
    else:
        valid = np.asarray(valid, dtype=bool) & np.all(
            np.isfinite(objectives),
            axis=1,
        )

    indices = np.flatnonzero(valid)
    keep = np.ones(indices.size, dtype=bool)

    for local_i, global_i in enumerate(indices):
        if not keep[local_i]:
            continue

        for local_j, global_j in enumerate(indices):
            if local_i == local_j or not keep[local_j]:
                continue

            if dominates(objectives[global_j], objectives[global_i]):
                keep[local_i] = False
                break

    return indices[keep]


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    n_points, n_objectives = objectives.shape
    distances = np.zeros(n_points, dtype=float)

    if n_points <= 2:
        distances[:] = np.inf
        return distances

    for objective_index in range(n_objectives):
        order = np.argsort(objectives[:, objective_index])
        ordered = objectives[order, objective_index]
        scale = ordered[-1] - ordered[0]

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if scale <= 1e-15:
            continue

        for rank in range(1, n_points - 1):
            distances[order[rank]] += (
                ordered[rank + 1] - ordered[rank - 1]
            ) / scale

    return distances