from __future__ import annotations

import numpy as np

from core.ga.nsga2 import (
    crowding_distance,
    dominates,
    non_dominated_sort,
    rank_and_crowding,
    tournament,
)


def test_dominates_basic_case() -> None:
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))


def test_non_dominated_sort_returns_expected_fronts() -> None:
    objectives = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 5.0],
        ]
    )
    fronts = non_dominated_sort(objectives)
    assert set(fronts[0]) == {0, 1, 2, 3}
    assert fronts[1] == [4]


def test_crowding_distance_marks_edges_infinite() -> None:
    objectives = np.array([[1.0], [2.0], [3.0]])
    distance = crowding_distance([0, 1, 2], objectives)
    assert np.isinf(distance[0])
    assert np.isinf(distance[2])
    assert distance[1] >= 0.0


def test_rank_and_crowding_shapes() -> None:
    objectives = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    fronts, rank, crowding = rank_and_crowding(objectives)
    assert len(fronts) >= 1
    assert rank.shape == (3,)
    assert crowding.shape == (3,)


def test_tournament_prefers_lower_rank() -> None:
    rng = np.random.default_rng(123)
    rank = np.array([0, 1])
    crowding = np.array([0.0, 100.0])
    winners = {tournament(rank, crowding, rng) for _ in range(20)}
    assert 0 in winners