from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ga.candidates import Candidate, Evaluation
from core.ga.operators import (
    choose_compromise_candidate,
    mutate,
    repair_candidate,
    select_survivors,
)


@dataclass
class DummyConfig:
    min_rf_width_m: float = 18e-6
    base_inner_m: float = 37.35e-6
    base_outer_m: float = 216.45e-6
    min_feature_m: float = 6e-6
    mutation_probability: float = 0.22
    topology_mutation_probability: float = 0.06


def test_repair_candidate_enforces_width_and_anchor() -> None:
    cfg = DummyConfig()
    candidate = Candidate(
        inner_m=np.full(8, 300e-6),
        outer_m=np.full(8, 10e-6),
        topology=99,
        topology_size_m=1.0,
        topology_width_m=0.0,
    )
    repaired = repair_candidate(candidate, cfg)
    assert np.all(
        repaired.outer_m >= repaired.inner_m + cfg.min_rf_width_m - 1e-15
    )
    assert repaired.inner_m[-1] == cfg.base_inner_m
    assert repaired.outer_m[-1] == cfg.base_outer_m
    assert 0 <= repaired.topology <= 4


def test_mutate_preserves_terminal_anchor() -> None:
    cfg = DummyConfig()
    rng = np.random.default_rng(5)
    candidate = Candidate(
        inner_m=np.full(8, cfg.base_inner_m),
        outer_m=np.full(8, cfg.base_outer_m),
    )
    mutated = mutate(candidate, cfg, rng, progress=0.5)
    assert mutated.inner_m[-1] == cfg.base_inner_m
    assert mutated.outer_m[-1] == cfg.base_outer_m


def test_select_survivors_returns_requested_count() -> None:
    population = [Candidate() for _ in range(4)]
    evaluations = [
        Evaluation(np.array([1.0, 1.0]), {"valid": True}),
        Evaluation(np.array([2.0, 2.0]), {"valid": True}),
        Evaluation(np.array([0.5, 3.0]), {"valid": True}),
        Evaluation(np.array([3.0, 0.5]), {"valid": True}),
    ]
    survivors, survivor_evaluations = select_survivors(
        population,
        evaluations,
        2,
    )
    assert len(survivors) == 2
    assert len(survivor_evaluations) == 2


def test_choose_compromise_prefers_valid_candidate() -> None:
    evaluations = [
        Evaluation(np.array([0.1, 0.4]), {"invalid": True}),
        Evaluation(np.array([0.2, 0.2]), {"valid": True}),
        Evaluation(np.array([0.4, 0.1]), {"valid": True}),
    ]
    selected = choose_compromise_candidate(evaluations)
    assert selected in {1, 2}