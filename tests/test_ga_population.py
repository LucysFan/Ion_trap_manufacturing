from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ga.candidates import Candidate, Evaluation
from core.ga.population import candidate_record, house_seed, initial_population


@dataclass
class DummyConfig:
    population: int = 10
    min_rf_width_m: float = 18e-6
    base_inner_m: float = 37.35e-6
    base_outer_m: float = 216.45e-6
    min_feature_m: float = 6e-6


def test_house_seed_uses_baseline_anchor() -> None:
    cfg = DummyConfig()
    candidate = house_seed(cfg)
    assert candidate.inner_m[-1] == cfg.base_inner_m
    assert candidate.outer_m[-1] == cfg.base_outer_m


def test_initial_population_respects_requested_size() -> None:
    cfg = DummyConfig(population=12)
    rng = np.random.default_rng(123)
    population = initial_population(cfg, rng)
    assert len(population) == 12


def test_candidate_record_contains_metrics_and_objectives() -> None:
    candidate = Candidate()
    evaluation = Evaluation(
        objectives=np.array([1.0, 2.0]),
        metrics={"height_peak_m": 1e-6, "valid": True},
    )
    record = candidate_record(candidate, evaluation, rank=0)
    assert record["pareto_rank"] == 0
    assert record["objective_0"] == 1.0
    assert record["height_peak_m"] == 1e-6