from __future__ import annotations

import numpy as np

from core.ga.candidates import Candidate, Evaluation, N_CTRL, N_FEATURES, TOPOLOGIES
from core.ga.operators import repair_candidate


def house_seed(cfg) -> Candidate:
    return repair_candidate(
        Candidate(
            inner_m=np.full(N_CTRL, cfg.base_inner_m),
            outer_m=np.full(N_CTRL, cfg.base_outer_m),
        ),
        cfg,
    )


def local_seed(cfg, rng: np.random.Generator) -> Candidate:
    candidate = house_seed(cfg)
    envelope = np.linspace(1.0, 0.0, N_CTRL)
    candidate.inner_m += rng.normal(0.0, 18e-6, N_CTRL) * envelope
    candidate.outer_m += rng.normal(0.0, 24e-6, N_CTRL) * envelope
    candidate.topology = int(rng.integers(0, len(TOPOLOGIES)))
    candidate.topology_size_m = float(rng.uniform(25e-6, 100e-6))
    candidate.topology_width_m = float(rng.uniform(8e-6, 35e-6))

    for index in range(N_FEATURES):
        if rng.random() < 0.55:
            candidate.feature_kind[index] = int(rng.integers(1, 3))
            candidate.feature_operation[index] = int(
                rng.choice((-1, 1))
            )
            candidate.feature_radius_m[index] = float(
                rng.uniform(45e-6, 260e-6)
            )
            candidate.feature_theta_rad[index] = float(
                rng.uniform(0.0, np.pi / 4.0)
            )
            candidate.feature_p1_m[index] = float(
                rng.uniform(8e-6, 50e-6)
            )
            candidate.feature_p2_m[index] = float(
                rng.uniform(8e-6, 50e-6)
            )
            candidate.feature_angle_rad[index] = float(
                rng.uniform(-np.pi, np.pi)
            )

    return repair_candidate(candidate, cfg)


def broad_seed(cfg, rng: np.random.Generator) -> Candidate:
    candidate = Candidate(
        inner_m=rng.uniform(12e-6, 115e-6, N_CTRL),
        outer_m=rng.uniform(120e-6, 310e-6, N_CTRL),
        topology=int(rng.integers(0, len(TOPOLOGIES))),
        topology_size_m=float(rng.uniform(18e-6, 150e-6)),
        topology_width_m=float(rng.uniform(7e-6, 60e-6)),
    )

    for index in range(N_FEATURES):
        candidate.feature_kind[index] = int(rng.integers(0, 3))
        candidate.feature_operation[index] = int(rng.choice((-1, 1)))
        candidate.feature_radius_m[index] = float(
            rng.uniform(30e-6, 310e-6)
        )
        candidate.feature_theta_rad[index] = float(
            rng.uniform(0.0, np.pi / 4.0)
        )
        candidate.feature_p1_m[index] = float(
            rng.uniform(7e-6, 85e-6)
        )
        candidate.feature_p2_m[index] = float(
            rng.uniform(7e-6, 85e-6)
        )
        candidate.feature_angle_rad[index] = float(
            rng.uniform(-np.pi, np.pi)
        )

    return repair_candidate(candidate, cfg)


def initial_population(
    cfg,
    rng: np.random.Generator,
) -> list[Candidate]:
    population = [house_seed(cfg)]
    while len(population) < cfg.population:
        if len(population) < int(0.65 * cfg.population):
            population.append(local_seed(cfg, rng))
        else:
            population.append(broad_seed(cfg, rng))
    return population


def candidate_record(
    candidate: Candidate,
    evaluation: Evaluation,
    rank: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "pareto_rank": rank,
        "topology": TOPOLOGIES[candidate.topology],
        "topology_size_m": candidate.topology_size_m,
        "topology_width_m": candidate.topology_width_m,
    }

    for control_index, value in enumerate(candidate.inner_m):
        record[f"inner_{control_index}_m"] = float(value)
    for control_index, value in enumerate(candidate.outer_m):
        record[f"outer_{control_index}_m"] = float(value)

    for feature_index in range(N_FEATURES):
        record[f"feature_{feature_index}_kind"] = int(
            candidate.feature_kind[feature_index]
        )
        record[f"feature_{feature_index}_operation"] = int(
            candidate.feature_operation[feature_index]
        )
        record[f"feature_{feature_index}_radius_m"] = float(
            candidate.feature_radius_m[feature_index]
        )
        record[f"feature_{feature_index}_theta_rad"] = float(
            candidate.feature_theta_rad[feature_index]
        )
        record[f"feature_{feature_index}_p1_m"] = float(
            candidate.feature_p1_m[feature_index]
        )
        record[f"feature_{feature_index}_p2_m"] = float(
            candidate.feature_p2_m[feature_index]
        )
        record[f"feature_{feature_index}_angle_rad"] = float(
            candidate.feature_angle_rad[feature_index]
        )

    for objective_index, value in enumerate(evaluation.objectives):
        record[f"objective_{objective_index}"] = float(value)

    record.update(evaluation.metrics)
    return record


__all__ = [
    "broad_seed",
    "candidate_record",
    "house_seed",
    "initial_population",
    "local_seed",
]