from __future__ import annotations

import numpy as np

from core.optimization.genome import Genome, TOPOLOGY_NAMES


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    *,
    progress: float = 0.0,
    mutation_probability: float = 0.22,
    topology_mutation_probability: float = 0.06,
) -> Genome:
    mutated = genome.copy()
    scale = 1.0 - 0.75 * float(progress)

    for array, sigma in (
        (mutated.inner_control_m, 18e-6),
        (mutated.outer_control_m, 24e-6),
    ):
        selected = rng.random(array.shape) < mutation_probability
        if np.any(selected):
            array[selected] += rng.normal(0.0, sigma * scale, size=int(np.sum(selected)))

    if rng.random() < topology_mutation_probability:
        mutated.topology = int(rng.integers(0, len(TOPOLOGY_NAMES)))

    if rng.random() < mutation_probability:
        mutated.topology_size_m += float(rng.normal(0.0, 18e-6 * scale))

    if rng.random() < mutation_probability:
        mutated.topology_width_m += float(rng.normal(0.0, 10e-6 * scale))

    for index in range(mutated.feature_kind.shape[0]):
        if rng.random() < topology_mutation_probability:
            mutated.feature_kind[index] = int(rng.integers(0, 3))

        if rng.random() < topology_mutation_probability:
            mutated.feature_operation[index] = -1 if rng.random() < 0.5 else 1

        if rng.random() < mutation_probability:
            mutated.feature_radius_m[index] += float(rng.normal(0.0, 25e-6 * scale))

        if rng.random() < mutation_probability:
            mutated.feature_theta_rad[index] += float(rng.normal(0.0, 0.12 * scale))

        if rng.random() < mutation_probability:
            mutated.feature_p1_m[index] += float(rng.normal(0.0, 12e-6 * scale))

        if rng.random() < mutation_probability:
            mutated.feature_p2_m[index] += float(rng.normal(0.0, 12e-6 * scale))

        if rng.random() < mutation_probability:
            mutated.feature_angle_rad[index] += float(rng.normal(0.0, 0.25 * scale))

    return mutated.clip()