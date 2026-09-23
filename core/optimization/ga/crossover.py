from __future__ import annotations

import numpy as np

from core.optimization.genome import Genome


def crossover(
    first: Genome,
    second: Genome,
    rng: np.random.Generator,
) -> Genome:
    child = first.copy()

    alpha = rng.uniform(-0.20, 1.20, size=first.inner_control_m.shape)
    child.inner_control_m = alpha * first.inner_control_m + (1.0 - alpha) * second.inner_control_m

    alpha = rng.uniform(-0.20, 1.20, size=first.outer_control_m.shape)
    child.outer_control_m = alpha * first.outer_control_m + (1.0 - alpha) * second.outer_control_m

    if rng.random() < 0.5:
        child.topology = int(second.topology)

    blend = float(rng.uniform(-0.20, 1.20))
    child.topology_size_m = blend * first.topology_size_m + (1.0 - blend) * second.topology_size_m

    blend = float(rng.uniform(-0.20, 1.20))
    child.topology_width_m = blend * first.topology_width_m + (1.0 - blend) * second.topology_width_m

    for index in range(child.feature_kind.shape[0]):
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
            child_array = getattr(child, attribute)

            weight = float(rng.uniform(-0.20, 1.20))
            child_array[index] = (
                weight * first_array[index] + (1.0 - weight) * second_array[index]
            )

    return child.clip()