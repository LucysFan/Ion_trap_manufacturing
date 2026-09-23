from __future__ import annotations

import numpy as np

from core.optimization.genome import Genome


def test_genome_defaults_have_consistent_shapes() -> None:
    genome = Genome()

    assert genome.inner_control_m.ndim == 1
    assert genome.outer_control_m.ndim == 1
    assert genome.inner_control_m.shape == genome.outer_control_m.shape

    assert genome.feature_kind.ndim == 1
    assert genome.feature_operation.ndim == 1
    assert genome.feature_radius_m.ndim == 1
    assert genome.feature_theta_rad.ndim == 1
    assert genome.feature_p1_m.ndim == 1
    assert genome.feature_p2_m.ndim == 1
    assert genome.feature_angle_rad.ndim == 1


def test_genome_copy_is_deep() -> None:
    genome = Genome()
    copied = genome.copy()

    copied.inner_control_m[0] += 1e-6
    copied.outer_control_m[1] += 2e-6
    copied.feature_kind[0] = 1

    assert not np.shares_memory(genome.inner_control_m, copied.inner_control_m)
    assert not np.shares_memory(genome.outer_control_m, copied.outer_control_m)

    assert genome.inner_control_m[0] != copied.inner_control_m[0]
    assert genome.outer_control_m[1] != copied.outer_control_m[1]
    assert genome.feature_kind[0] != copied.feature_kind[0]