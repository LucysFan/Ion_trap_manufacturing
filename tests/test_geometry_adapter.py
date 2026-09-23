from __future__ import annotations

import numpy as np

from core.optimization.geometry_adapter import (
    GeometryAdapterConfig,
    build_simple_mask_from_genome,
)
from core.optimization.genome import (
    FEATURE_CIRCLE,
    FEATURE_OP_ADD_RF,
    Genome,
    TOPOLOGY_CENTRAL_RF_DISK,
)


def make_grid(n: int = 81, extent_m: float = 400e-6) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-extent_m, extent_m, n)
    x_m, y_m = np.meshgrid(axis, axis, indexing="xy")
    return x_m, y_m


def test_build_simple_mask_returns_same_shape() -> None:
    genome = Genome()
    x_m, y_m = make_grid()

    mask = build_simple_mask_from_genome(genome, x_m, y_m)

    assert mask.shape == x_m.shape
    assert mask.dtype == np.float64


def test_baseline_mask_is_not_empty() -> None:
    genome = Genome()
    x_m, y_m = make_grid()

    mask = build_simple_mask_from_genome(genome, x_m, y_m)

    assert np.any(mask > 0.5)
    assert np.any(mask < 0.5)


def test_central_topology_changes_mask() -> None:
    baseline = Genome()
    modified = baseline.copy()
    modified.topology = TOPOLOGY_CENTRAL_RF_DISK
    modified.topology_size_m = 60e-6

    x_m, y_m = make_grid()

    mask0 = build_simple_mask_from_genome(baseline, x_m, y_m)
    mask1 = build_simple_mask_from_genome(modified, x_m, y_m)

    assert np.any(mask0 != mask1)


def test_feature_changes_mask() -> None:
    baseline = Genome()
    modified = baseline.copy()

    modified.feature_kind[0] = FEATURE_CIRCLE
    modified.feature_operation[0] = FEATURE_OP_ADD_RF
    modified.feature_radius_m[0] = 100e-6
    modified.feature_theta_rad[0] = 0.0
    modified.feature_p1_m[0] = 20e-6

    x_m, y_m = make_grid()

    mask0 = build_simple_mask_from_genome(baseline, x_m, y_m)
    mask1 = build_simple_mask_from_genome(modified, x_m, y_m)

    assert np.any(mask0 != mask1)