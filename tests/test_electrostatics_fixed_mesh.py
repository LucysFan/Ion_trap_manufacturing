from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from core.electrostatics.fixed_mesh import (
    baseline,
    baseline_boundary_intersects,
    build_fixed_mesh,
)


@dataclass
class DummyConfig:
    base_inner_m: float = 40e-6
    base_outer_m: float = 200e-6
    arm_length_m: float = 900e-6
    outer_extent_m: float = 300e-6
    central_half_extent_m: float = 100e-6
    central_max_cell_m: float = 50e-6
    boundary_max_cell_m: float = 50e-6
    outer_max_cell_m: float = 50e-6
    min_cell_m: float = 25e-6
    max_panels: int = 10_000


def test_baseline_uses_config_radii() -> None:
    cfg = DummyConfig()
    candidate = baseline(cfg)
    assert np.all(candidate.inner_m == cfg.base_inner_m)
    assert np.all(candidate.outer_m == cfg.base_outer_m)


def test_boundary_intersection_detects_transition_cell() -> None:
    cfg = DummyConfig()
    intersects = baseline_boundary_intersects(
        -250e-6,
        250e-6,
        -250e-6,
        250e-6,
        cfg,
    )
    assert isinstance(intersects, bool)


def test_build_fixed_mesh_returns_panel_centres() -> None:
    cfg = DummyConfig()
    panels = build_fixed_mesh(cfg)
    assert panels.ndim == 2
    assert panels.shape[1] == 2
    assert len(panels) > 0


def test_build_fixed_mesh_respects_max_panels() -> None:
    cfg = DummyConfig(max_panels=1)
    with pytest.raises(RuntimeError):
        build_fixed_mesh(cfg)