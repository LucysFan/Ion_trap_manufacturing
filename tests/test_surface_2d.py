"""Tests for the analytic 2D surface-electrode model."""

from __future__ import annotations

import numpy as np

from core.electrostatics.surface_2d import build_linear_surface_trap


def test_analytic_field_matches_potential_derivatives() -> None:
    trap = build_linear_surface_trap(
        ground_width_m=74.7e-6,
        rf_width_m=179.1e-6,
    )

    assert trap.verify_consistency(
        x_m=7e-6,
        z_m=90e-6,
        finite_difference_step_m=1e-8,
        tolerance=1e-4,
    )


def test_symmetric_trap_has_no_horizontal_field_on_axis() -> None:
    trap = build_linear_surface_trap(
        ground_width_m=74.7e-6,
        rf_width_m=179.1e-6,
    )

    e_x, _ = trap.electric_field(0.0, 90e-6)
    assert np.isclose(e_x, 0.0, atol=1e-9)