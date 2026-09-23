"""Tests for the rectangular-panel BEM solver."""

from __future__ import annotations

import numpy as np

from core.electrostatics.bem import BEM2D
from core.electrostatics.bem_mesh import uniform_rectangular_grid


def _symmetric_two_rail_bem() -> BEM2D:
    x_edges_m = np.array([-200e-6, -50e-6, 50e-6, 200e-6])
    y_edges_m = np.array([-300e-6, 300e-6])

    electrode_map = np.array([[1.0, 0.0, 1.0]])

    panels_m, voltages_v = uniform_rectangular_grid(
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        cell_values=electrode_map,
    )
    return BEM2D(panels_m, voltages_v)


def test_bem_solves_for_finite_surface_charge() -> None:
    bem = _symmetric_two_rail_bem()
    charge = bem.solve(show_progress=False)

    assert charge.shape == (3,)
    assert np.all(np.isfinite(charge))


def test_symmetric_geometry_has_zero_horizontal_field_on_axis() -> None:
    bem = _symmetric_two_rail_bem()
    e_x, e_y, _ = bem.electric_field(0.0, 0.0, 100e-6)

    assert np.isclose(e_x, 0.0, atol=1e-7)
    assert np.isclose(e_y, 0.0, atol=1e-7)


def test_electric_field_matches_potential_derivative() -> None:
    bem = _symmetric_two_rail_bem()

    x_m = 20e-6
    y_m = 0.0
    z_m = 100e-6
    step_m = 0.1e-6

    potential_plus = bem.potential(x_m + step_m, y_m, z_m)
    potential_minus = bem.potential(x_m - step_m, y_m, z_m)
    e_x_numeric = -(potential_plus - potential_minus) / (2.0 * step_m)

    e_x, _, _ = bem.electric_field(x_m, y_m, z_m)

    assert np.isclose(e_x, e_x_numeric, rtol=1e-3, atol=1e-2)