"""Tests for X-junction geometry, masks, and edge-aligned meshes."""

from __future__ import annotations

import numpy as np

from core.geometry.junction_templates import (
    ISLAND_CIRCLE,
    ISLAND_NONE,
    ISLAND_SQUARE,
    make_house_style_x_junction,
)
from core.geometry.mask_builder import build_x_junction_bem


def test_empty_cross_has_no_rf_at_origin() -> None:
    parameters = make_house_style_x_junction(
        ion_height_m=90e-6,
        central_island_radius_m=0.0,
        central_island_kind=ISLAND_NONE,
    )

    model = build_x_junction_bem(
        parameters,
        edge_aligned=False,
        n_central=2,
        n_ground=2,
        n_rf=2,
        n_arm=2,
        n_outer=1,
    )

    centre_y = model.rf_mask.shape[0] // 2
    centre_x = model.rf_mask.shape[1] // 2

    assert not model.rf_mask[centre_y, centre_x]


def test_island_models_build() -> None:
    for island_kind in (ISLAND_SQUARE, ISLAND_CIRCLE):
        parameters = make_house_style_x_junction(
            ion_height_m=90e-6,
            central_island_radius_m=50e-6,
            central_island_kind=island_kind,
        )

        model = build_x_junction_bem(
            parameters,
            edge_aligned=False,
            n_central=2,
            n_ground=2,
            n_rf=2,
            n_arm=2,
            n_outer=1,
        )

        assert model.n_panels > 0
        assert model.rf_mask.shape[0] == model.rf_mask.shape[1]


def test_outer_edge_shift_changes_edge_aligned_mesh() -> None:
    baseline = make_house_style_x_junction(
        ion_height_m=90e-6,
        taper_length_m=150e-6,
        inner_edge_shift_at_centre_m=0.0,
        outer_edge_shift_at_centre_m=0.0,
    )
    expanded = make_house_style_x_junction(
        ion_height_m=90e-6,
        taper_length_m=150e-6,
        inner_edge_shift_at_centre_m=0.0,
        outer_edge_shift_at_centre_m=25e-6,
    )

    baseline_model = build_x_junction_bem(
        baseline,
        edge_aligned=True,
        n_longitudinal_samples=11,
        max_cell_size_centre_m=20e-6,
    )
    expanded_model = build_x_junction_bem(
        expanded,
        edge_aligned=True,
        n_longitudinal_samples=11,
        max_cell_size_centre_m=20e-6,
    )

    same_edges = np.array_equal(
        baseline_model.x_edges_m,
        expanded_model.x_edges_m,
    )
    same_mask = (
        baseline_model.rf_mask.shape == expanded_model.rf_mask.shape
        and np.array_equal(
            baseline_model.rf_mask,
            expanded_model.rf_mask,
        )
    )

    assert not (same_edges and same_mask)


def test_edge_aligned_mesh_is_symmetric_and_strictly_ordered() -> None:
    parameters = make_house_style_x_junction(
        ion_height_m=90e-6,
        taper_length_m=150e-6,
        inner_edge_shift_at_centre_m=-15e-6,
        outer_edge_shift_at_centre_m=25e-6,
    )

    model = build_x_junction_bem(
        parameters,
        edge_aligned=True,
        n_longitudinal_samples=13,
        max_cell_size_centre_m=15e-6,
    )

    assert np.all(np.diff(model.x_edges_m) > 0.0)
    assert np.allclose(model.x_edges_m, -model.x_edges_m[::-1])
    assert np.all(np.diff(model.y_edges_m) > 0.0)
    assert np.allclose(model.y_edges_m, -model.y_edges_m[::-1])