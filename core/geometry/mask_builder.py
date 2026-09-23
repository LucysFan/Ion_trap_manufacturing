"""Build BEM models from parameterized X-junction geometry templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.electrostatics.bem import BEM2D
from core.electrostatics.bem_mesh import (
    edge_aligned_axis_edges,
    edge_aligned_sparse_partition,
    geometry_aware_quadtree_rectangular_panels,
    quadtree_rectangular_panels,
    uniform_rectangular_grid,
)

from core.geometry.junction_templates import (
    XJunctionParameters,
    baseline_x_junction_rf_mask,
)


@dataclass
class XJunctionBEMModel:
    """BEM model together with mesh geometry and RF classification."""

    bem: BEM2D
    parameters: XJunctionParameters
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    rf_mask: np.ndarray
    panel_centres_x_m: np.ndarray
    panel_centres_y_m: np.ndarray

    @property
    def n_panels(self) -> int:
        """Total number of rectangular BEM panels."""
        return self.bem.n_panels


def _sample_boundary_positions_m(
    parameters: XJunctionParameters,
    *,
    n_longitudinal_samples: int,
) -> np.ndarray:
    """Sample both RF boundaries so taper changes become mesh-aligned."""
    if n_longitudinal_samples < 2:
        raise ValueError("n_longitudinal_samples must be at least two.")

    if parameters.taper_length_m > 0.0:
        longitudinal_m = np.linspace(
            0.0,
            parameters.taper_length_m,
            n_longitudinal_samples,
        )
    else:
        longitudinal_m = np.array([0.0, parameters.arm_length_m])

    inner_m, outer_m = parameters.rail_boundaries_m(longitudinal_m)

    positions = [
        0.0,
        parameters.half_ground_width_m,
        parameters.rf_outer_radius_m,
        parameters.rf_start_radius_m,
        parameters.arm_length_m,
        parameters.central_island_radius_m,
    ]
    positions.extend(np.asarray(inner_m, dtype=float).tolist())
    positions.extend(np.asarray(outer_m, dtype=float).tolist())

    return np.asarray(positions, dtype=float)


def build_x_junction_bem(
    parameters: XJunctionParameters,
    *,
    n_central: int = 3,
    n_ground: int = 3,
    n_rf: int = 6,
    n_arm: int = 5,
    n_outer: int = 2,
    edge_aligned: bool = True,
    n_longitudinal_samples: int = 17,
    max_cell_size_centre_m: float = 12.5e-6,
    max_cell_size_outer_m: float = 75e-6,
    centre_refinement_radius_m: float = 250e-6,
) -> XJunctionBEMModel:
    """Create idealized BEM model for a symmetric X-junction.

    When ``edge_aligned`` is true, both inner and outer RF boundaries are
    inserted explicitly into the mesh. This is the preferred configuration
    for controlled geometry scans and later GA evaluation.
    """
    parameters.validate()

    if edge_aligned:
        critical_positions_m = _sample_boundary_positions_m(
            parameters,
            n_longitudinal_samples=n_longitudinal_samples,
        )

        axis_edges_m = edge_aligned_axis_edges(
            outer_extent_m=parameters.outer_extent_m,
            critical_positions_m=critical_positions_m,
            max_cell_size_centre_m=max_cell_size_centre_m,
            max_cell_size_outer_m=max_cell_size_outer_m,
            centre_refinement_radius_m=centre_refinement_radius_m,
        )
    else:
        axis_edges_m = _legacy_critical_axis_edges(
            parameters,
            n_central=n_central,
            n_ground=n_ground,
            n_rf=n_rf,
            n_arm=n_arm,
            n_outer=n_outer,
        )

    x_edges_m = axis_edges_m
    y_edges_m = axis_edges_m

    x_centres_m = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y_centres_m = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])

    x_grid_m, y_grid_m = np.meshgrid(
        x_centres_m,
        y_centres_m,
        indexing="xy",
    )

    rf_mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        parameters,
    )

    panels_m, voltages_v = uniform_rectangular_grid(
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        cell_values=rf_mask.astype(float),
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    return XJunctionBEMModel(
        bem=bem,
        parameters=parameters,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        rf_mask=rf_mask,
        panel_centres_x_m=x_grid_m,
        panel_centres_y_m=y_grid_m,
    )


def _legacy_critical_axis_edges(
    parameters: XJunctionParameters,
    *,
    n_central: int,
    n_ground: int,
    n_rf: int,
    n_arm: int,
    n_outer: int,
) -> np.ndarray:
    """Return prior coarse symmetric mesh for backwards comparison."""
    parameters.validate()

    if min(n_central, n_ground, n_rf, n_arm, n_outer) < 1:
        raise ValueError("All subdivision counts must be at least one.")

    positive_nodes_m = sorted(
        {
            0.0,
            parameters.central_island_radius_m,
            parameters.half_ground_width_m,
            parameters.rf_outer_radius_m,
            parameters.arm_length_m,
            parameters.outer_extent_m,
        }
    )

    positive_edges: list[float] = [0.0]

    for left_m, right_m in zip(positive_nodes_m[:-1], positive_nodes_m[1:]):
        if right_m <= left_m:
            continue

        if right_m <= max(parameters.central_island_radius_m, 1e-30):
            count = n_central
        elif right_m <= parameters.half_ground_width_m:
            count = n_ground
        elif right_m <= parameters.rf_outer_radius_m:
            count = n_rf
        elif right_m <= parameters.arm_length_m:
            count = n_arm
        else:
            count = n_outer

        positive_edges.extend(
            np.linspace(left_m, right_m, count + 1)[1:].tolist()
        )

    positive = np.asarray(positive_edges, dtype=float)
    return np.concatenate([-positive[:0:-1], positive])

def build_sparse_x_junction_bem(
    parameters: XJunctionParameters,
    *,
    n_longitudinal_samples: int = 13,
    central_half_extent_m: float = 180e-6,
    central_cell_size_m: float = 20e-6,
    transition_cell_size_m: float = 55e-6,
    outer_cell_size_m: float = 180e-6,
) -> XJunctionBEMModel:
    """Create a lower-panel-count edge-aligned X-junction BEM model.

    This is the intended L1 screening builder for two-edge scans and the
    first nominal GA. It keeps inner/outer physical boundary positions in
    the mesh while using larger cells far from the central junction.

    The solver remains dense BEM; therefore use panel budgets below roughly
    1500–2500 for routine repeated evaluation.
    """
    parameters.validate()

    critical_positions_m = _sample_boundary_positions_m(
        parameters,
        n_longitudinal_samples=n_longitudinal_samples,
    )

    def classify_cell(x_m: float, y_m: float) -> float:
        mask = baseline_x_junction_rf_mask(
            np.asarray([x_m], dtype=float),
            np.asarray([y_m], dtype=float),
            parameters,
        )
        return float(mask[0])

    panels_m, voltages_v, axis_edges_m = edge_aligned_sparse_partition(
        outer_extent_m=parameters.outer_extent_m,
        critical_positions_m=critical_positions_m,
        central_half_extent_m=central_half_extent_m,
        central_cell_size_m=central_cell_size_m,
        transition_cell_size_m=transition_cell_size_m,
        outer_cell_size_m=outer_cell_size_m,
        classify_cell=classify_cell,
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    x_centres_m = 0.5 * (axis_edges_m[:-1] + axis_edges_m[1:])
    y_centres_m = 0.5 * (axis_edges_m[:-1] + axis_edges_m[1:])
    x_grid_m, y_grid_m = np.meshgrid(
        x_centres_m,
        y_centres_m,
        indexing="xy",
    )

    rf_mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        parameters,
    )

    return XJunctionBEMModel(
        bem=bem,
        parameters=parameters,
        x_edges_m=axis_edges_m,
        y_edges_m=axis_edges_m,
        rf_mask=rf_mask,
        panel_centres_x_m=x_grid_m,
        panel_centres_y_m=y_grid_m,
    )

def build_quadtree_x_junction_bem(
    parameters: XJunctionParameters,
    *,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 20e-6,
    boundary_max_cell_m: float = 20e-6,
    outer_max_cell_m: float = 180e-6,
    min_cell_m: float = 8e-6,
) -> XJunctionBEMModel:
    """Build an adaptive quadtree BEM model for an X-junction.

    This is the recommended screening model before the first GA. It keeps
    fine panels near RF boundaries and the junction while representing
    large uniform ground regions by large panels.
    """
    parameters.validate()

    def classify_point(x_m: float, y_m: float) -> float:
        mask = baseline_x_junction_rf_mask(
            np.asarray([x_m], dtype=float),
            np.asarray([y_m], dtype=float),
            parameters,
        )
        return float(mask[0])

    panels_m, voltages_v = quadtree_rectangular_panels(
        outer_extent_m=parameters.outer_extent_m,
        classify_point=classify_point,
        central_half_extent_m=central_half_extent_m,
        central_max_cell_m=central_max_cell_m,
        boundary_max_cell_m=boundary_max_cell_m,
        outer_max_cell_m=outer_max_cell_m,
        min_cell_m=min_cell_m,
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    # Quadtree has no shared x/y axis. Store panel centres as one-dimensional
    # arrays reshaped to column vectors; code that requires a regular mask
    # must use the dedicated visualization helper below.
    centres_m = bem.panel_centres_m
    x_centres_m = centres_m[:, 0]
    y_centres_m = centres_m[:, 1]
    rf_mask = voltages_v > 0.5

    return XJunctionBEMModel(
        bem=bem,
        parameters=parameters,
        x_edges_m=np.array([], dtype=float),
        y_edges_m=np.array([], dtype=float),
        rf_mask=rf_mask,
        panel_centres_x_m=x_centres_m,
        panel_centres_y_m=y_centres_m,
    )

def _interval_overlaps(
    lower_a_m: float,
    upper_a_m: float,
    lower_b_m: float,
    upper_b_m: float,
) -> bool:
    """Return true if two closed scalar intervals overlap."""
    return not (upper_a_m < lower_b_m or upper_b_m < lower_a_m)


def _sample_interval_extrema(
    function,
    left_m: float,
    right_m: float,
    *,
    n_samples: int = 9,
) -> tuple[float, float]:
    """Approximate min/max of scalar boundary function on one interval."""
    samples_m = np.linspace(left_m, right_m, n_samples)
    values_m = np.asarray(function(samples_m), dtype=float)
    return float(np.min(values_m)), float(np.max(values_m))


def _x_oriented_boundary_intersects_cell(
    parameters: XJunctionParameters,
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
) -> bool:
    """Check intersection with horizontal-arm RF boundaries.

    Horizontal RF rails are bounded by:

        |y| = r_in(|x|)
        |y| = r_out(|x|)

    only where |x| lies in the longitudinal RF arm interval.
    """
    abs_x_min_m = 0.0 if x_left_m <= 0.0 <= x_right_m else min(
        abs(x_left_m),
        abs(x_right_m),
    )
    abs_x_max_m = max(abs(x_left_m), abs(x_right_m))

    if abs_x_max_m < parameters.rf_start_radius_m:
        return False
    if abs_x_min_m > parameters.arm_length_m:
        return False

    longitudinal_left_m = max(abs_x_min_m, parameters.rf_start_radius_m)
    longitudinal_right_m = min(abs_x_max_m, parameters.arm_length_m)

    if longitudinal_right_m < longitudinal_left_m:
        return False

    def inner_function(s_m: np.ndarray) -> np.ndarray:
        inner_m, _ = parameters.rail_boundaries_m(s_m)
        return inner_m

    def outer_function(s_m: np.ndarray) -> np.ndarray:
        _, outer_m = parameters.rail_boundaries_m(s_m)
        return outer_m

    inner_min_m, inner_max_m = _sample_interval_extrema(
        inner_function,
        longitudinal_left_m,
        longitudinal_right_m,
    )
    outer_min_m, outer_max_m = _sample_interval_extrema(
        outer_function,
        longitudinal_left_m,
        longitudinal_right_m,
    )

    positive_y_overlap = (
        _interval_overlaps(y_bottom_m, y_top_m, inner_min_m, inner_max_m)
        or _interval_overlaps(y_bottom_m, y_top_m, outer_min_m, outer_max_m)
    )
    negative_y_overlap = (
        _interval_overlaps(y_bottom_m, y_top_m, -inner_max_m, -inner_min_m)
        or _interval_overlaps(y_bottom_m, y_top_m, -outer_max_m, -outer_min_m)
    )

    return positive_y_overlap or negative_y_overlap


def _physical_boundary_intersects_cell(
    parameters: XJunctionParameters,
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
) -> bool:
    """Return true if any ideal RF/ground boundary intersects one cell.

    This checks horizontal arms directly and vertical arms through coordinate
    exchange. It also checks the longitudinal start/end boundaries and
    central island edges when relevant.
    """
    if _x_oriented_boundary_intersects_cell(
        parameters,
        x_left_m,
        x_right_m,
        y_bottom_m,
        y_top_m,
    ):
        return True

    if _x_oriented_boundary_intersects_cell(
        parameters,
        y_bottom_m,
        y_top_m,
        x_left_m,
        x_right_m,
    ):
        return True

    start_m = parameters.rf_start_radius_m
    arm_m = parameters.arm_length_m

    # RF rails start/end at longitudinal ±start and ±arm. If those vertical
    # or horizontal lines cross the cell, refine it.
    for coordinate_m in (-arm_m, -start_m, start_m, arm_m):
        if x_left_m <= coordinate_m <= x_right_m:
            if (
                y_bottom_m <= parameters.maximum_rf_outer_radius_m
                and y_top_m >= -parameters.maximum_rf_outer_radius_m
            ):
                return True

        if y_bottom_m <= coordinate_m <= y_top_m:
            if (
                x_left_m <= parameters.maximum_rf_outer_radius_m
                and x_right_m >= -parameters.maximum_rf_outer_radius_m
            ):
                return True

    radius_m = parameters.central_island_radius_m

    if radius_m > 0.0:
        if parameters.central_island_kind == "square":
            if (
                _interval_overlaps(x_left_m, x_right_m, -radius_m, radius_m)
                and _interval_overlaps(y_bottom_m, y_top_m, -radius_m, radius_m)
            ):
                # A cell touching the bounding square is conservatively
                # treated as boundary-adjacent.
                return True

        elif parameters.central_island_kind == "circle":
            x_closest_m = np.clip(0.0, x_left_m, x_right_m)
            y_closest_m = np.clip(0.0, y_bottom_m, y_top_m)
            x_farthest_m = max(abs(x_left_m), abs(x_right_m))
            y_farthest_m = max(abs(y_bottom_m), abs(y_top_m))

            min_radius_sq_m2 = x_closest_m**2 + y_closest_m**2
            max_radius_sq_m2 = x_farthest_m**2 + y_farthest_m**2

            if min_radius_sq_m2 <= radius_m**2 <= max_radius_sq_m2:
                return True

    return False


def build_geometry_aware_quadtree_x_junction_bem(
    parameters: XJunctionParameters,
    *,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 28e-6,
    boundary_max_cell_m: float = 10e-6,
    outer_max_cell_m: float = 180e-6,
    min_cell_m: float = 5e-6,
) -> XJunctionBEMModel:
    """Build geometry-aware adaptive quadtree BEM model.

    The exact inner and outer RF boundaries trigger refinement whenever they
    cross a panel rectangle. This is the L1 screening model intended for
    two-edge scans and the first compact GA.
    """
    parameters.validate()

    def classify_point(x_m: float, y_m: float) -> float:
        mask = baseline_x_junction_rf_mask(
            np.asarray([x_m], dtype=float),
            np.asarray([y_m], dtype=float),
            parameters,
        )
        return float(mask[0])

    def intersects_boundary(
        x_left_m: float,
        x_right_m: float,
        y_bottom_m: float,
        y_top_m: float,
    ) -> bool:
        return _physical_boundary_intersects_cell(
            parameters,
            x_left_m,
            x_right_m,
            y_bottom_m,
            y_top_m,
        )

    panels_m, voltages_v = geometry_aware_quadtree_rectangular_panels(
        outer_extent_m=parameters.outer_extent_m,
        classify_point=classify_point,
        intersects_boundary=intersects_boundary,
        central_half_extent_m=central_half_extent_m,
        central_max_cell_m=central_max_cell_m,
        boundary_max_cell_m=boundary_max_cell_m,
        outer_max_cell_m=outer_max_cell_m,
        min_cell_m=min_cell_m,
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    centres_m = bem.panel_centres_m
    rf_mask = voltages_v > 0.5

    return XJunctionBEMModel(
        bem=bem,
        parameters=parameters,
        x_edges_m=np.array([], dtype=float),
        y_edges_m=np.array([], dtype=float),
        rf_mask=rf_mask,
        panel_centres_x_m=centres_m[:, 0],
        panel_centres_y_m=centres_m[:, 1],
    )