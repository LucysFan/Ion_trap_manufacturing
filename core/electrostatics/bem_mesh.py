"""Mesh-generation helpers for rectangular-panel BEM models."""

from __future__ import annotations

import numpy as np


def graded_interval_edges(
    left_m: float,
    right_m: float,
    n_panels: int,
    *,
    clustering: float = 2.5,
) -> np.ndarray:
    """Return monotonic interval edges with denser nodes near both ends."""
    if right_m <= left_m:
        raise ValueError("right_m must exceed left_m.")
    if n_panels < 1:
        raise ValueError("n_panels must be at least one.")
    if clustering < 0.0:
        raise ValueError("clustering must be non-negative.")

    parameter = np.linspace(0.0, 1.0, n_panels + 1)

    if clustering == 0.0:
        mapped = parameter
    else:
        mapped = 0.5 * (
            1.0
            + np.tanh(clustering * (2.0 * parameter - 1.0))
            / np.tanh(clustering)
        )

    return left_m + (right_m - left_m) * mapped


def rectangular_strip_panels(
    x_edges_m: np.ndarray | list[float],
    electrode_voltages_v: np.ndarray | list[float],
    *,
    y_min_m: float,
    y_max_m: float,
    panels_per_strip: int = 1,
    graded: bool = True,
    clustering: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide each x-strip into rectangular panels."""
    x_edges = np.asarray(x_edges_m, dtype=float)
    voltages = np.asarray(electrode_voltages_v, dtype=float)

    if x_edges.ndim != 1 or voltages.ndim != 1:
        raise ValueError("x_edges_m and electrode_voltages_v must be 1D.")
    if len(x_edges) != len(voltages) + 1:
        raise ValueError("Need one electrode voltage per x-strip.")
    if np.any(np.diff(x_edges) <= 0.0):
        raise ValueError("x_edges_m must be strictly increasing.")
    if y_max_m <= y_min_m:
        raise ValueError("y_max_m must exceed y_min_m.")
    if panels_per_strip < 1:
        raise ValueError("panels_per_strip must be at least one.")

    panels: list[list[float]] = []
    panel_voltages: list[float] = []

    for index, voltage_v in enumerate(voltages):
        x_left_m = float(x_edges[index])
        x_right_m = float(x_edges[index + 1])

        local_edges = (
            graded_interval_edges(
                x_left_m,
                x_right_m,
                panels_per_strip,
                clustering=clustering,
            )
            if graded
            else np.linspace(x_left_m, x_right_m, panels_per_strip + 1)
        )

        for left_m, right_m in zip(local_edges[:-1], local_edges[1:]):
            panels.append([left_m, right_m, y_min_m, y_max_m])
            panel_voltages.append(float(voltage_v))

    return np.asarray(panels, dtype=float), np.asarray(panel_voltages, dtype=float)


def uniform_rectangular_grid(
    x_edges_m: np.ndarray | list[float],
    y_edges_m: np.ndarray | list[float],
    cell_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rectangular x/y grid of cell values into BEM panels."""
    x_edges = np.asarray(x_edges_m, dtype=float)
    y_edges = np.asarray(y_edges_m, dtype=float)
    values = np.asarray(cell_values, dtype=float)

    expected_shape = (len(y_edges) - 1, len(x_edges) - 1)
    if values.shape != expected_shape:
        raise ValueError(
            f"cell_values must have shape {expected_shape}, got {values.shape}."
        )
    if np.any(np.diff(x_edges) <= 0.0):
        raise ValueError("x_edges_m must be strictly increasing.")
    if np.any(np.diff(y_edges) <= 0.0):
        raise ValueError("y_edges_m must be strictly increasing.")

    panels: list[list[float]] = []
    panel_voltages: list[float] = []

    for y_index in range(len(y_edges) - 1):
        for x_index in range(len(x_edges) - 1):
            panels.append(
                [
                    x_edges[x_index],
                    x_edges[x_index + 1],
                    y_edges[y_index],
                    y_edges[y_index + 1],
                ]
            )
            panel_voltages.append(values[y_index, x_index])

    return np.asarray(panels, dtype=float), np.asarray(panel_voltages, dtype=float)


def _unique_sorted_edges(
    edges_m: list[float] | np.ndarray,
    *,
    tolerance_m: float = 1e-15,
) -> np.ndarray:
    """Sort and merge numerically coincident mesh edges."""
    sorted_edges = np.sort(np.asarray(edges_m, dtype=float))

    if len(sorted_edges) < 2:
        raise ValueError("At least two mesh edges are required.")

    merged = [float(sorted_edges[0])]
    for edge_m in sorted_edges[1:]:
        if float(edge_m) - merged[-1] > tolerance_m:
            merged.append(float(edge_m))

    output = np.asarray(merged, dtype=float)
    if np.any(np.diff(output) <= 0.0):
        raise RuntimeError("Failed to create strictly increasing mesh edges.")

    return output


def edge_aligned_axis_edges(
    *,
    outer_extent_m: float,
    critical_positions_m: np.ndarray | list[float],
    max_cell_size_centre_m: float = 10e-6,
    max_cell_size_outer_m: float = 75e-6,
    centre_refinement_radius_m: float = 250e-6,
) -> np.ndarray:
    """Build a symmetric nonuniform axis with explicit critical edge positions.

    Every critical position and its mirror image are inserted. Intervals in
    the central area use a smaller maximum cell size. This makes RF-edge
    motions visible to BEM rather than hiding them inside one raster cell.
    """
    if outer_extent_m <= 0.0:
        raise ValueError("outer_extent_m must be positive.")
    if max_cell_size_centre_m <= 0.0:
        raise ValueError("max_cell_size_centre_m must be positive.")
    if max_cell_size_outer_m <= 0.0:
        raise ValueError("max_cell_size_outer_m must be positive.")
    if centre_refinement_radius_m <= 0.0:
        raise ValueError("centre_refinement_radius_m must be positive.")

    critical = np.asarray(critical_positions_m, dtype=float)
    if not np.all(np.isfinite(critical)):
        raise ValueError("critical_positions_m must contain finite values.")

    positive_critical = np.abs(critical)
    positive_critical = positive_critical[positive_critical < outer_extent_m]

    base_positive = [
        0.0,
        min(centre_refinement_radius_m, outer_extent_m),
        outer_extent_m,
    ]
    base_positive.extend(positive_critical.tolist())

    positive_nodes = _unique_sorted_edges(base_positive)
    positive_edges: list[float] = [0.0]

    for left_m, right_m in zip(positive_nodes[:-1], positive_nodes[1:]):
        midpoint_m = 0.5 * (left_m + right_m)
        max_cell_size_m = (
            max_cell_size_centre_m
            if midpoint_m <= centre_refinement_radius_m
            else max_cell_size_outer_m
        )

        n_cells = max(1, int(np.ceil((right_m - left_m) / max_cell_size_m)))
        segment = np.linspace(left_m, right_m, n_cells + 1)[1:]
        positive_edges.extend(segment.tolist())

    positive = _unique_sorted_edges(positive_edges)
    negative = -positive[:0:-1]
    return np.concatenate([negative, positive])

def _append_panel(
    panels: list[list[float]],
    values: list[float],
    *,
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
    value: float,
) -> None:
    """Append one valid rectangular panel."""
    if x_right_m <= x_left_m or y_top_m <= y_bottom_m:
        raise ValueError("Cannot append a panel with non-positive dimensions.")

    panels.append([x_left_m, x_right_m, y_bottom_m, y_top_m])
    values.append(float(value))


def sparse_rectangular_partition(
    *,
    outer_extent_m: float,
    central_half_extent_m: float,
    central_cell_size_m: float,
    transition_cell_size_m: float,
    outer_cell_size_m: float,
    classify_cell,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a hierarchical rectangular panel mesh.

    The plane is split into three square regions:

    1. central square:    |x|, |y| <= central_half_extent_m
    2. transition square: central_half_extent_m < max(|x|,|y|)
                          <= 2 * central_half_extent_m
    3. outer frame:       remaining area up to outer_extent_m

    Each zone has its own panel size. ``classify_cell`` is called on every
    panel centre and must return the panel boundary voltage, normally 0 or 1.

    This function reduces panel count relative to a globally fine grid, but
    it does not yet enforce curved or row-dependent RF boundaries by itself.
    Boundary alignment is supplied by the adaptive edge list in the next
    function.
    """
    if outer_extent_m <= 0.0:
        raise ValueError("outer_extent_m must be positive.")
    if not 0.0 < central_half_extent_m < outer_extent_m:
        raise ValueError(
            "central_half_extent_m must lie strictly inside outer_extent_m."
        )
    if min(
        central_cell_size_m,
        transition_cell_size_m,
        outer_cell_size_m,
    ) <= 0.0:
        raise ValueError("All cell sizes must be positive.")

    transition_half_extent_m = min(
        2.0 * central_half_extent_m,
        outer_extent_m,
    )

    def interval_edges(
        left_m: float,
        right_m: float,
        max_cell_m: float,
    ) -> np.ndarray:
        n_cells = max(1, int(np.ceil((right_m - left_m) / max_cell_m)))
        return np.linspace(left_m, right_m, n_cells + 1)

    central_edges = interval_edges(
        -central_half_extent_m,
        central_half_extent_m,
        central_cell_size_m,
    )

    transition_positive = interval_edges(
        central_half_extent_m,
        transition_half_extent_m,
        transition_cell_size_m,
    )[1:]

    outer_positive = interval_edges(
        transition_half_extent_m,
        outer_extent_m,
        outer_cell_size_m,
    )[1:]

    axis_edges = np.concatenate(
        [
            -outer_positive[::-1],
            -transition_positive[::-1],
            central_edges,
            transition_positive,
            outer_positive,
        ]
    )
    axis_edges = _unique_sorted_edges(axis_edges)

    panels: list[list[float]] = []
    values: list[float] = []

    for y_bottom_m, y_top_m in zip(axis_edges[:-1], axis_edges[1:]):
        for x_left_m, x_right_m in zip(axis_edges[:-1], axis_edges[1:]):
            x_centre_m = 0.5 * (x_left_m + x_right_m)
            y_centre_m = 0.5 * (y_bottom_m + y_top_m)
            value = classify_cell(x_centre_m, y_centre_m)

            _append_panel(
                panels,
                values,
                x_left_m=x_left_m,
                x_right_m=x_right_m,
                y_bottom_m=y_bottom_m,
                y_top_m=y_top_m,
                value=value,
            )

    return np.asarray(panels, dtype=float), np.asarray(values, dtype=float)


def edge_aligned_sparse_partition(
    *,
    outer_extent_m: float,
    critical_positions_m: np.ndarray | list[float],
    central_half_extent_m: float = 220e-6,
    central_cell_size_m: float = 15e-6,
    transition_cell_size_m: float = 40e-6,
    outer_cell_size_m: float = 150e-6,
    classify_cell,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse panel partition with physical critical edges inserted.

    The returned axis is nonuniform and contains every value in
    ``critical_positions_m`` and its mirror image. It uses fine cells around
    the centre and coarser cells outside. The mesh is still a rectangular
    tensor grid, but is significantly smaller than the previous global
    edge-aligned construction for ordinary settings.

    Returns
    -------
    panels_m:
        Shape (N, 4), panels [x_left, x_right, y_bottom, y_top].
    voltages_v:
        Shape (N,), boundary voltage per panel.
    axis_edges_m:
        Shared x/y mesh edges, useful for visualisation.
    """
    if outer_extent_m <= 0.0:
        raise ValueError("outer_extent_m must be positive.")
    if not 0.0 < central_half_extent_m < outer_extent_m:
        raise ValueError(
            "central_half_extent_m must lie strictly inside outer_extent_m."
        )
    if min(
        central_cell_size_m,
        transition_cell_size_m,
        outer_cell_size_m,
    ) <= 0.0:
        raise ValueError("All cell sizes must be positive.")

    critical = np.asarray(critical_positions_m, dtype=float)
    if not np.all(np.isfinite(critical)):
        raise ValueError("critical_positions_m must contain finite values.")

    transition_half_extent_m = min(
        2.0 * central_half_extent_m,
        outer_extent_m,
    )

    positive_critical = np.abs(critical)
    positive_critical = positive_critical[
        positive_critical < outer_extent_m
    ]

    positive_nodes = _unique_sorted_edges(
        np.concatenate(
            [
                np.array(
                    [
                        0.0,
                        central_half_extent_m,
                        transition_half_extent_m,
                        outer_extent_m,
                    ],
                    dtype=float,
                ),
                positive_critical,
            ]
        )
    )

    positive_edges: list[float] = [0.0]

    for left_m, right_m in zip(positive_nodes[:-1], positive_nodes[1:]):
        midpoint_m = 0.5 * (left_m + right_m)

        if midpoint_m <= central_half_extent_m:
            max_cell_m = central_cell_size_m
        elif midpoint_m <= transition_half_extent_m:
            max_cell_m = transition_cell_size_m
        else:
            max_cell_m = outer_cell_size_m

        n_cells = max(1, int(np.ceil((right_m - left_m) / max_cell_m)))
        positive_edges.extend(
            np.linspace(left_m, right_m, n_cells + 1)[1:].tolist()
        )

    positive = _unique_sorted_edges(positive_edges)
    axis_edges_m = np.concatenate([-positive[:0:-1], positive])

    panels: list[list[float]] = []
    voltages: list[float] = []

    for y_bottom_m, y_top_m in zip(axis_edges_m[:-1], axis_edges_m[1:]):
        for x_left_m, x_right_m in zip(axis_edges_m[:-1], axis_edges_m[1:]):
            x_centre_m = 0.5 * (x_left_m + x_right_m)
            y_centre_m = 0.5 * (y_bottom_m + y_top_m)

            voltage = classify_cell(x_centre_m, y_centre_m)

            _append_panel(
                panels,
                voltages,
                x_left_m=x_left_m,
                x_right_m=x_right_m,
                y_bottom_m=y_bottom_m,
                y_top_m=y_top_m,
                value=voltage,
            )

    return (
        np.asarray(panels, dtype=float),
        np.asarray(voltages, dtype=float),
        axis_edges_m,
    )

def quadtree_rectangular_panels(
    *,
    outer_extent_m: float,
    classify_point,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 12.5e-6,
    boundary_max_cell_m: float = 15e-6,
    outer_max_cell_m: float = 150e-6,
    min_cell_m: float = 5e-6,
    boundary_probe_fraction: float = 0.24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an adaptive quadtree mesh for a binary electrode layout.

    Parameters
    ----------
    outer_extent_m:
        The mesh covers [-outer_extent_m, outer_extent_m]^2.
    classify_point:
        Callable ``f(x_m, y_m) -> float`` returning the boundary voltage at
        one location, typically 0.0 for ground and 1.0 for RF.
    central_half_extent_m:
        The central square is recursively refined until panel size is at
        most ``central_max_cell_m``.
    boundary_max_cell_m:
        A mixed RF/ground cell is refined until its panel size is at most
        this value. This resolves electrode boundaries.
    outer_max_cell_m:
        Largest panel size permitted in uniform outer regions.
    min_cell_m:
        Hard lower panel-size limit. It prevents infinite recursion if a
        boundary passes extremely close to a corner.
    boundary_probe_fraction:
        Offset from centre for four interior probes. It must be between 0
        and 0.5.

    Returns
    -------
    panels_m, voltages_v:
        Rectangular panels and voltages suitable for ``BEM2D``.

    Notes
    -----
    This is a geometry-adaptive panel list, not a tensor grid. The BEM
    matrix remains dense, but its number of unknowns scales much more
    favorably for layouts consisting mostly of large grounded regions.
    """
    if outer_extent_m <= 0.0:
        raise ValueError("outer_extent_m must be positive.")
    if not 0.0 < central_half_extent_m < outer_extent_m:
        raise ValueError(
            "central_half_extent_m must lie strictly inside outer_extent_m."
        )
    if min(
        central_max_cell_m,
        boundary_max_cell_m,
        outer_max_cell_m,
        min_cell_m,
    ) <= 0.0:
        raise ValueError("All cell sizes must be positive.")
    if not 0.0 < boundary_probe_fraction < 0.5:
        raise ValueError("boundary_probe_fraction must be in (0, 0.5).")

    panels: list[list[float]] = []
    voltages: list[float] = []

    def classify_samples(
        x_left_m: float,
        x_right_m: float,
        y_bottom_m: float,
        y_top_m: float,
    ) -> np.ndarray:
        x_centre_m = 0.5 * (x_left_m + x_right_m)
        y_centre_m = 0.5 * (y_bottom_m + y_top_m)

        half_x_m = 0.5 * (x_right_m - x_left_m)
        half_y_m = 0.5 * (y_top_m - y_bottom_m)

        offset_x_m = boundary_probe_fraction * 2.0 * half_x_m
        offset_y_m = boundary_probe_fraction * 2.0 * half_y_m

        positions = [
            (x_centre_m, y_centre_m),
            (x_centre_m - offset_x_m, y_centre_m - offset_y_m),
            (x_centre_m - offset_x_m, y_centre_m + offset_y_m),
            (x_centre_m + offset_x_m, y_centre_m - offset_y_m),
            (x_centre_m + offset_x_m, y_centre_m + offset_y_m),
        ]

        return np.asarray(
            [float(classify_point(x_m, y_m)) for x_m, y_m in positions],
            dtype=float,
        )

    def recurse(
        x_left_m: float,
        x_right_m: float,
        y_bottom_m: float,
        y_top_m: float,
    ) -> None:
        width_m = x_right_m - x_left_m
        height_m = y_top_m - y_bottom_m
        cell_size_m = max(width_m, height_m)

        x_centre_m = 0.5 * (x_left_m + x_right_m)
        y_centre_m = 0.5 * (y_bottom_m + y_top_m)

        samples = classify_samples(
            x_left_m,
            x_right_m,
            y_bottom_m,
            y_top_m,
        )

        mixed = bool(np.any(samples != samples[0]))

        in_central_region = (
            abs(x_centre_m) <= central_half_extent_m
            and abs(y_centre_m) <= central_half_extent_m
        )

        target_size_m = outer_max_cell_m
        if in_central_region:
            target_size_m = min(target_size_m, central_max_cell_m)
        if mixed:
            target_size_m = min(target_size_m, boundary_max_cell_m)

        can_split = (
            cell_size_m > target_size_m
            and 0.5 * min(width_m, height_m) >= min_cell_m
        )

        if can_split:
            x_middle_m = 0.5 * (x_left_m + x_right_m)
            y_middle_m = 0.5 * (y_bottom_m + y_top_m)

            recurse(x_left_m, x_middle_m, y_bottom_m, y_middle_m)
            recurse(x_middle_m, x_right_m, y_bottom_m, y_middle_m)
            recurse(x_left_m, x_middle_m, y_middle_m, y_top_m)
            recurse(x_middle_m, x_right_m, y_middle_m, y_top_m)
            return

        _append_panel(
            panels,
            voltages,
            x_left_m=x_left_m,
            x_right_m=x_right_m,
            y_bottom_m=y_bottom_m,
            y_top_m=y_top_m,
            value=float(classify_point(x_centre_m, y_centre_m)),
        )

    recurse(
        -outer_extent_m,
        outer_extent_m,
        -outer_extent_m,
        outer_extent_m,
    )

    return np.asarray(panels, dtype=float), np.asarray(voltages, dtype=float)


def geometry_aware_quadtree_rectangular_panels(
    *,
    outer_extent_m: float,
    classify_point,
    intersects_boundary,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 25e-6,
    boundary_max_cell_m: float = 10e-6,
    outer_max_cell_m: float = 180e-6,
    min_cell_m: float = 5e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build quadtree panels refined by exact geometry-boundary intersection.

    Unlike ``quadtree_rectangular_panels``, this mesher does not infer a
    boundary only from sampled binary mask values. The supplied
    ``intersects_boundary`` callback receives one cell rectangle:

        intersects_boundary(x_left, x_right, y_bottom, y_top) -> bool

    and must return true whenever an exact physical electrode boundary
    crosses that rectangle. Such cells are recursively refined down to
    ``boundary_max_cell_m``. This ensures that both inner and outer RF
    boundaries affect the actual BEM mesh and therefore the GA fitness.

    The resulting panels still use centre-point voltage classification;
    however, the error is bounded by the refined boundary-cell size.
    """
    if outer_extent_m <= 0.0:
        raise ValueError("outer_extent_m must be positive.")
    if not 0.0 < central_half_extent_m < outer_extent_m:
        raise ValueError(
            "central_half_extent_m must lie strictly inside outer_extent_m."
        )
    if min(
        central_max_cell_m,
        boundary_max_cell_m,
        outer_max_cell_m,
        min_cell_m,
    ) <= 0.0:
        raise ValueError("All cell sizes must be positive.")

    panels: list[list[float]] = []
    voltages: list[float] = []

    def recurse(
        x_left_m: float,
        x_right_m: float,
        y_bottom_m: float,
        y_top_m: float,
    ) -> None:
        width_m = x_right_m - x_left_m
        height_m = y_top_m - y_bottom_m
        cell_size_m = max(width_m, height_m)

        x_centre_m = 0.5 * (x_left_m + x_right_m)
        y_centre_m = 0.5 * (y_bottom_m + y_top_m)

        in_central_region = (
            abs(x_centre_m) <= central_half_extent_m
            and abs(y_centre_m) <= central_half_extent_m
        )

        boundary_crosses = bool(
            intersects_boundary(
                x_left_m,
                x_right_m,
                y_bottom_m,
                y_top_m,
            )
        )

        target_size_m = outer_max_cell_m

        if in_central_region:
            target_size_m = min(target_size_m, central_max_cell_m)

        if boundary_crosses:
            target_size_m = min(target_size_m, boundary_max_cell_m)

        can_split = (
            cell_size_m > target_size_m
            and 0.5 * min(width_m, height_m) >= min_cell_m
        )

        if can_split:
            x_middle_m = 0.5 * (x_left_m + x_right_m)
            y_middle_m = 0.5 * (y_bottom_m + y_top_m)

            recurse(x_left_m, x_middle_m, y_bottom_m, y_middle_m)
            recurse(x_middle_m, x_right_m, y_bottom_m, y_middle_m)
            recurse(x_left_m, x_middle_m, y_middle_m, y_top_m)
            recurse(x_middle_m, x_right_m, y_middle_m, y_top_m)
            return

        _append_panel(
            panels,
            voltages,
            x_left_m=x_left_m,
            x_right_m=x_right_m,
            y_bottom_m=y_bottom_m,
            y_top_m=y_top_m,
            value=float(classify_point(x_centre_m, y_centre_m)),
        )

    recurse(
        -outer_extent_m,
        outer_extent_m,
        -outer_extent_m,
        outer_extent_m,
    )

    return np.asarray(panels, dtype=float), np.asarray(voltages, dtype=float)