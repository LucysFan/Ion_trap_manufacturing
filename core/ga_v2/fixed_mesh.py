"""core/ga_v2/fixed_mesh.py

Vectorized fixed-mesh construction. Drop-in replacement for
core.ga.fixed_mesh with identical behaviour and interface.

The only change is implementation: the original module built the mesh
with a nested Python loop that invoked rf_mask once per probe point per
cell (roughly 6 * N^2 separate calls). This module evaluates the entire
probe set in a single batched rf_mask call and cell centres in a second
batched call, so the total is exactly two calls regardless of grid size.

This is not the dominant cost in the GA pipeline -- the dominant cost is
the O(N^2) BEM matrix assembly in fixed_bem.py -- but it removes the
per-cell Python overhead for anyone who still uses build_fixed_mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.ga.candidates import Candidate
from core.geometry.rf_mask import rf_mask


@dataclass
class MeshPanels:
    centres_m: np.ndarray

    def __len__(self) -> int:
        return int(len(self.centres_m))


@dataclass
class BaselineGeometry:
    candidate: Candidate


def baseline(cfg) -> Candidate:
    return Candidate(
        inner_m=np.full(8, cfg.base_inner_m),
        outer_m=np.full(8, cfg.base_outer_m),
        topology=0,
    )


def baseline_boundary_intersects(
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    cfg,
) -> bool:
    """Scalar version kept for backward compatibility."""
    reference = baseline(cfg)
    probe_x = np.asarray(
        [x_left, x_right, x_left, x_right, 0.5 * (x_left + x_right)]
    )
    probe_y = np.asarray(
        [y_bottom, y_bottom, y_top, y_top, 0.5 * (y_bottom + y_top)]
    )
    values = rf_mask(reference, probe_x, probe_y, cfg)
    return bool(np.any(values > 0.5) and np.any(values < 0.5))


def geometry_aware_quadtree_rectangular_panels(
    outer_extent_m: float,
    classify_point: Callable[[np.ndarray, np.ndarray], np.ndarray],
    intersects_boundary: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
    central_half_extent_m: float,
    central_max_cell_m: float,
    boundary_max_cell_m: float,
    outer_max_cell_m: float,
    min_cell_m: float,
) -> tuple[np.ndarray, None]:
    """Vectorized uniform-grid panel builder.

    The original function ignored central_half_extent_m and
    boundary_max_cell_m and produced a uniform grid. Behaviour is
    preserved here exactly:

        step        = max(min(outer_max_cell_m, central_max_cell_m), min_cell_m)
        coordinates = arange(-outer_extent + step/2, outer_extent, step)
        keep        = intersects_boundary(...) OR classify_point(centre) >= 0

    The iteration order matches the original x-major / y-minor loop.
    """
    del central_half_extent_m, boundary_max_cell_m

    step = min(outer_max_cell_m, central_max_cell_m)
    step = max(step, min_cell_m)

    coordinates = np.arange(
        -outer_extent_m + 0.5 * step, outer_extent_m, step
    )

    if coordinates.size == 0:
        return np.zeros((0, 2), dtype=float), None

    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    centres_x = xx.reshape(-1)
    centres_y = yy.reshape(-1)

    half = 0.5 * step
    x_left = centres_x - half
    x_right = centres_x + half
    y_bottom = centres_y - half
    y_top = centres_y + half

    crosses = np.asarray(
        intersects_boundary(x_left, x_right, y_bottom, y_top),
        dtype=bool,
    )
    centre_values = np.asarray(
        classify_point(centres_x, centres_y),
        dtype=float,
    )

    keep = crosses | (centre_values >= 0.0)

    return np.column_stack((centres_x[keep], centres_y[keep])), None


def build_fixed_mesh(cfg) -> np.ndarray:
    """Build the coarse fixed mesh using batched rf_mask calls."""
    reference = baseline(cfg)

    def vectorized_classify(
        x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return rf_mask(reference, x, y, cfg)

    def vectorized_intersects(
        x_left: np.ndarray,
        x_right: np.ndarray,
        y_bottom: np.ndarray,
        y_top: np.ndarray,
    ) -> np.ndarray:
        x_centre = 0.5 * (x_left + x_right)
        y_centre = 0.5 * (y_bottom + y_top)

        probe_x = np.concatenate(
            [x_left, x_right, x_left, x_right, x_centre]
        )
        probe_y = np.concatenate(
            [y_bottom, y_bottom, y_top, y_top, y_centre]
        )

        values = rf_mask(reference, probe_x, probe_y, cfg)
        values = values.reshape(5, x_left.size)

        any_rf = np.any(values > 0.5, axis=0)
        any_ground = np.any(values < 0.5, axis=0)
        return any_rf & any_ground

    panels, _ = geometry_aware_quadtree_rectangular_panels(
        outer_extent_m=cfg.outer_extent_m,
        classify_point=vectorized_classify,
        intersects_boundary=vectorized_intersects,
        central_half_extent_m=cfg.central_half_extent_m,
        central_max_cell_m=cfg.central_max_cell_m,
        boundary_max_cell_m=cfg.boundary_max_cell_m,
        outer_max_cell_m=cfg.outer_max_cell_m,
        min_cell_m=cfg.min_cell_m,
    )

    if len(panels) > cfg.max_panels:
        raise RuntimeError(
            f"Mesh has {len(panels)} panels, "
            f"exceeding --max-panels={cfg.max_panels}. "
            "Increase mesh cell sizes or raise the panel limit "
            "only if you have enough RAM."
        )

    return panels


__all__ = [
    "baseline",
    "baseline_boundary_intersects",
    "build_fixed_mesh",
    "geometry_aware_quadtree_rectangular_panels",
]