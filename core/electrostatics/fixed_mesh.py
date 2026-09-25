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
    probe_x = np.asarray([x_left, x_right, x_left, x_right, 0.5 * (x_left + x_right)])
    probe_y = np.asarray([y_bottom, y_bottom, y_top, y_top, 0.5 * (y_bottom + y_top)])
    reference = baseline(cfg)
    values = rf_mask(reference, probe_x, probe_y, cfg)
    return bool(np.any(values > 0.5) and np.any(values < 0.5))


def geometry_aware_quadtree_rectangular_panels(
    outer_extent_m: float,
    classify_point: Callable[[float, float], float],
    intersects_boundary: Callable[[float, float, float, float], bool],
    central_half_extent_m: float,
    central_max_cell_m: float,
    boundary_max_cell_m: float,
    outer_max_cell_m: float,
    min_cell_m: float,
) -> tuple[np.ndarray, None]:
    del central_half_extent_m, boundary_max_cell_m

    step = min(outer_max_cell_m, central_max_cell_m)
    step = max(step, min_cell_m)
    coordinates = np.arange(-outer_extent_m + 0.5 * step, outer_extent_m, step)
    centres: list[tuple[float, float]] = []

    for x in coordinates:
        for y in coordinates:
            x_left = x - 0.5 * step
            x_right = x + 0.5 * step
            y_bottom = y - 0.5 * step
            y_top = y + 0.5 * step
            if intersects_boundary(x_left, x_right, y_bottom, y_top) or classify_point(x, y) >= 0.0:
                centres.append((float(x), float(y)))

    return np.asarray(centres, dtype=float), None


def build_fixed_mesh(cfg) -> np.ndarray:
    reference = baseline(cfg)

    def classify_point(x: float, y: float) -> float:
        return float(
            rf_mask(
                reference,
                np.asarray([x]),
                np.asarray([y]),
                cfg,
            )[0]
        )

    def intersects_boundary(
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
    ) -> bool:
        return baseline_boundary_intersects(
            x_left,
            x_right,
            y_bottom,
            y_top,
            cfg,
        )

    panels, _ = geometry_aware_quadtree_rectangular_panels(
        outer_extent_m=cfg.outer_extent_m,
        classify_point=classify_point,
        intersects_boundary=intersects_boundary,
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