"""Movable-knot geometry adapter for the conventional open-centre C4v X-junction.

The project template already accepts piecewise-linear contour knots. This module
adds a safe parameterization in which the longitudinal knot positions are also
variables. Inner and outer contours share the same knot x/s positions, which
preserves an ordinary RF rail rather than two unrelated polygonal edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import XJunctionParameters, make_house_style_x_junction


@dataclass(frozen=True)
class MovableKnotContour:
    """Validated common-knot inner/outer contour in a finite junction core."""

    knots_m: np.ndarray
    inner_offsets_m: np.ndarray
    outer_offsets_m: np.ndarray
    lock_m: float
    min_gap_m: float

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots_m, dtype=float)
        inner = np.asarray(self.inner_offsets_m, dtype=float)
        outer = np.asarray(self.outer_offsets_m, dtype=float)
        if knots.ndim != 1 or inner.ndim != 1 or outer.ndim != 1:
            raise ValueError("movable contour arrays must be one-dimensional")
        if len(knots) < 3 or len(knots) != len(inner) or len(knots) != len(outer):
            raise ValueError("knots and both offset arrays need equal length >= 3")
        if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
            raise ValueError("movable contour values must be finite")
        if abs(knots[0]) > 1e-15 or abs(knots[-1] - self.lock_m) > 1e-15:
            raise ValueError("movable contour must start at 0 and end at lock_m")
        if self.lock_m <= 0.0 or self.min_gap_m <= 0.0:
            raise ValueError("lock_m and min_gap_m must be positive")
        if np.any(np.diff(knots) < self.min_gap_m - 1e-15):
            raise ValueError("movable contour knot gap below min_gap_m")
        if abs(inner[0]) > 1e-15 or abs(inner[-1]) > 1e-15:
            raise ValueError("inner offsets must be exactly zero at core endpoints")
        if abs(outer[0]) > 1e-15 or abs(outer[-1]) > 1e-15:
            raise ValueError("outer offsets must be exactly zero at core endpoints")

    @property
    def n_knots(self) -> int:
        return int(len(self.knots_m))


def softmax_gap_knots(
    logits: np.ndarray,
    *,
    lock_m: float,
    min_gap_m: float,
) -> np.ndarray:
    """Turn arbitrary logits into strictly increasing knots [0, lock_m]."""
    q = np.asarray(logits, dtype=float)
    if q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)):
        raise ValueError("gap logits must be a finite one-dimensional array of length >= 2")
    if lock_m <= q.size * min_gap_m:
        raise ValueError("lock_m is too small for requested min_gap_m and interval count")
    stable = q - np.max(q)
    weights = np.exp(stable)
    weights /= np.sum(weights)
    gaps = min_gap_m + (lock_m - q.size*min_gap_m) * weights
    knots = np.r_[0.0, np.cumsum(gaps)]
    knots[-1] = lock_m
    return knots


def fixed_knots_to_logits(knots_m: np.ndarray, *, min_gap_m: float) -> np.ndarray:
    """Produce softmax logits that reproduce supplied valid knot gaps."""
    knots = np.asarray(knots_m, dtype=float)
    gaps = np.diff(knots)
    residual = gaps - min_gap_m
    if np.any(residual <= 0.0):
        raise ValueError("all fixed gaps must be strictly greater than min_gap_m")
    return np.log(residual / np.sum(residual))


def build_movable_knot_parameters(
    *,
    contour: MovableKnotContour,
    inner_edge_shift_at_centre_m: float,
    outer_edge_shift_at_centre_m: float,
    rf_start_radius_override_m: float,
    taper_length_m: float,
    taper_power: float,
    outer_bulge_amplitude_m: float,
    outer_bulge_center_m: float,
    outer_bulge_sigma_m: float,
    arm_length_m: float = 600e-6,
    outer_extent_m: float = 900e-6,
    ion_height_m: float = TARGET_ION_HEIGHT_M,
) -> XJunctionParameters:
    """Build a standard project XJunctionParameters object from movable knots.

    After contour.lock_m the template interpolation returns zero offsets, thus
    each of the four arms returns exactly to its conventional straight rail.
    """
    knots = tuple(np.asarray(contour.knots_m, dtype=float))
    return make_house_style_x_junction(
        ion_height_m=ion_height_m,
        arm_length_m=arm_length_m,
        outer_extent_m=outer_extent_m,
        inner_edge_shift_at_centre_m=float(inner_edge_shift_at_centre_m),
        outer_edge_shift_at_centre_m=float(outer_edge_shift_at_centre_m),
        rf_start_radius_override_m=float(rf_start_radius_override_m),
        taper_length_m=float(taper_length_m),
        taper_power=float(taper_power),
        outer_bulge_amplitude_m=float(outer_bulge_amplitude_m),
        outer_bulge_center_m=float(outer_bulge_center_m),
        outer_bulge_sigma_m=float(outer_bulge_sigma_m),
        inner_contour_knots_m=knots,
        inner_contour_offsets_m=tuple(np.asarray(contour.inner_offsets_m, dtype=float)),
        outer_contour_knots_m=knots,
        outer_contour_offsets_m=tuple(np.asarray(contour.outer_offsets_m, dtype=float)),
    )
