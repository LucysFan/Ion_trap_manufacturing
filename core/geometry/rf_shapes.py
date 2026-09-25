from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def spline_values(values_m: np.ndarray, s: np.ndarray) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, len(values_m))
    spline = CubicSpline(knots, values_m, bc_type="natural")
    return np.asarray(spline(np.clip(s, 0.0, 1.0)), dtype=float)


def rectangle_mask(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    dx = x - cx
    dy = y - cy
    xr = cosine * dx + sine * dy
    yr = -sine * dx + cosine * dy
    return (np.abs(xr) <= width / 2.0) & (np.abs(yr) <= height / 2.0)


__all__ = ["rectangle_mask", "spline_values"]