from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


MODE_LINEAR = "linear"
MODE_SPLINE = "spline"

SUPPORTED_MODES: tuple[str, ...] = (
    MODE_LINEAR,
    MODE_SPLINE,
)


def normalized_arm_coordinate(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    arm_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)

    ax = np.abs(x)
    ay = np.abs(y)

    along_x = ax >= ay
    longitudinal_m = np.where(along_x, ax, ay)
    transverse_m = np.where(along_x, ay, ax)

    if arm_length_m <= 0.0:
        raise ValueError("arm_length_m must be positive")

    s = longitudinal_m / arm_length_m
    return s, longitudinal_m, transverse_m, along_x


def evaluate_boundary(
    control_values_m: np.ndarray,
    s: np.ndarray,
    *,
    mode: str = MODE_SPLINE,
    clamp: bool = True,
) -> np.ndarray:
    values = np.asarray(control_values_m, dtype=np.float64).reshape(-1)
    coordinate = np.asarray(s, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("control_values_m must be one-dimensional")

    if values.size < 2:
        raise ValueError("At least two control values are required")

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported boundary mode: {mode}")

    sample = np.clip(coordinate, 0.0, 1.0) if clamp else coordinate
    knots = np.linspace(0.0, 1.0, values.size)

    if mode == MODE_LINEAR:
        return np.interp(sample, knots, values)

    spline = CubicSpline(knots, values, bc_type="natural", extrapolate=True)
    return spline(sample)


def evaluate_inner_outer_boundaries(
    inner_control_m: np.ndarray,
    outer_control_m: np.ndarray,
    s: np.ndarray,
    *,
    mode: str = MODE_SPLINE,
    clamp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    inner = evaluate_boundary(
        inner_control_m,
        s,
        mode=mode,
        clamp=clamp,
    )
    outer = evaluate_boundary(
        outer_control_m,
        s,
        mode=mode,
        clamp=clamp,
    )
    return inner, outer