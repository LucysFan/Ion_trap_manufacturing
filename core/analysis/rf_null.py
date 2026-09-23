"""Local three-dimensional RF-null search for field models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from config.numerical import (
    RF_NULL_MAX_HEIGHT_M,
    RF_NULL_MIN_HEIGHT_M,
    RF_NULL_ROOT_TOL_V_M,
)


@dataclass(frozen=True)
class RFNullResult:
    """Result of a local RF-null solve."""

    position_m: np.ndarray
    residual_field_v_m: float
    converged: bool
    message: str
    n_function_evaluations: int

    @property
    def x_m(self) -> float:
        return float(self.position_m[0])

    @property
    def y_m(self) -> float:
        return float(self.position_m[1])

    @property
    def z_m(self) -> float:
        return float(self.position_m[2])


def evaluate_field(
    field_model: object,
    position_m: np.ndarray,
) -> np.ndarray:
    """Return a finite 3-vector from a model exposing electric_field."""
    position = np.asarray(position_m, dtype=float)

    if position.shape != (3,):
        raise ValueError("position_m must have shape (3,).")

    field = np.asarray(
        field_model.electric_field(
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ),
        dtype=float,
    )

    if field.shape != (3,) or not np.all(np.isfinite(field)):
        raise ValueError(
            "field_model.electric_field must return three finite components."
        )

    return field


def find_rf_null(
    field_model: object,
    initial_guess_m: np.ndarray | list[float],
    *,
    residual_tolerance_v_m: float = RF_NULL_ROOT_TOL_V_M,
    min_height_m: float = RF_NULL_MIN_HEIGHT_M,
    max_height_m: float = RF_NULL_MAX_HEIGHT_M,
) -> RFNullResult:
    """Find a local 3D root of the RF electric field."""
    guess = np.asarray(initial_guess_m, dtype=float)

    if guess.shape != (3,):
        raise ValueError("initial_guess_m must have shape (3,).")
    if not 0.0 < min_height_m < max_height_m:
        raise ValueError("Invalid allowed RF-null height interval.")

    def residual(position_m: np.ndarray) -> np.ndarray:
        if position_m[2] <= 0.0:
            return np.array(
                [
                    position_m[0] * 1e12,
                    position_m[1] * 1e12,
                    (position_m[2] - min_height_m) * 1e12,
                ]
            )
        return evaluate_field(field_model, position_m)

    solution = root(residual, guess, method="hybr")
    position_m = np.asarray(solution.x, dtype=float)

    try:
        residual_norm_v_m = float(np.linalg.norm(residual(position_m)))
    except ValueError:
        residual_norm_v_m = np.inf

    height_valid = bool(
        np.isfinite(position_m[2])
        and min_height_m <= position_m[2] <= max_height_m
    )

    converged = bool(
        solution.success
        and np.all(np.isfinite(position_m))
        and np.isfinite(residual_norm_v_m)
        and residual_norm_v_m <= residual_tolerance_v_m
        and height_valid
    )

    message = str(solution.message)

    if not height_valid:
        message += (
            f"; height z={position_m[2] * 1e6:.3f} um outside "
            f"[{min_height_m * 1e6:.3f}, {max_height_m * 1e6:.3f}] um"
        )
    elif residual_norm_v_m > residual_tolerance_v_m:
        message += (
            f"; residual |E|={residual_norm_v_m:.3e} V/m exceeds "
            f"{residual_tolerance_v_m:.3e} V/m"
        )

    return RFNullResult(
        position_m=position_m,
        residual_field_v_m=residual_norm_v_m,
        converged=converged,
        message=message,
        n_function_evaluations=int(getattr(solution, "nfev", 0)),
    )