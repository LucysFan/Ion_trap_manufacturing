"""Continuation tracing of a transverse RF minimum through an X-junction.

For transport along a prescribed longitudinal coordinate x, the ion is
transversely confined by the RF pseudopotential. At every fixed x, this
module solves E_y = E_z = 0 for y and z. The solution from the preceding
point is used as the initial condition for the next point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from config.numerical import (
    MAX_NULL_TRANSVERSE_SHIFT_M,
    RF_NULL_MAX_HEIGHT_M,
    RF_NULL_MIN_HEIGHT_M,
)


@dataclass(frozen=True)
class RFNullTrace:
    """Result of tracing a transverse RF minimum along one transport arm."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    transverse_residual_v_m: np.ndarray
    full_field_norm_v_m: np.ndarray
    converged: np.ndarray
    messages: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """True only when every requested path point converged."""
        return bool(np.all(self.converged))

    @property
    def positions_m(self) -> np.ndarray:
        """Return trace positions with shape (n_points, 3)."""
        return np.column_stack([self.x_m, self.y_m, self.z_m])

    @property
    def first_failure_index(self) -> int | None:
        """Return first failed index, or None if trace is complete."""
        failures = np.flatnonzero(~self.converged)
        return None if len(failures) == 0 else int(failures[0])


def _field_vector(
    field_model: object,
    x_m: float,
    y_m: float,
    z_m: float,
) -> np.ndarray:
    """Evaluate and validate a three-component RF field."""
    field = np.asarray(
        field_model.electric_field(float(x_m), float(y_m), float(z_m)),
        dtype=float,
    )

    if field.shape != (3,) or not np.all(np.isfinite(field)):
        raise ValueError(
            "field_model.electric_field must return three finite components."
        )

    return field


def _transverse_residual(
    yz_m: np.ndarray,
    *,
    field_model: object,
    x_m: float,
) -> np.ndarray:
    """Return exactly the physical transverse residual (E_y, E_z)."""
    y_m, z_m = yz_m

    if z_m <= 0.0:
        return np.array(
            [
                y_m * 1e12,
                (z_m - RF_NULL_MIN_HEIGHT_M) * 1e12,
            ],
            dtype=float,
        )

    field = _field_vector(field_model, x_m, y_m, z_m)
    return np.array([field[1], field[2]], dtype=float)


def trace_rf_transverse_minimum(
    field_model: object,
    x_m: np.ndarray | list[float],
    *,
    initial_y_m: float = 0.0,
    initial_z_m: float,
    residual_tolerance_v_m: float = 1e-3,
    min_height_m: float = RF_NULL_MIN_HEIGHT_M,
    max_height_m: float = RF_NULL_MAX_HEIGHT_M,
    max_transverse_shift_m: float = MAX_NULL_TRANSVERSE_SHIFT_M,
) -> RFNullTrace:
    """Trace transverse RF minimum at prescribed transport coordinates.

    The path need not be a full 3D RF-null: at fixed x, only E_y and E_z
    are constrained to zero. A longitudinal field E_x may remain nonzero,
    and it becomes part of the RF pseudopotential proxy/barrier diagnostics.
    """
    x_values_m = np.asarray(x_m, dtype=float)

    if x_values_m.ndim != 1 or len(x_values_m) < 2:
        raise ValueError(
            "x_m must be a one-dimensional array with at least two points."
        )
    if not np.all(np.isfinite(x_values_m)):
        raise ValueError("x_m must contain only finite values.")
    if initial_z_m <= 0.0:
        raise ValueError("initial_z_m must be positive.")
    if residual_tolerance_v_m <= 0.0:
        raise ValueError("residual_tolerance_v_m must be positive.")
    if not 0.0 < min_height_m < max_height_m:
        raise ValueError("Invalid allowed height interval.")
    if max_transverse_shift_m <= 0.0:
        raise ValueError("max_transverse_shift_m must be positive.")

    n_points = len(x_values_m)

    y_values_m = np.full(n_points, np.nan, dtype=float)
    z_values_m = np.full(n_points, np.nan, dtype=float)
    transverse_residuals_v_m = np.full(n_points, np.inf, dtype=float)
    full_norms_v_m = np.full(n_points, np.inf, dtype=float)
    converged = np.zeros(n_points, dtype=bool)
    messages: list[str] = []

    previous_yz_m = np.array([initial_y_m, initial_z_m], dtype=float)

    for index, x_value_m in enumerate(x_values_m):
        y_reference_m = float(previous_yz_m[0])

        result = least_squares(
            _transverse_residual,
            x0=previous_yz_m,
            bounds=(
                np.array(
                    [
                        y_reference_m - max_transverse_shift_m,
                        min_height_m,
                    ]
                ),
                np.array(
                    [
                        y_reference_m + max_transverse_shift_m,
                        max_height_m,
                    ]
                ),
            ),
            kwargs={
                "field_model": field_model,
                "x_m": float(x_value_m),
            },
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )

        y_value_m, z_value_m = result.x
        field = _field_vector(
            field_model,
            float(x_value_m),
            y_value_m,
            z_value_m,
        )

        transverse_norm_v_m = float(np.linalg.norm(field[1:]))
        full_norm_v_m = float(np.linalg.norm(field))
        yz_step_m = float(np.linalg.norm(result.x - previous_yz_m))

        accepted = bool(
            result.success
            and np.isfinite(transverse_norm_v_m)
            and transverse_norm_v_m <= residual_tolerance_v_m
            and min_height_m <= z_value_m <= max_height_m
            and yz_step_m <= max_transverse_shift_m
        )

        y_values_m[index] = y_value_m
        z_values_m[index] = z_value_m
        transverse_residuals_v_m[index] = transverse_norm_v_m
        full_norms_v_m[index] = full_norm_v_m
        converged[index] = accepted

        message = (
            f"x={x_value_m * 1e6:.3f} um: {result.message}; "
            f"|E_yz|={transverse_norm_v_m:.3e} V/m; "
            f"|E|={full_norm_v_m:.3e} V/m; "
            f"step_yz={yz_step_m * 1e6:.3f} um"
        )
        if not accepted:
            message += "; rejected"
        messages.append(message)

        if not accepted:
            break

        previous_yz_m = result.x

    return RFNullTrace(
        x_m=x_values_m,
        y_m=y_values_m,
        z_m=z_values_m,
        transverse_residual_v_m=transverse_residuals_v_m,
        full_field_norm_v_m=full_norms_v_m,
        converged=converged,
        messages=tuple(messages),
    )