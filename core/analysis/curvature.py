"""RF-field Jacobian, pseudopotential curvature, and secular frequencies."""

from __future__ import annotations

import numpy as np

from config.numerical import FINITE_DIFFERENCE_STEP_M
from core.analysis.rf_null import evaluate_field
from core.electrostatics.pseudopotential import (
    secular_frequencies_from_field_squared_hessian_hz,
)


def field_jacobian(
    field_model: object,
    position_m: np.ndarray | list[float],
    *,
    step_m: float = FINITE_DIFFERENCE_STEP_M,
) -> np.ndarray:
    """Return J[i, j] = dE_i / dr_j by central finite differences."""
    position = np.asarray(position_m, dtype=float)

    if position.shape != (3,):
        raise ValueError("position_m must have shape (3,).")
    if step_m <= 0.0:
        raise ValueError("step_m must be positive.")
    if position[2] - step_m <= 0.0:
        raise ValueError("Finite-difference step crosses electrode plane.")

    jacobian = np.empty((3, 3), dtype=float)

    for axis in range(3):
        displacement = np.zeros(3, dtype=float)
        displacement[axis] = step_m

        field_plus = evaluate_field(field_model, position + displacement)
        field_minus = evaluate_field(field_model, position - displacement)

        jacobian[:, axis] = (field_plus - field_minus) / (2.0 * step_m)

    return jacobian


def field_squared_hessian_at_null(
    field_model: object,
    position_m: np.ndarray | list[float],
    *,
    step_m: float = FINITE_DIFFERENCE_STEP_M,
) -> np.ndarray:
    """Return Hessian of |E_RF|² at an RF-null, in V²/m⁴."""
    jacobian = field_jacobian(
        field_model,
        position_m,
        step_m=step_m,
    )
    return 2.0 * jacobian.T @ jacobian


def secular_frequencies_at_null_hz(
    field_model: object,
    position_m: np.ndarray | list[float],
    *,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
    step_m: float = FINITE_DIFFERENCE_STEP_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies, Hessian eigenvalues, and principal axes."""
    hessian = field_squared_hessian_at_null(
        field_model,
        position_m,
        step_m=step_m,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)

    frequencies_hz, _ = secular_frequencies_from_field_squared_hessian_hz(
        hessian_unit_v2_m4=hessian,
        rf_voltage_peak_v=rf_voltage_peak_v,
        rf_angular_frequency_rad_s=rf_angular_frequency_rad_s,
    )

    return frequencies_hz, eigenvalues, eigenvectors