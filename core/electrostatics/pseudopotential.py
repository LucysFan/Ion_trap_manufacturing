"""RF pseudopotential and secular-frequency utilities."""

from __future__ import annotations

import numpy as np

from config.physical_constants import CHARGE_CA40_C, MASS_CA40_KG


def pseudopotential_energy_j(
    electric_field_unit_v_m: np.ndarray,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
) -> float:
    """Return RF pseudopotential energy for a singly charged 40Ca+ ion.

    ``electric_field_unit_v_m`` is the field from the geometry at unit
    RF peak voltage. The actual field is V_RF times this vector.
    """
    electric_field = np.asarray(electric_field_unit_v_m, dtype=float)
    if electric_field.shape != (3,):
        raise ValueError("electric_field_unit_v_m must have shape (3,).")
    if rf_voltage_peak_v < 0.0:
        raise ValueError("rf_voltage_peak_v must be non-negative.")
    if rf_angular_frequency_rad_s <= 0.0:
        raise ValueError("rf_angular_frequency_rad_s must be positive.")

    actual_field_squared = (rf_voltage_peak_v**2) * float(
        np.dot(electric_field, electric_field)
    )

    return (
        CHARGE_CA40_C**2
        * actual_field_squared
        / (4.0 * MASS_CA40_KG * rf_angular_frequency_rad_s**2)
    )


def pseudopotential_energy_ev(
    electric_field_unit_v_m: np.ndarray,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
) -> float:
    """Return RF pseudopotential energy in electronvolts."""
    energy_j = pseudopotential_energy_j(
        electric_field_unit_v_m,
        rf_voltage_peak_v,
        rf_angular_frequency_rad_s,
    )
    return energy_j / CHARGE_CA40_C


def secular_frequencies_from_field_squared_hessian_hz(
    hessian_unit_v2_m4: np.ndarray,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate secular frequencies from Hessian of |E_unit|² at an RF-null."""
    hessian = np.asarray(hessian_unit_v2_m4, dtype=float)
    if hessian.shape != (3, 3):
        raise ValueError("hessian_unit_v2_m4 must have shape (3, 3).")

    eigenvalues = np.linalg.eigvalsh(hessian)
    prefactor = (
        CHARGE_CA40_C
        * rf_voltage_peak_v
        / (2.0 * MASS_CA40_KG * rf_angular_frequency_rad_s)
    )
    angular_frequencies = prefactor * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return angular_frequencies / (2.0 * np.pi), eigenvalues