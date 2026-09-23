"""Utilities for analysing the analytic straight surface-trap section."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from config.physical_constants import CHARGE_CA40_C, MASS_CA40_KG, TWO_PI
from core.electrostatics.surface_2d import Surface2D


def find_rf_null_on_axis(
    trap: Surface2D,
    *,
    x_m: float = 0.0,
    z_min_m: float = 1e-6,
    z_max_m: float = 2e-3,
    n_scan: int = 200,
) -> tuple[float, float]:
    """Find the symmetry-axis RF-null as a root of E_z(x=0, z)."""
    if z_min_m <= 0.0 or z_max_m <= z_min_m:
        raise ValueError("Invalid z search interval.")
    if n_scan < 2:
        raise ValueError("n_scan must be at least 2.")

    z_grid = np.geomspace(z_min_m, z_max_m, n_scan)
    e_z = np.array([trap.electric_field(x_m, z)[1] for z in z_grid])

    for z_left, z_right, e_left, e_right in zip(
        z_grid[:-1],
        z_grid[1:],
        e_z[:-1],
        e_z[1:],
    ):
        if e_left == 0.0:
            return float(x_m), float(z_left)
        if e_left * e_right < 0.0:
            z_root = brentq(
                lambda z: trap.electric_field(x_m, z)[1],
                float(z_left),
                float(z_right),
                xtol=1e-14,
                rtol=1e-14,
            )
            return float(x_m), float(z_root)

    raise RuntimeError("No RF-null found in the requested z interval.")


def secular_frequencies_hz(
    trap: Surface2D,
    x_m: float,
    z_m: float,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial secular frequencies and Hessian eigenvalues."""
    if rf_voltage_peak_v <= 0.0:
        raise ValueError("rf_voltage_peak_v must be positive.")
    if rf_angular_frequency_rad_s <= 0.0:
        raise ValueError("rf_angular_frequency_rad_s must be positive.")

    hessian = trap.hessian_field_squared_at_null(x_m, z_m)
    eigenvalues = np.linalg.eigvalsh(hessian)

    prefactor = (
        CHARGE_CA40_C
        * rf_voltage_peak_v
        / (2.0 * MASS_CA40_KG * rf_angular_frequency_rad_s)
    )
    angular_frequencies = prefactor * np.sqrt(np.abs(eigenvalues))
    return angular_frequencies / TWO_PI, eigenvalues


def required_rf_voltage_peak_v(
    trap: Surface2D,
    x_m: float,
    z_m: float,
    target_frequency_hz: float,
    rf_angular_frequency_rad_s: float,
    *,
    mode_index: int = 0,
) -> tuple[float, np.ndarray]:
    """Return RF peak voltage required to obtain one target radial frequency."""
    if target_frequency_hz <= 0.0:
        raise ValueError("target_frequency_hz must be positive.")

    hessian = trap.hessian_field_squared_at_null(x_m, z_m)
    eigenvalues = np.sort(np.linalg.eigvalsh(hessian))
    selected = float(np.abs(eigenvalues[mode_index]))

    if selected <= 0.0:
        return np.inf, eigenvalues

    voltage = (
        TWO_PI
        * 2.0
        * MASS_CA40_KG
        * rf_angular_frequency_rad_s
        * target_frequency_hz
        / (CHARGE_CA40_C * np.sqrt(selected))
    )
    return float(voltage), eigenvalues