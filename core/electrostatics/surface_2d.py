"""Analytic 2D surface-electrode trap model.

The model assumes infinitely long electrode strips in the y direction,
an electrode plane at z = 0, zero-width gaps, and zero metal thickness.
It is a fast baseline model for a straight surface-electrode trap arm.
"""

from __future__ import annotations

import numpy as np


class Surface2D:
    """Piecewise-constant electrode voltages on an infinite planar surface."""

    def __init__(
        self,
        edges_m: np.ndarray | list[float],
        voltages_v: np.ndarray | list[float],
    ) -> None:
        self.edges_m = np.asarray(edges_m, dtype=float)
        self.voltages_v = np.asarray(voltages_v, dtype=float)

        if self.edges_m.ndim != 1 or self.voltages_v.ndim != 1:
            raise ValueError("edges_m and voltages_v must be one-dimensional.")
        if len(self.edges_m) != len(self.voltages_v) + 1:
            raise ValueError("len(edges_m) must equal len(voltages_v) + 1.")
        if np.any(np.diff(self.edges_m) <= 0.0):
            raise ValueError("Electrode edges must be strictly increasing.")

    def potential(self, x_m: float, z_m: float) -> float:
        """Return electrostatic potential in volts."""
        if z_m <= 0.0:
            raise ValueError("z_m must be positive.")

        u = self.edges_m - float(x_m)
        contribution = np.arctan(u[1:] / z_m) - np.arctan(u[:-1] / z_m)
        return float(np.dot(self.voltages_v, contribution) / np.pi)

    def electric_field(self, x_m: float, z_m: float) -> tuple[float, float]:
        """Return (E_x, E_z) in V/m."""
        if z_m <= 0.0:
            raise ValueError("z_m must be positive.")

        u = self.edges_m - float(x_m)
        denominator = u * u + z_m * z_m

        e_x = np.dot(
            self.voltages_v,
            z_m / denominator[1:] - z_m / denominator[:-1],
        ) / np.pi

        e_z = np.dot(
            self.voltages_v,
            u[1:] / denominator[1:] - u[:-1] / denominator[:-1],
        ) / np.pi

        return float(e_x), float(e_z)

    def field_jacobian(self, x_m: float, z_m: float) -> np.ndarray:
        """Return d(E_x, E_z)/d(x, z) as a 2x2 matrix in V/m²."""
        if z_m <= 0.0:
            raise ValueError("z_m must be positive.")

        u = self.edges_m - float(x_m)
        denominator = u * u + z_m * z_m
        numerator = u * u - z_m * z_m

        d_ex_dx = np.dot(
            self.voltages_v,
            2.0 * z_m * u[1:] / denominator[1:] ** 2
            - 2.0 * z_m * u[:-1] / denominator[:-1] ** 2,
        ) / np.pi

        d_ez_dx = np.dot(
            self.voltages_v,
            numerator[1:] / denominator[1:] ** 2
            - numerator[:-1] / denominator[:-1] ** 2,
        ) / np.pi

        d_ex_dz = d_ez_dx
        d_ez_dz = -d_ex_dx

        return np.array(
            [[d_ex_dx, d_ex_dz], [d_ez_dx, d_ez_dz]],
            dtype=float,
        )

    def hessian_field_squared_at_null(self, x_m: float, z_m: float) -> np.ndarray:
        """Return Hessian of |E|² at an RF-null in units of V²/m⁴."""
        jacobian = self.field_jacobian(x_m, z_m)
        return 2.0 * jacobian.T @ jacobian

    def verify_consistency(
        self,
        x_m: float,
        z_m: float,
        *,
        finite_difference_step_m: float = 1e-8,
        tolerance: float = 1e-4,
    ) -> bool:
        """Check analytic field identities against central differences."""
        step = finite_difference_step_m

        ex_numeric = -(
            self.potential(x_m + step, z_m)
            - self.potential(x_m - step, z_m)
        ) / (2.0 * step)

        ez_numeric = -(
            self.potential(x_m, z_m + step)
            - self.potential(x_m, z_m - step)
        ) / (2.0 * step)

        ex, ez = self.electric_field(x_m, z_m)
        jacobian = self.field_jacobian(x_m, z_m)

        checks = (
            np.isclose(ex, ex_numeric, rtol=tolerance, atol=tolerance),
            np.isclose(ez, ez_numeric, rtol=tolerance, atol=tolerance),
            np.isclose(jacobian[0, 1], jacobian[1, 0], rtol=tolerance),
            np.isclose(jacobian[0, 0], -jacobian[1, 1], rtol=tolerance),
        )
        return bool(all(checks))


def build_linear_surface_trap(
    ground_width_m: float,
    rf_width_m: float,
    outer_extent_m: float | None = None,
) -> Surface2D:
    """Build a symmetric five-strip surface-electrode trap cross-section.

    The central and outer electrodes are grounded. The two electrodes
    adjacent to the central ground rail carry a unit RF potential.
    """
    if ground_width_m <= 0.0 or rf_width_m <= 0.0:
        raise ValueError("ground_width_m and rf_width_m must be positive.")

    if outer_extent_m is None:
        outer_extent_m = 20.0 * (ground_width_m + rf_width_m)

    minimum_extent_m = 0.5 * ground_width_m + rf_width_m
    if outer_extent_m <= minimum_extent_m:
        raise ValueError("outer_extent_m is too small for the electrode layout.")

    half_ground = 0.5 * ground_width_m
    edges_m = np.array(
        [
            -outer_extent_m,
            -half_ground - rf_width_m,
            -half_ground,
            half_ground,
            half_ground + rf_width_m,
            outer_extent_m,
        ],
        dtype=float,
    )
    voltages_v = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    return Surface2D(edges_m=edges_m, voltages_v=voltages_v)