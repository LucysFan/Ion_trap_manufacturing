"""Boundary-element solver for rectangular planar electrode panels.

Each planar panel carries a constant surface-charge density. The solver
enforces prescribed electrode potentials at panel centres and evaluates
potential and electric field above the electrode plane z = 0.

Units:
    panel coordinates: metres
    electrode potentials: volts
    potential: volts
    electric field: V/m
    surface charge density: C/m^2
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from config.numerical import BEM_ASSEMBLY_HEIGHT_M
from config.physical_constants import EPSILON_0_F_M

COULOMB_CONSTANT = 1.0 / (4.0 * np.pi * EPSILON_0_F_M)
MAX_PANELS_WARNING = 2_000


def _antiderivative_potential(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: float,
) -> np.ndarray:
    """Antiderivative of 1/sqrt(x^2 + y^2 + z^2) over x and y."""
    radius_x = np.sqrt(x_m * x_m + z_m * z_m)
    radius_y = np.sqrt(y_m * y_m + z_m * z_m)
    radius = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)

    safe_radius_x = np.where(radius_x > 1e-300, radius_x, 1.0)
    safe_radius_y = np.where(radius_y > 1e-300, radius_y, 1.0)

    term_x = np.where(
        radius_x > 1e-300,
        x_m * np.arcsinh(y_m / safe_radius_x),
        0.0,
    )
    term_y = np.where(
        radius_y > 1e-300,
        y_m * np.arcsinh(x_m / safe_radius_y),
        0.0,
    )
    term_z = -z_m * np.arctan2(x_m * y_m, z_m * radius)

    return term_x + term_y + term_z


def _antiderivative_potential_dz(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: float,
) -> np.ndarray:
    """Derivative with respect to z of the potential antiderivative."""
    radius = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    xz = x_m * x_m + z_m * z_m
    yz = y_m * y_m + z_m * z_m

    term_1 = -x_m * y_m * z_m / (xz * radius)
    term_2 = -x_m * y_m * z_m / (yz * radius)
    term_3 = -np.arctan2(x_m * y_m, z_m * radius)
    term_4 = (
        z_m
        * x_m
        * y_m
        * (x_m * x_m + y_m * y_m + 2.0 * z_m * z_m)
        / (radius * xz * yz)
    )

    return term_1 + term_2 + term_3 + term_4


def rectangular_panel_integral(
    x_m: float,
    y_m: float,
    z_m: float,
    panels_m: np.ndarray,
) -> np.ndarray:
    """Return integral panel dA / r for every rectangular panel."""
    if z_m <= 0.0:
        raise ValueError("Evaluation height z_m must be positive.")

    x_left = panels_m[:, 0]
    x_right = panels_m[:, 1]
    y_bottom = panels_m[:, 2]
    y_top = panels_m[:, 3]

    return (
        _antiderivative_potential(x_m - x_left, y_m - y_bottom, z_m)
        - _antiderivative_potential(x_m - x_left, y_m - y_top, z_m)
        - _antiderivative_potential(x_m - x_right, y_m - y_bottom, z_m)
        + _antiderivative_potential(x_m - x_right, y_m - y_top, z_m)
    )


def _field_x_integral(
    x_m: float,
    y_m: float,
    z_m: float,
    panels_m: np.ndarray,
) -> np.ndarray:
    """Return integral panel (x-x') / r^3 dA for every panel."""
    x_left = panels_m[:, 0]
    x_right = panels_m[:, 1]
    y_bottom = panels_m[:, 2]
    y_top = panels_m[:, 3]

    u_left = x_m - x_right
    u_right = x_m - x_left
    v_bottom = y_m - y_top
    v_top = y_m - y_bottom

    def primitive(u_m: np.ndarray, v_m: np.ndarray) -> np.ndarray:
        return -np.arcsinh(v_m / np.sqrt(u_m * u_m + z_m * z_m))

    return (
        primitive(u_right, v_top)
        - primitive(u_right, v_bottom)
        - primitive(u_left, v_top)
        + primitive(u_left, v_bottom)
    )


def _field_y_integral(
    x_m: float,
    y_m: float,
    z_m: float,
    panels_m: np.ndarray,
) -> np.ndarray:
    """Return integral panel (y-y') / r^3 dA for every panel."""
    x_left = panels_m[:, 0]
    x_right = panels_m[:, 1]
    y_bottom = panels_m[:, 2]
    y_top = panels_m[:, 3]

    u_left = x_m - x_right
    u_right = x_m - x_left
    v_bottom = y_m - y_top
    v_top = y_m - y_bottom

    def primitive(u_m: np.ndarray, v_m: np.ndarray) -> np.ndarray:
        return -np.arcsinh(u_m / np.sqrt(v_m * v_m + z_m * z_m))

    return (
        primitive(u_right, v_top)
        - primitive(u_right, v_bottom)
        - primitive(u_left, v_top)
        + primitive(u_left, v_bottom)
    )


def _field_z_integral(
    x_m: float,
    y_m: float,
    z_m: float,
    panels_m: np.ndarray,
) -> np.ndarray:
    """Return integral panel z/r^3 dA for every panel."""
    x_left = panels_m[:, 0]
    x_right = panels_m[:, 1]
    y_bottom = panels_m[:, 2]
    y_top = panels_m[:, 3]

    u_left = x_m - x_right
    u_right = x_m - x_left
    v_bottom = y_m - y_top
    v_top = y_m - y_bottom

    return -(
        _antiderivative_potential_dz(u_right, v_top, z_m)
        - _antiderivative_potential_dz(u_right, v_bottom, z_m)
        - _antiderivative_potential_dz(u_left, v_top, z_m)
        + _antiderivative_potential_dz(u_left, v_bottom, z_m)
    )


def rectangular_panel_self_integral(width_m: float, height_m: float) -> float:
    """Return singular self-panel integral evaluated in the panel plane."""
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Panel width and height must be positive.")

    return float(
        2.0 * width_m * np.arcsinh(height_m / width_m)
        + 2.0 * height_m * np.arcsinh(width_m / height_m)
    )


class BEM2D:
    """Constant-charge BEM model for rectangular electrode panels."""

    def __init__(
        self,
        panels_m: np.ndarray,
        electrode_voltages_v: np.ndarray,
        *,
        assembly_height_m: float = BEM_ASSEMBLY_HEIGHT_M,
    ) -> None:
        panels = np.asarray(panels_m, dtype=float)
        voltages = np.asarray(electrode_voltages_v, dtype=float)

        if panels.ndim != 2 or panels.shape[1] != 4:
            raise ValueError("panels_m must have shape (n_panels, 4).")
        if voltages.shape != (len(panels),):
            raise ValueError(
                "electrode_voltages_v must have shape (n_panels,)."
            )
        if np.any(panels[:, 1] <= panels[:, 0]):
            raise ValueError("Every panel must have x_right > x_left.")
        if np.any(panels[:, 3] <= panels[:, 2]):
            raise ValueError("Every panel must have y_top > y_bottom.")
        if assembly_height_m <= 0.0:
            raise ValueError("assembly_height_m must be positive.")

        self.panels_m = panels
        self.electrode_voltages_v = voltages
        self.assembly_height_m = float(assembly_height_m)

        self.n_panels = len(panels)
        self.surface_charge_density_c_m2: np.ndarray | None = None
        self.influence_matrix: np.ndarray | None = None

        if self.n_panels > MAX_PANELS_WARNING:
            print(
                "[BEM2D] Warning: "
                f"{self.n_panels} panels imply "
                f"{self.n_panels**2 / 1e6:.1f} million matrix elements."
            )

    @property
    def panel_centres_m(self) -> np.ndarray:
        """Return panel-centre coordinates as an array of shape (N, 2)."""
        return np.column_stack(
            [
                0.5 * (self.panels_m[:, 0] + self.panels_m[:, 1]),
                0.5 * (self.panels_m[:, 2] + self.panels_m[:, 3]),
            ]
        )

    def assemble(self, *, show_progress: bool = True) -> np.ndarray:
        """Assemble the dense BEM influence matrix."""
        centres = self.panel_centres_m
        matrix = np.empty((self.n_panels, self.n_panels), dtype=float)

        iterator = tqdm(
            range(self.n_panels),
            desc="BEM assembly",
            disable=not show_progress,
        )

        for row in iterator:
            x_m, y_m = centres[row]
            integral = rectangular_panel_integral(
                x_m,
                y_m,
                self.assembly_height_m,
                self.panels_m,
            )

            panel_width_m = self.panels_m[row, 1] - self.panels_m[row, 0]
            panel_height_m = self.panels_m[row, 3] - self.panels_m[row, 2]
            integral[row] = rectangular_panel_self_integral(
                panel_width_m,
                panel_height_m,
            )

            matrix[row, :] = COULOMB_CONSTANT * integral

        self.influence_matrix = matrix
        return matrix

    def solve(self, *, show_progress: bool = True) -> np.ndarray:
        """Solve for panel surface-charge densities."""
        if self.influence_matrix is None:
            self.assemble(show_progress=show_progress)

        self.surface_charge_density_c_m2 = np.linalg.solve(
            self.influence_matrix,
            self.electrode_voltages_v,
        )
        return self.surface_charge_density_c_m2

    def set_electrode_voltages(
        self,
        electrode_voltages_v: np.ndarray,
    ) -> None:
        """Replace boundary voltages and invalidate the charge solution."""
        voltages = np.asarray(electrode_voltages_v, dtype=float)
        if voltages.shape != (self.n_panels,):
            raise ValueError(
                "electrode_voltages_v must have shape (n_panels,)."
            )

        self.electrode_voltages_v = voltages
        self.surface_charge_density_c_m2 = None

    def potential(self, x_m: float, y_m: float, z_m: float) -> float:
        """Evaluate electric potential in volts above the electrode plane."""
        if self.surface_charge_density_c_m2 is None:
            self.solve(show_progress=False)

        integral = rectangular_panel_integral(
            x_m,
            y_m,
            z_m,
            self.panels_m,
        )
        return float(
            COULOMB_CONSTANT
            * np.dot(self.surface_charge_density_c_m2, integral)
        )

    def electric_field(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> tuple[float, float, float]:
        """Evaluate electric field (E_x, E_y, E_z) in V/m."""
        if z_m <= 0.0:
            raise ValueError("Evaluation height z_m must be positive.")

        if self.surface_charge_density_c_m2 is None:
            self.solve(show_progress=False)

        field_x = np.dot(
            self.surface_charge_density_c_m2,
            _field_x_integral(x_m, y_m, z_m, self.panels_m),
        )
        field_y = np.dot(
            self.surface_charge_density_c_m2,
            _field_y_integral(x_m, y_m, z_m, self.panels_m),
        )
        field_z = np.dot(
            self.surface_charge_density_c_m2,
            _field_z_integral(x_m, y_m, z_m, self.panels_m),
        )

        return (
            float(COULOMB_CONSTANT * field_x),
            float(COULOMB_CONSTANT * field_y),
            float(COULOMB_CONSTANT * field_z),
        )

    def potential_grid(
        self,
        x_grid_m: np.ndarray,
        y_grid_m: np.ndarray,
        z_m: float,
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Evaluate potential on matching x/y grid arrays."""
        x_grid = np.asarray(x_grid_m, dtype=float)
        y_grid = np.asarray(y_grid_m, dtype=float)

        if x_grid.shape != y_grid.shape:
            raise ValueError("x_grid_m and y_grid_m must have the same shape.")

        values = np.empty_like(x_grid, dtype=float)
        iterator = np.ndindex(x_grid.shape)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=x_grid.size,
                desc=f"BEM potential grid at z={z_m * 1e6:.1f} um",
            )

        for index in iterator:
            values[index] = self.potential(
                float(x_grid[index]),
                float(y_grid[index]),
                z_m,
            )

        return values

    def electric_field_grid(
        self,
        x_grid_m: np.ndarray,
        y_grid_m: np.ndarray,
        z_m: float,
        *,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate electric field on matching x/y grid arrays."""
        x_grid = np.asarray(x_grid_m, dtype=float)
        y_grid = np.asarray(y_grid_m, dtype=float)

        if x_grid.shape != y_grid.shape:
            raise ValueError("x_grid_m and y_grid_m must have the same shape.")

        field_x = np.empty_like(x_grid, dtype=float)
        field_y = np.empty_like(x_grid, dtype=float)
        field_z = np.empty_like(x_grid, dtype=float)

        iterator = np.ndindex(x_grid.shape)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=x_grid.size,
                desc=f"BEM field grid at z={z_m * 1e6:.1f} um",
            )

        for index in iterator:
            ex, ey, ez = self.electric_field(
                float(x_grid[index]),
                float(y_grid[index]),
                z_m,
            )
            field_x[index] = ex
            field_y[index] = ey
            field_z[index] = ez

        return field_x, field_y, field_z