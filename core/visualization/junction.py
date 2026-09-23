"""Matplotlib plotting utilities for X-junction BEM models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.geometry.mask_builder import XJunctionBEMModel


def _ensure_parent_directory(path: str | Path) -> Path:
    """Create parent directory for a figure path and return Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_x_junction_mask(
    model: XJunctionBEMModel,
    *,
    output_path: str | Path,
    title: str = "X-junction RF electrode mask",
) -> None:
    """Plot RF and ground cells of the rectangular BEM mesh."""
    path = _ensure_parent_directory(output_path)

    fig, axis = plt.subplots(figsize=(7.5, 7.0))
    image = axis.pcolormesh(
        model.x_edges_m * 1e6,
        model.y_edges_m * 1e6,
        model.rf_mask.astype(float),
        shading="flat",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = fig.colorbar(image, ax=axis, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["Ground", "RF"])

    axis.set_aspect("equal")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _field_squared_map(
    model: XJunctionBEMModel,
    *,
    z_m: float,
    x_limits_m: tuple[float, float],
    y_limits_m: tuple[float, float],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate |E|² on a horizontal grid."""
    x_values_m = np.linspace(x_limits_m[0], x_limits_m[1], n_points)
    y_values_m = np.linspace(y_limits_m[0], y_limits_m[1], n_points)

    x_grid_m, y_grid_m = np.meshgrid(
        x_values_m,
        y_values_m,
        indexing="xy",
    )

    field_x, field_y, field_z = model.bem.electric_field_grid(
        x_grid_m,
        y_grid_m,
        z_m,
        show_progress=True,
    )

    field_squared = field_x**2 + field_y**2 + field_z**2
    return x_grid_m, y_grid_m, field_squared


def plot_field_squared_map(
    model: XJunctionBEMModel,
    *,
    z_m: float,
    output_path: str | Path,
    x_limits_m: tuple[float, float] = (-450e-6, 450e-6),
    y_limits_m: tuple[float, float] = (-450e-6, 450e-6),
    n_points: int = 61,
) -> None:
    """Plot log10(|E_RF|²) in a horizontal plane."""
    path = _ensure_parent_directory(output_path)

    x_grid_m, y_grid_m, field_squared = _field_squared_map(
        model,
        z_m=z_m,
        x_limits_m=x_limits_m,
        y_limits_m=y_limits_m,
        n_points=n_points,
    )

    log_field_squared = np.log10(field_squared + 1e-20)

    fig, axis = plt.subplots(figsize=(8.0, 7.0))
    image = axis.pcolormesh(
        x_grid_m * 1e6,
        y_grid_m * 1e6,
        log_field_squared,
        shading="auto",
        cmap="inferno",
    )

    contours = axis.contour(
        x_grid_m * 1e6,
        y_grid_m * 1e6,
        log_field_squared,
        levels=10,
        colors="white",
        linewidths=0.45,
        alpha=0.55,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%.1f")

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}|E_{\rm RF}|^2$ [$(\mathrm{V/m})^2$]")

    axis.set_aspect("equal")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(
        rf"RF field map at $z={z_m * 1e6:.1f}\,\mathrm{{um}}$"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_xz_field_squared_slice(
    model: XJunctionBEMModel,
    *,
    output_path: str | Path,
    y_m: float = 0.0,
    x_limits_m: tuple[float, float] = (-450e-6, 450e-6),
    z_limits_m: tuple[float, float] = (15e-6, 180e-6),
    n_x: int = 81,
    n_z: int = 61,
) -> None:
    """Plot log10(|E_RF|²) in x-z plane at fixed y."""
    path = _ensure_parent_directory(output_path)

    x_values_m = np.linspace(x_limits_m[0], x_limits_m[1], n_x)
    z_values_m = np.linspace(z_limits_m[0], z_limits_m[1], n_z)

    x_grid_m, z_grid_m = np.meshgrid(
        x_values_m,
        z_values_m,
        indexing="xy",
    )
    field_squared = np.empty_like(x_grid_m)

    total_points = x_grid_m.size
    point_index = 0

    for row in range(n_z):
        for column in range(n_x):
            x_m = float(x_grid_m[row, column])
            z_m = float(z_grid_m[row, column])
            field = model.bem.electric_field(x_m, y_m, z_m)
            field_squared[row, column] = float(np.dot(field, field))

            point_index += 1
            if point_index % 500 == 0 or point_index == total_points:
                print(
                    f"x-z field map: {point_index}/{total_points} points",
                    end="\r",
                )

    print()

    log_field_squared = np.log10(field_squared + 1e-20)

    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    image = axis.pcolormesh(
        x_grid_m * 1e6,
        z_grid_m * 1e6,
        log_field_squared,
        shading="auto",
        cmap="inferno",
    )

    contours = axis.contour(
        x_grid_m * 1e6,
        z_grid_m * 1e6,
        log_field_squared,
        levels=12,
        colors="white",
        linewidths=0.4,
        alpha=0.55,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%.1f")

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}|E_{\rm RF}|^2$ [$(\mathrm{V/m})^2$]")

    axis.set_xlabel("x [um]")
    axis.set_ylabel("z [um]")
    axis.set_title(
        rf"x-z RF field slice at $y={y_m * 1e6:.1f}\,\mathrm{{um}}$"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)