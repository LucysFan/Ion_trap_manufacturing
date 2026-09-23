"""Step 5g: validate geometry-aware quadtree resolution of both RF edges."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)

if TYPE_CHECKING:
    from core.geometry.mask_builder import XJunctionBEMModel


def plot_layouts(
    models: list[tuple[str, "XJunctionBEMModel"]],
    *,
    output_path: Path,
) -> None:
    """Draw irregular panel meshes and RF masks."""
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.2 * len(models), 5.0),
        sharex=True,
        sharey=True,
    )

    if len(models) == 1:
        axes = [axes]

    for axis, (label, model) in zip(axes, models):
        panels_m = model.bem.panels_m
        rf = model.bem.electrode_voltages_v > 0.5

        patches = [
            Rectangle(
                (panel[0] * 1e6, panel[2] * 1e6),
                (panel[1] - panel[0]) * 1e6,
                (panel[3] - panel[2]) * 1e6,
            )
            for panel in panels_m
        ]

        collection = PatchCollection(
            patches,
            array=rf.astype(float),
            cmap="RdYlBu_r",
            edgecolor=(0.25, 0.25, 0.25, 0.25),
            linewidth=0.25,
        )
        collection.set_clim(0.0, 1.0)
        axis.add_collection(collection)

        axis.set_aspect("equal")
        axis.set_xlim(-260.0, 260.0)
        axis.set_ylim(-260.0, 260.0)
        axis.set_xlabel("x [um]")
        axis.set_ylabel("y [um]")
        axis.set_title(f"{label}\nN={model.n_panels} panels")
        axis.grid(alpha=0.15)

    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.12,
        top=0.85,
        wspace=0.18,
    )
    figure.suptitle("Geometry-aware quadtree meshes")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def boundary_panel_count(model: "XJunctionBEMModel") -> int:
    """Count panels whose centre-neighbours include both RF and ground.

    This is a lightweight diagnostic. Exact boundary detection was already
    used to create the mesh; here we simply report a proxy complexity metric.
    """
    panels_m = model.bem.panels_m
    voltages = model.bem.electrode_voltages_v > 0.5

    widths_m = panels_m[:, 1] - panels_m[:, 0]
    heights_m = panels_m[:, 3] - panels_m[:, 2]

    fine = (widths_m <= 12e-6) | (heights_m <= 12e-6)
    return int(np.sum(fine & voltages))


def main() -> None:
    output_directory = (
        Path("reports") / "figures" / "05g_geometry_aware_quadtree"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("baseline", 0.0, 0.0),
        ("outer +20 um", 0.0, 20e-6),
        ("outer +40 um", 0.0, 40e-6),
        ("inner -20 / outer +20", -20e-6, 20e-6),
    ]

    models: list[tuple[str, XJunctionBEMModel]] = []

    print("Geometry-aware quadtree validation")
    print("-" * 108)
    print(
        f"{'candidate':<26} {'inner [um]':>12} {'outer [um]':>12} "
        f"{'panels':>10} {'RF panels':>12} {'fine RF panels':>16} "
        f"{'min cell [um]':>15}"
    )

    for label, inner_shift_m, outer_shift_m in candidates:
        parameters = make_house_style_x_junction(
            ion_height_m=TARGET_ION_HEIGHT_M,
            arm_length_m=600e-6,
            outer_extent_m=900e-6,
            taper_length_m=150e-6,
            inner_edge_shift_at_centre_m=inner_shift_m,
            outer_edge_shift_at_centre_m=outer_shift_m,
            taper_power=2.0,
        )

        model = build_geometry_aware_quadtree_x_junction_bem(
            parameters,
            central_half_extent_m=180e-6,
            central_max_cell_m=30e-6,
            boundary_max_cell_m=10e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=5e-6,
        )
        models.append((label, model))

        panels_m = model.bem.panels_m
        widths_m = panels_m[:, 1] - panels_m[:, 0]
        heights_m = panels_m[:, 3] - panels_m[:, 2]
        min_cell_m = min(np.min(widths_m), np.min(heights_m))
        rf_panels = int(np.sum(model.bem.electrode_voltages_v > 0.5))

        print(
            f"{label:<26} "
            f"{inner_shift_m * 1e6:12.1f} "
            f"{outer_shift_m * 1e6:12.1f} "
            f"{model.n_panels:10d} "
            f"{rf_panels:12d} "
            f"{boundary_panel_count(model):16d} "
            f"{min_cell_m * 1e6:15.3f}"
        )

    plot_layouts(
        models,
        output_path=output_directory / "geometry_aware_quadtree_meshes.png",
    )

    print()
    print(
        "Expected outcome: outer +20 and outer +40 must now have visibly "
        "different RF contours and/or RF panel counts. For dense BEM field "
        "solves, reduce panel count below ~1500 if necessary by increasing "
        "central_max_cell_m or boundary_max_cell_m modestly."
    )
    print(
        f"Figure saved to: "
        f"{output_directory / 'geometry_aware_quadtree_meshes.png'}"
    )


if __name__ == "__main__":
    main()