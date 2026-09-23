"""Step 5f: validate quadtree X-junction meshes before field scans."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.mask_builder import build_quadtree_x_junction_bem

if TYPE_CHECKING:
    from core.geometry.mask_builder import XJunctionBEMModel


def plot_quadtree_layout(
    models: list[tuple[str, "XJunctionBEMModel"]],
    *,
    output_path: Path,
) -> None:
    """Render irregular quadtree panels as matplotlib rectangles."""
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

        colors = np.where(rf, 1.0, 0.0)

        collection = PatchCollection(
            patches,
            array=colors,
            cmap="RdYlBu_r",
            edgecolor=(0.25, 0.25, 0.25, 0.24),
            linewidth=0.28,
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
    figure.suptitle("Adaptive quadtree panel meshes")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_directory = Path("reports") / "figures" / "05f_quadtree_mesh"
    output_directory.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("baseline", 0.0, 0.0),
        ("outer +20 um", 0.0, 20e-6),
        ("outer +40 um", 0.0, 40e-6),
        ("inner -20 / outer +20", -20e-6, 20e-6),
    ]

    models: list[tuple[str, XJunctionBEMModel]] = []

    print("Quadtree mesh validation")
    print("-" * 96)
    print(
        f"{'candidate':<26} {'inner [um]':>12} {'outer [um]':>12} "
        f"{'panels':>10} {'RF panels':>12} {'min cell [um]':>15}"
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

        model = build_quadtree_x_junction_bem(
            parameters,
            central_half_extent_m=180e-6,
            central_max_cell_m=25e-6,
            boundary_max_cell_m=20e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=10e-6,
        )
        models.append((label, model))

        panels_m = model.bem.panels_m
        widths_m = panels_m[:, 1] - panels_m[:, 0]
        heights_m = panels_m[:, 3] - panels_m[:, 2]
        min_cell_m = min(np.min(widths_m), np.min(heights_m))

        print(
            f"{label:<26} "
            f"{inner_shift_m * 1e6:12.1f} "
            f"{outer_shift_m * 1e6:12.1f} "
            f"{model.n_panels:10d} "
            f"{int(np.sum(model.bem.electrode_voltages_v > 0.5)):12d} "
            f"{min_cell_m * 1e6:15.3f}"
        )

    plot_quadtree_layout(
        models,
        output_path=output_directory / "quadtree_meshes.png",
    )

    print()
    print(
        "For the first BEM field scan, target a panel count below roughly "
        "1500. If a candidate exceeds that, increase central_max_cell_m or "
        "boundary_max_cell_m before assembling dense BEM."
    )
    print(f"Figure saved to: {output_directory / 'quadtree_meshes.png'}")


if __name__ == "__main__":
    main()