"""Step 5d: verify that BEM mesh resolves both RF rail boundaries.

This workflow does not solve the full BEM system. It only proves that
different continuous geometry parameters create different discretised
edge-aligned meshes and RF masks. That check is required before a GA
can meaningfully optimise both inner and outer RF boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.mask_builder import build_x_junction_bem

if TYPE_CHECKING:
    from core.geometry.mask_builder import XJunctionBEMModel


def plot_masks(
    models: list[tuple[str, "XJunctionBEMModel"]],
    *,
    output_path: Path,
) -> None:
    """Draw edge-aligned meshes and RF masks for comparison candidates."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 11),
        sharex=True,
        sharey=True,
    )

    image = None

    for axis, (label, model) in zip(axes.flat, models):
        image = axis.pcolormesh(
            model.x_edges_m * 1e6,
            model.y_edges_m * 1e6,
            model.rf_mask.astype(float),
            shading="flat",
            cmap="RdYlBu_r",
            vmin=0.0,
            vmax=1.0,
        )

        # Show local mesh lines only near the junction. The lines make it
        # visually clear that shifted RF boundaries become panel boundaries.
        for edge_um in model.x_edges_m * 1e6:
            if abs(edge_um) <= 200.0:
                axis.axvline(
                    edge_um,
                    color="white",
                    alpha=0.08,
                    linewidth=0.5,
                )

        for edge_um in model.y_edges_m * 1e6:
            if abs(edge_um) <= 200.0:
                axis.axhline(
                    edge_um,
                    color="white",
                    alpha=0.08,
                    linewidth=0.5,
                )

        axis.set_aspect("equal")
        axis.set_title(f"{label}\nN={model.n_panels} panels")
        axis.set_xlabel("x [um]")
        axis.set_ylabel("y [um]")
        axis.set_xlim(-250.0, 250.0)
        axis.set_ylim(-250.0, 250.0)

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.8,
        )
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["Ground", "RF"])

    figure.suptitle("Edge-aligned BEM meshes: both RF boundaries are resolved")
    figure.subplots_adjust(
    left=0.07,
    right=0.89,
    bottom=0.07,
    top=0.91,
    wspace=0.12,
    hspace=0.16,
)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_directory = Path("reports") / "figures" / "05d_edge_aligned"
    output_directory.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, float, float]] = [
        ("baseline", 0.0, 0.0),
        ("outer +20 um", 0.0, 20e-6),
        ("outer +40 um", 0.0, 40e-6),
        ("inner -20, outer +20 um", -20e-6, 20e-6),
    ]

    models: list[tuple[str, XJunctionBEMModel]] = []

    print("Edge-aligned mesh validation")
    print("-" * 94)
    print(
        f"{'candidate':<28} {'inner [um]':>12} {'outer [um]':>12} "
        f"{'axis cells':>12} {'panels':>10} {'RF cells':>10}"
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

        model = build_x_junction_bem(
            parameters,
            edge_aligned=True,
            n_longitudinal_samples=17,
            max_cell_size_centre_m=15e-6,
            max_cell_size_outer_m=75e-6,
            centre_refinement_radius_m=250e-6,
        )
        models.append((label, model))

        print(
            f"{label:<28} "
            f"{inner_shift_m * 1e6:12.1f} "
            f"{outer_shift_m * 1e6:12.1f} "
            f"{len(model.x_edges_m) - 1:12d} "
            f"{model.n_panels:10d} "
            f"{int(np.sum(model.rf_mask)):10d}"
        )

    plot_masks(
        models,
        output_path=output_directory / "edge_aligned_masks.png",
    )

    print()
    print(f"Figure saved to: {output_directory / 'edge_aligned_masks.png'}")
    print()
    print(
        "This is a geometry-only validation. It intentionally does not "
        "assemble or solve the dense BEM matrix. The next workflow will "
        "perform a two-edge field scan with a panel budget chosen to keep "
        "the dense BEM solve tractable."
    )


if __name__ == "__main__":
    main()