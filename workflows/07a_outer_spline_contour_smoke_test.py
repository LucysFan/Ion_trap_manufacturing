"""Step 07a: smoke test for a multi-control-point outer RF contour.

This workflow verifies that the outer boundary can have an arbitrary smooth,
fourfold-symmetric shape defined by spline control points. It does not solve
the BEM system; it validates geometry, manufacturability, mesh refinement,
and contour profiles before the 07b physical scan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)


BASE_START_RADIUS_M = 30e-6
BASE_INNER_SHIFT_M = -25e-6
BASE_OUTER_SHIFT_M = 20e-6
BASE_TAPER_LENGTH_M = 150e-6
BASE_TAPER_POWER = 2.0

KNOTS_M = (
    0e-6,
    35e-6,
    70e-6,
    105e-6,
    140e-6,
    170e-6,
)


def make_parameters(
    offsets_um: tuple[float, ...],
):
    """Build the 06c base junction with a spline-defined outer contour."""
    if len(offsets_um) != len(KNOTS_M):
        raise ValueError("offsets_um must have one value per contour knot.")

    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=BASE_TAPER_LENGTH_M,
        inner_edge_shift_at_centre_m=BASE_INNER_SHIFT_M,
        outer_edge_shift_at_centre_m=BASE_OUTER_SHIFT_M,
        taper_power=BASE_TAPER_POWER,
        rf_start_radius_override_m=BASE_START_RADIUS_M,
        outer_bulge_amplitude_m=0.0,
        outer_contour_knots_m=KNOTS_M,
        outer_contour_offsets_m=tuple(
            offset_um * 1e-6
            for offset_um in offsets_um
        ),
    )


def plot_layout(
    model,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Draw quadtree panels, colored by RF/ground classification."""
    panels_m = model.bem.panels_m
    rf = model.bem.electrode_voltages_v > 0.5

    figure, axis = plt.subplots(figsize=(8.2, 8.2))

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
        edgecolor=(0.25, 0.25, 0.25, 0.18),
        linewidth=0.22,
    )
    collection.set_clim(0.0, 1.0)
    axis.add_collection(collection)

    axis.set_aspect("equal")
    axis.set_xlim(-210.0, 210.0)
    axis.set_ylim(-210.0, 210.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.15)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_profiles(
    profiles: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Compare outer-contour offsets and RF rail boundaries."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.0),
        sharex=True,
    )

    for profile in profiles:
        s_um = np.asarray(profile["s_m"], dtype=float) * 1e6

        axes[0].plot(
            s_um,
            np.asarray(profile["outer_offset_m"], dtype=float) * 1e6,
            linewidth=2.0,
            label=str(profile["label"]),
        )

        axes[1].plot(
            s_um,
            np.asarray(profile["inner_m"], dtype=float) * 1e6,
            linewidth=1.7,
            linestyle="--",
            alpha=0.75,
        )
        axes[1].plot(
            s_um,
            np.asarray(profile["outer_m"], dtype=float) * 1e6,
            linewidth=2.1,
            label=str(profile["label"]),
        )

    for knot_m in KNOTS_M:
        axes[0].axvline(
            knot_m * 1e6,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
        )
        axes[1].axvline(
            knot_m * 1e6,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"Outer-contour offset [um]")
    axes[0].set_title("Spline-defined external RF boundary")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_xlabel(r"Longitudinal coordinate $s=|x|$ or $|y|$ [um]")
    axes[1].set_ylabel("Boundary distance from arm axis [um]")
    axes[1].set_title(
        "Solid: outer RF edge; dashed: inner RF edge (shared)"
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_summary(
    *,
    label: str,
    offsets_um: tuple[float, ...],
    parameters,
    model,
) -> None:
    """Print numeric profile and mesh diagnostics."""
    sample_s_m = np.asarray(KNOTS_M, dtype=float)
    inner_m, outer_m = parameters.rail_boundaries_m(sample_s_m)
    contour_m = parameters.outer_contour_profile_m(sample_s_m)
    width_m = outer_m - inner_m

    print()
    print(f"=== {label} ===")
    print(
        "knots [um]:          "
        + ", ".join(f"{value * 1e6:.0f}" for value in KNOTS_M)
    )
    print(
        "requested offsets:   "
        + ", ".join(f"{value:+.1f}" for value in offsets_um)
    )
    print(
        "realized offsets:    "
        + ", ".join(f"{value * 1e6:+.4f}" for value in contour_m)
    )
    print(
        "inner edge [um]:     "
        + ", ".join(f"{value * 1e6:.4f}" for value in inner_m)
    )
    print(
        "outer edge [um]:     "
        + ", ".join(f"{value * 1e6:.4f}" for value in outer_m)
    )
    print(
        "RF width [um]:       "
        + ", ".join(f"{value * 1e6:.4f}" for value in width_m)
    )
    print(f"minimum sampled RF width [um]: {np.min(width_m) * 1e6:.4f}")
    print(f"maximum outer edge [um]: {np.max(outer_m) * 1e6:.4f}")
    print(f"quadtree panel count: {model.n_panels}")


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07a_outer_spline_contour_smoke_test"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # The final value must be exactly zero: beyond 170 um the contour must
    # return to the normal arm. Values are in micrometres.
    contour_cases: dict[str, tuple[float, ...]] = {
        "reference": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),

        # One broad, smooth outward lobe. No alternating ripple.
        "broad_outer_bulge": (
            0.0,
            8.0,
            18.0,
            14.0,
            4.0,
            0.0,
        ),

        # One broad, smooth inward lobe. No sharp return or extra lobe.
        "broad_outer_notch": (
            0.0,
            -6.0,
            -16.0,
            -12.0,
            -4.0,
            0.0,
        ),
    }
    profiles: list[dict[str, object]] = []

    print("Step 07a: arbitrary smooth outer RF-contour smoke test")
    print("=" * 100)

    for label, offsets_um in contour_cases.items():
        parameters = make_parameters(offsets_um)
        report = check_x_junction_manufacturability(parameters)

        print(
            f"{label}: valid={report.valid}, messages={report.messages}"
        )

        model = build_geometry_aware_quadtree_x_junction_bem(
            parameters,
            central_half_extent_m=180e-6,
            central_max_cell_m=30e-6,
            boundary_max_cell_m=10e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=5e-6,
        )

        print_summary(
            label=label,
            offsets_um=offsets_um,
            parameters=parameters,
            model=model,
        )

        plot_layout(
            model,
            output_path=output_directory / f"layout_{label}.png",
            title=f"07a outer contour: {label}",
        )

        s_m = np.linspace(0.0, 220e-6, 1001)
        inner_m, outer_m = parameters.rail_boundaries_m(s_m)

        profiles.append(
            {
                "label": label,
                "s_m": s_m,
                "inner_m": inner_m,
                "outer_m": outer_m,
                "outer_offset_m": parameters.outer_contour_profile_m(s_m),
            }
        )

    plot_profiles(
        profiles,
        output_path=output_directory / "outer_spline_profiles.png",
    )

    print()
    print(f"Saved figures to: {output_directory}")


if __name__ == "__main__":
    main()