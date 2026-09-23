"""Step 07c: inner RF-contour control points on the validated baseline trap.

The workflow keeps the 06c baseline X-junction unchanged except for its
existing INNER RF boundary. This is the RF/DC boundary facing the central
transport channel and therefore the relevant contour for suppressing the
central RF-null-height volcano.

No disconnected RF islands are created. No new rail topology is introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.junction_templates import (
    baseline_x_junction_rf_mask,
    make_house_style_x_junction,
)
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

INNER_KNOTS_M = (
    30e-6,
    60e-6,
    90e-6,
    120e-6,
    150e-6,
)


def make_parameters(
    inner_offsets_um: tuple[float, ...],
):
    """Build the fixed 06c baseline with a changed inner RF boundary only."""
    if len(inner_offsets_um) != len(INNER_KNOTS_M):
        raise ValueError(
            "inner_offsets_um must have one value per INNER_KNOTS_M."
        )

    if abs(inner_offsets_um[0]) > 1e-12:
        raise ValueError("First inner offset must be zero.")
    if abs(inner_offsets_um[-1]) > 1e-12:
        raise ValueError("Last inner offset must be zero.")

    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=BASE_TAPER_LENGTH_M,
        inner_edge_shift_at_centre_m=BASE_INNER_SHIFT_M,
        outer_edge_shift_at_centre_m=BASE_OUTER_SHIFT_M,
        taper_power=BASE_TAPER_POWER,
        rf_start_radius_override_m=BASE_START_RADIUS_M,
        inner_contour_knots_m=INNER_KNOTS_M,
        inner_contour_offsets_m=tuple(
            value_um * 1e-6
            for value_um in inner_offsets_um
        ),
    )


def plot_layout(
    model,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot the BEM quadtree layout around the actual transport-channel edge."""
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
        edgecolor=(0.25, 0.25, 0.25, 0.15),
        linewidth=0.20,
    )
    collection.set_clim(0.0, 1.0)
    axis.add_collection(collection)

    axis.set_aspect("equal")
    axis.set_xlim(-180.0, 180.0)
    axis.set_ylim(-180.0, 180.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.16)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_exact_inner_contour(
    reference,
    candidate,
    *,
    label: str,
    output_path: Path,
) -> None:
    """Plot the exact inner RF/DC boundary deformation."""
    s_m = np.linspace(25e-6, 160e-6, 1001)

    ref_inner_m, ref_outer_m = reference.rail_boundaries_m(s_m)
    cand_inner_m, cand_outer_m = candidate.rail_boundaries_m(s_m)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )

    axes[0].plot(
        s_m * 1e6,
        ref_inner_m * 1e6,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Reference inner RF edge",
    )
    axes[0].plot(
        s_m * 1e6,
        cand_inner_m * 1e6,
        color="tab:red",
        linewidth=2.5,
        label=f"{label} inner RF edge",
    )
    axes[0].plot(
        s_m * 1e6,
        ref_outer_m * 1e6,
        color="tab:blue",
        linestyle=":",
        linewidth=1.7,
        label="Shared outer RF edge",
    )

    for knot_m in INNER_KNOTS_M:
        axes[0].axvline(
            knot_m * 1e6,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

    axes[0].set_ylabel(r"Boundary radius [um]")
    axes[0].set_title(
        f"Exact inner RF/DC-boundary deformation: {label}"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    delta_inner_m = cand_inner_m - ref_inner_m

    axes[1].fill_between(
        s_m * 1e6,
        0.0,
        delta_inner_m * 1e6,
        where=delta_inner_m >= 0.0,
        color="tab:red",
        alpha=0.45,
        label="RF retracts from transport channel",
    )
    axes[1].fill_between(
        s_m * 1e6,
        0.0,
        delta_inner_m * 1e6,
        where=delta_inner_m < 0.0,
        color="tab:blue",
        alpha=0.45,
        label="RF protrudes toward transport channel",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_xlabel(r"Longitudinal coordinate $s$ [um]")
    axes[1].set_ylabel(r"$\Delta r_{\rm in}$ [um]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_mask_difference(
    reference,
    candidate,
    *,
    label: str,
    output_path: Path,
) -> None:
    """Show changed RF area near the central channel only."""
    x_edges_m = np.linspace(-170e-6, 170e-6, 1001)
    y_edges_m = np.linspace(-170e-6, 170e-6, 1001)

    x_centres_m = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y_centres_m = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])

    x_grid_m, y_grid_m = np.meshgrid(
        x_centres_m,
        y_centres_m,
        indexing="xy",
    )

    reference_mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        reference,
    )
    candidate_mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        candidate,
    )

    difference = np.zeros_like(reference_mask, dtype=np.int8)
    difference[(~reference_mask) & candidate_mask] = +1
    difference[reference_mask & (~candidate_mask)] = -1

    changed = int(np.count_nonzero(difference))

    figure, axis = plt.subplots(figsize=(8.0, 8.0))

    cmap = ListedColormap(
        [
            "#2f6db2",  # -1: removed RF / enlarged channel
            "#f5f5f5",  #  0: unchanged
            "#d62728",  # +1: RF added / narrowed channel
        ]
    )

    axis.pcolormesh(
        x_edges_m * 1e6,
        y_edges_m * 1e6,
        difference,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        shading="auto",
    )

    axis.set_aspect("equal")
    axis.set_xlim(-170.0, 170.0)
    axis.set_ylim(-170.0, 170.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(
        f"RF-mask difference: {label}\n"
        f"red = RF added toward channel; blue = RF removed; "
        f"changed samples = {changed}"
    )
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"{label}: changed mask samples = {changed}")


def print_summary(
    *,
    label: str,
    offsets_um: tuple[float, ...],
    parameters,
    model,
) -> None:
    """Print compact physical geometry diagnostics."""
    knots_m = np.asarray(INNER_KNOTS_M, dtype=float)
    inner_m, outer_m = parameters.rail_boundaries_m(knots_m)
    offset_m = parameters.inner_contour_profile_m(knots_m)

    dense_s_m = np.linspace(0.0, 220e-6, 2001)
    dense_inner_m, dense_outer_m = parameters.rail_boundaries_m(
        dense_s_m
    )
    width_m = dense_outer_m - dense_inner_m

    print()
    print(f"=== {label} ===")
    print(
        "control knots [um]:     "
        + ", ".join(
            f"{value * 1e6:.0f}"
            for value in INNER_KNOTS_M
        )
    )
    print(
        "requested offsets [um]: "
        + ", ".join(
            f"{value:+.2f}" for value in offsets_um
        )
    )
    print(
        "realized offsets [um]:  "
        + ", ".join(
            f"{value * 1e6:+.4f}" for value in offset_m
        )
    )
    print(
        "inner RF edge [um]:     "
        + ", ".join(
            f"{value * 1e6:.4f}" for value in inner_m
        )
    )
    print(
        "outer RF edge [um]:     "
        + ", ".join(
            f"{value * 1e6:.4f}" for value in outer_m
        )
    )
    print(
        "minimum RF width [um]:  "
        f"{np.min(width_m) * 1e6:.4f}"
    )
    print(
        "minimum inner radius [um]: "
        f"{np.min(dense_inner_m) * 1e6:.4f}"
    )
    print(f"quadtree panel count: {model.n_panels}")


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07c_inner_contour_smoke_test"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    cases: dict[str, tuple[float, ...]] = {
        "reference": (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        # Positive: RF moves away from channel, white channel expands.
        "inner_retracts": (
            0.0,
            0.0,
            +12.0,
            0.0,
            0.0,
        ),
        # Negative: RF moves into channel, white channel narrows.
        "inner_protrudes": (
            0.0,
            0.0,
            -12.0,
            0.0,
            0.0,
        ),
        # Piecewise-linear connected contour with both directions.
        "mixed_inner_contour": (
            0.0,
            +8.0,
            -10.0,
            +7.0,
            0.0,
        ),
    }

    reference = make_parameters(cases["reference"])

    print("Step 07c: inner RF-contour smoke test")
    print("=" * 100)
    print(
        "Fixed baseline: start=30 um, inner=-25 um, outer=+20 um, "
        "L=150 um, p=2.0"
    )
    print(
        "Only the existing INNER RF/DC boundary facing the central "
        "transport channel is changed."
    )

    for label, offsets_um in tqdm(
        cases.items(),
        total=len(cases),
        desc="07c inner-contour cases",
        unit="case",
        dynamic_ncols=True,
    ):
        parameters = make_parameters(offsets_um)

        report = check_x_junction_manufacturability(parameters)

        print(
            f"{label}: manufacturable={report.valid}, "
            f"messages={report.messages}"
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
            title=f"07c inner contour: {label}",
        )

        if label != "reference":
            plot_exact_inner_contour(
                reference,
                parameters,
                label=label,
                output_path=output_directory / f"exact_{label}.png",
            )
            plot_mask_difference(
                reference,
                parameters,
                label=label,
                output_path=output_directory / f"difference_{label}.png",
            )

    print()
    print(f"Saved figures to: {output_directory}")


if __name__ == "__main__":
    main()