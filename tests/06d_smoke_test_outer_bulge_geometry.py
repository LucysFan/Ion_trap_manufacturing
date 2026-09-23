"""Step 06d smoke test: visual verification of an outer RF-edge bulge."""

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


def plot_layout(model, *, output_path: Path, title: str) -> None:
    panels_m = model.bem.panels_m
    rf = model.bem.electrode_voltages_v > 0.5

    figure, axis = plt.subplots(figsize=(8.0, 8.0))

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
    axis.set_xlim(-180.0, 180.0)
    axis.set_ylim(-180.0, 180.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.15)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def print_boundary_summary(
    *,
    amplitude_um: float,
    parameters,
    model,
    reference_inner_m: np.ndarray | None,
    reference_outer_m: np.ndarray | None,
    sample_s_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Print numerical contour diagnostics and return boundary profiles."""
    inner_m, outer_m = parameters.rail_boundaries_m(sample_s_m)

    s0_index = int(np.argmin(np.abs(sample_s_m - 0.0)))
    s70_index = int(np.argmin(np.abs(sample_s_m - 70e-6)))
    s150_index = int(np.argmin(np.abs(sample_s_m - 150e-6)))

    if reference_inner_m is None:
        inner_delta_m = np.zeros_like(inner_m)
        outer_delta_m = np.zeros_like(outer_m)
    else:
        inner_delta_m = inner_m - reference_inner_m
        outer_delta_m = outer_m - reference_outer_m

    max_outer_index = int(np.argmax(np.abs(outer_delta_m)))
    max_inner_index = int(np.argmax(np.abs(inner_delta_m)))

    print()
    print(f"=== outer bulge {amplitude_um:+.1f} um ===")
    print(
        "inner edge at s=0 / 70 / 150 um [um]: "
        f"{inner_m[s0_index] * 1e6:.4f}, "
        f"{inner_m[s70_index] * 1e6:.4f}, "
        f"{inner_m[s150_index] * 1e6:.4f}"
    )
    print(
        "outer edge at s=0 / 70 / 150 um [um]: "
        f"{outer_m[s0_index] * 1e6:.4f}, "
        f"{outer_m[s70_index] * 1e6:.4f}, "
        f"{outer_m[s150_index] * 1e6:.4f}"
    )
    print(
        "RF rail width at s=0 / 70 / 150 um [um]: "
        f"{(outer_m[s0_index] - inner_m[s0_index]) * 1e6:.4f}, "
        f"{(outer_m[s70_index] - inner_m[s70_index]) * 1e6:.4f}, "
        f"{(outer_m[s150_index] - inner_m[s150_index]) * 1e6:.4f}"
    )
    print(
        "max |outer-edge change| vs A=0 [um]: "
        f"{np.abs(outer_delta_m[max_outer_index]) * 1e6:.6f} "
        f"at s={sample_s_m[max_outer_index] * 1e6:.3f} um"
    )
    print(
        "max |inner-edge change| vs A=0 [um]: "
        f"{np.abs(inner_delta_m[max_inner_index]) * 1e6:.6f} "
        f"at s={sample_s_m[max_inner_index] * 1e6:.3f} um"
    )
    print(f"quadtree panel count: {model.n_panels}")

    return inner_m, outer_m

def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "06d_smoke_test_outer_bulge_geometry"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    amplitudes_um = (0.0, 30.0)
    sample_s_m = np.linspace(0.0, 180e-6, 1001)

    reference_inner_m: np.ndarray | None = None
    reference_outer_m: np.ndarray | None = None
    for amplitude_um in amplitudes_um:
        parameters = make_house_style_x_junction(
            ion_height_m=TARGET_ION_HEIGHT_M,
            arm_length_m=600e-6,
            outer_extent_m=900e-6,
            taper_length_m=150e-6,
            inner_edge_shift_at_centre_m=-25e-6,
            outer_edge_shift_at_centre_m=20e-6,
            taper_power=2.0,
            rf_start_radius_override_m=30e-6,
            outer_bulge_amplitude_m=amplitude_um * 1e-6,
            outer_bulge_center_m=70e-6,
            outer_bulge_sigma_m=25e-6,
        )

        report = check_x_junction_manufacturability(parameters)
        print(
            f"bulge={amplitude_um:+.1f} um "
            f"valid={report.valid} "
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
        
        inner_m, outer_m = print_boundary_summary(
        amplitude_um=amplitude_um,
        parameters=parameters,
        model=model,
        reference_inner_m=reference_inner_m,
        reference_outer_m=reference_outer_m,
        sample_s_m=sample_s_m,
        )

        if amplitude_um == 0.0:
            reference_inner_m = inner_m.copy()
            reference_outer_m = outer_m.copy()

        plot_layout(
            model,
            output_path=output_directory / f"layout_bulge_{amplitude_um:+.0f}um.png",
            title=(
                "Outer RF-edge bulge: "
                f"{amplitude_um:+.1f} um at |x| = 70 um"
            ),
        )

        s_m = np.linspace(0.0, 180e-6, 501)
        inner_m, outer_m = parameters.rail_boundaries_m(s_m)

        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        axis.plot(s_m * 1e6, inner_m * 1e6, label="Inner RF edge")
        axis.plot(s_m * 1e6, outer_m * 1e6, label="Outer RF edge")
        axis.axvline(70.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel(r"$|x|$ or $|y|$ [um]")
        axis.set_ylabel("Boundary distance from axis [um]")
        axis.set_title(f"RF-rail boundaries, outer bulge {amplitude_um:+.1f} um")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_directory / f"boundary_profile_{amplitude_um:+.0f}um.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    print(f"Saved smoke-test figures to: {output_directory}")



if __name__ == "__main__":
    main()