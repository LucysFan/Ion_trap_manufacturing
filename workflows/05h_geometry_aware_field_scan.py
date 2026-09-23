"""Step 5h: accurate two-edge RF field scan on geometry-aware quadtree meshes.

This is a small, high-fidelity design-of-experiments scan. It evaluates
inner and outer RF-boundary changes independently with the exact
geometry-aware quadtree panelization before introducing a GA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from config.targets import (
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
)
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.path_metrics import compute_path_metrics
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.validation import validate_rf_transport_path
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.visualization.geometry_scan import (
    plot_height_profiles,
    plot_transition_scan_summary,
)

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from core.geometry.mask_builder import XJunctionBEMModel
    from core.analysis.rf_null_trace import RFNullTrace
    from core.analysis.path_metrics import PathMetrics
    from core.analysis.barrier import BarrierMetrics

def plot_candidate_layout(
    model: "XJunctionBEMModel",
    trace: "RFNullTrace",
    *,
    output_path: Path,
    title: str,
) -> None:
    """Draw quadtree electrode panels and traced RF transport path."""
    panels_m = model.bem.panels_m
    rf = model.bem.electrode_voltages_v > 0.5

    figure, axis = plt.subplots(figsize=(8.0, 7.5))

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

    good = trace.converged
    axis.plot(
        trace.x_m[good] * 1e6,
        trace.y_m[good] * 1e6,
        color="lime",
        linewidth=2.5,
        label="Traced transverse RF minimum",
    )

    if np.any(~good):
        axis.scatter(
            trace.x_m[~good] * 1e6,
            np.zeros(np.sum(~good)),
            marker="x",
            color="cyan",
            s=45,
            label="Trace failure",
        )

    axis.set_aspect("equal")
    axis.set_xlim(-350.0, 350.0)
    axis.set_ylim(-350.0, 350.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_diagnostics(
    trace: "RFNullTrace",
    metrics: "PathMetrics",
    barrier: "BarrierMetrics",
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot z-path, lateral displacement, field residual, and RF proxy."""
    good = trace.converged
    x_um = trace.x_m[good] * 1e6

    figure, axes = plt.subplots(4, 1, figsize=(10.0, 11.0), sharex=True)

    axes[0].plot(
        x_um,
        trace.z_m[good] * 1e6,
        color="tab:blue",
        linewidth=2.0,
    )
    axes[0].axhline(
        metrics.arm_reference_height_m * 1e6,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Arm reference",
    )
    axes[0].set_ylabel(r"$z_{\rm min}$ [um]")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        x_um,
        trace.y_m[good] * 1e6,
        color="tab:green",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"$y_{\rm min}$ [um]")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(
        x_um,
        trace.transverse_residual_v_m[good],
        color="tab:red",
        linewidth=1.8,
        label=r"$\sqrt{E_y^2+E_z^2}$",
    )
    axes[2].semilogy(
        x_um,
        trace.full_field_norm_v_m[good],
        color="tab:orange",
        linestyle="--",
        linewidth=1.3,
        label=r"$|E|$",
    )
    axes[2].set_ylabel("Field [V/m]")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3, which="both")

    axes[3].plot(
        x_um,
        barrier.energies_ev[good] * 1e3,
        color="tab:purple",
        linewidth=2.0,
    )
    axes[3].axhline(
        barrier.reference_energy_ev * 1e3,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Arm reference",
    )
    axes[3].set_xlabel("x [um]")
    axes[3].set_ylabel(r"$U_{\rm ps}$ [meV]")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_candidate(
    *,
    label: str,
    start_radius_m: float,
    inner_shift_m: float,
    outer_shift_m: float,
    taper_length_m: float,
    taper_power: float,
    x_values_m: np.ndarray,
    candidate_directory: Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Build, solve, trace, and evaluate one resolved X-junction candidate."""
    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=taper_length_m,
        inner_edge_shift_at_centre_m=inner_shift_m,
        outer_edge_shift_at_centre_m=outer_shift_m,
        taper_power=taper_power,
        rf_start_radius_override_m=start_radius_m,
    )

    try:
        manufacturability = check_x_junction_manufacturability(parameters)
    except ValueError as error:
        return (
            {
                "label": label,
                "start_radius_m": start_radius_m,
                "inner_shift_m": inner_shift_m,
                "outer_shift_m": outer_shift_m,
                "trace_valid": False,
                "height_peak_m": np.inf,
                "height_rms_m": np.inf,
                "barrier_ev": np.inf,
                "n_panels": 0,
                "reason": f"invalid geometry: {error}",
            },
            None,
        )

    if not manufacturability.valid:
        return (
            {
                "label": label,
                "start_radius_m": start_radius_m,
                "inner_shift_m": inner_shift_m,
                "outer_shift_m": outer_shift_m,
                "trace_valid": False,
                "height_peak_m": np.inf,
                "height_rms_m": np.inf,
                "barrier_ev": np.inf,
                "n_panels": 0,
                "reason": "; ".join(manufacturability.messages),
            },
            None,
        )

    model = build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )

    model.bem.assemble(show_progress=True)
    model.bem.solve(show_progress=False)

    trace = trace_rf_transverse_minimum(
        model.bem,
        x_values_m,
        initial_y_m=0.0,
        initial_z_m=TARGET_ION_HEIGHT_M,
        residual_tolerance_v_m=1e-3,
        max_transverse_shift_m=25e-6,
    )
    metrics = compute_path_metrics(trace)

    energies_ev = pseudopotential_profile_ev(
        model.bem,
        trace,
        rf_voltage_peak_v=100.0,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )
    barrier = compute_barrier_metrics(energies_ev, trace)
    validation = validate_rf_transport_path(metrics)

    candidate_directory.mkdir(parents=True, exist_ok=True)

    title = (
        f"{label}: start={start_radius_m * 1e6:.1f} um, "
        f"din={inner_shift_m * 1e6:.1f} um, "
        f"dout={outer_shift_m * 1e6:.1f} um"
    )

    plot_candidate_layout(
        model,
        trace,
        output_path=candidate_directory / "layout_and_trace.png",
        title=title,
    )
    plot_candidate_diagnostics(
        trace,
        metrics,
        barrier,
        output_path=candidate_directory / "path_diagnostics.png",
        title=title,
    )

    row: dict[str, object] = {
        "label": label,
        "start_radius_m": start_radius_m,
        "inner_shift_m": inner_shift_m,
        "outer_shift_m": outer_shift_m,
        "trace_valid": trace.valid,
        "path_valid": validation.valid,
        "height_peak_m": metrics.height_peak_deviation_m,
        "height_rms_m": metrics.height_rms_deviation_m,
        "barrier_ev": barrier.barrier_height_ev,
        "maximum_y_m": metrics.maximum_lateral_offset_m,
        "n_panels": model.n_panels,
        "reason": "; ".join(validation.messages),
        "candidate_directory": str(candidate_directory),
    }

    profile: dict[str, object] | None = None

    if trace.valid:
        profile = {
            "x_m": trace.x_m,
            "deviation_m": trace.z_m - metrics.arm_reference_height_m,
            "label": (
                f"{label}: dz={metrics.height_peak_deviation_m * 1e6:.1f} um, "
                f"U={barrier.barrier_height_ev * 1e3:.1f} meV"
            ),
        }

    return row, profile


def main() -> None:
    output_directory = (
        Path("reports") / "figures" / "05h_geometry_aware_field_scan"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Fewer trace points than final validation, because each candidate has a
    # resolved 4.5k-panel BEM model. Use 71 later for selected finalists.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    # label, start radius [um], inner shift [um], outer shift [um],
    # taper length [um], taper power
    scan_plan = [
        ("baseline", 37.35, 0.0, 0.0, 150.0, 2.0),
        ("start30", 30.0, 0.0, 0.0, 150.0, 2.0),
        ("inner-10", 30.0, -10.0, 0.0, 150.0, 2.0),
        ("inner-20", 30.0, -20.0, 0.0, 150.0, 2.0),
        ("outer+20", 30.0, 0.0, 20.0, 150.0, 2.0),
        ("outer+40", 30.0, 0.0, 40.0, 150.0, 2.0),
        ("wide-10+20", 30.0, -10.0, 20.0, 150.0, 2.0),
        ("wide-20+20", 30.0, -20.0, 20.0, 150.0, 2.0),
        ("wide-10+40", 30.0, -10.0, 40.0, 150.0, 2.0),
        ("long-wide", 30.0, -10.0, 20.0, 250.0, 2.0),
    ]

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []

    print("Geometry-aware two-edge field scan")
    print("-" * 130)
    print(
        f"{'label':<16} {'start':>8} {'din':>8} {'dout':>8} "
        f"{'trace':>7} {'peak dz [um]':>15} {'rms dz [um]':>14} "
        f"{'barrier [meV]':>16} {'panels':>8}"
    )

    for index, (
        label,
        start_um,
        inner_um,
        outer_um,
        taper_um,
        power,
    ) in enumerate(scan_plan, start=1):
        print(
            f"[{index:02d}/{len(scan_plan):02d}] {label}: "
            f"building/resolving field..."
        )

        candidate_directory = output_directory / "candidates" / label

        row, profile = evaluate_candidate(
            label=label,
            start_radius_m=start_um * 1e-6,
            inner_shift_m=inner_um * 1e-6,
            outer_shift_m=outer_um * 1e-6,
            taper_length_m=taper_um * 1e-6,
            taper_power=power,
            x_values_m=x_values_m,
            candidate_directory=candidate_directory,
        )
        results.append(row)

        if profile is not None:
            profiles.append(profile)

        print(
            f"{label:<16} "
            f"{start_um:8.2f} "
            f"{inner_um:8.1f} "
            f"{outer_um:8.1f} "
            f"{str(bool(row['trace_valid'])):>7} "
            f"{float(row['height_peak_m']) * 1e6:15.4f} "
            f"{float(row['height_rms_m']) * 1e6:14.4f} "
            f"{float(row['barrier_ev']) * 1e3:16.5f} "
            f"{int(row['n_panels']):8d}"
        )

    valid_results = [
        row
        for row in results
        if bool(row["trace_valid"])
        and np.isfinite(float(row["height_peak_m"]))
    ]
    valid_results.sort(
        key=lambda row: (
            float(row["height_peak_m"]),
            float(row["barrier_ev"]),
        )
    )

    print()
    print("Best resolved two-edge candidates")
    print("-" * 130)

    for rank, row in enumerate(valid_results[:5], start=1):
        print(
            f"{rank}. {row['label']}: "
            f"start={float(row['start_radius_m']) * 1e6:.1f} um, "
            f"din={float(row['inner_shift_m']) * 1e6:.1f} um, "
            f"dout={float(row['outer_shift_m']) * 1e6:.1f} um, "
            f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
            f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
        )

    plot_transition_scan_summary(
        results,
        output_path=output_directory / "two_edge_tradeoff.png",
    )

    selected_profiles: list[dict[str, object]] = []

    for row in valid_results[:5]:
        prefix = f"{row['label']}:"
        matches = [
            profile
            for profile in profiles
            if str(profile["label"]).startswith(prefix)
        ]
        if matches:
            selected_profiles.append(matches[0])

    if selected_profiles:
        plot_height_profiles(
            selected_profiles,
            output_path=output_directory / "best_two_edge_profiles.png",
        )

    np.savez(
        output_directory / "two_edge_scan_results.npz",
        label=np.asarray([str(row["label"]) for row in results]),
        start_radius_m=np.asarray(
            [float(row["start_radius_m"]) for row in results]
        ),
        inner_shift_m=np.asarray(
            [float(row["inner_shift_m"]) for row in results]
        ),
        outer_shift_m=np.asarray(
            [float(row["outer_shift_m"]) for row in results]
        ),
        trace_valid=np.asarray(
            [bool(row["trace_valid"]) for row in results]
        ),
        path_valid=np.asarray(
            [bool(row.get("path_valid", False)) for row in results]
        ),
        height_peak_m=np.asarray(
            [float(row["height_peak_m"]) for row in results]
        ),
        height_rms_m=np.asarray(
            [float(row["height_rms_m"]) for row in results]
        ),
        barrier_ev=np.asarray(
            [float(row["barrier_ev"]) for row in results]
        ),
        n_panels=np.asarray([int(row["n_panels"]) for row in results]),
    )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()