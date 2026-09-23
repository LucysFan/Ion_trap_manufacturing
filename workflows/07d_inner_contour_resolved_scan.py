"""Step 07d: resolved BEM scan of connected inner RF-boundary control points.

The fixed base geometry is the best 06c candidate:
    start_radius = 30 um
    inner_shift  = -25 um
    outer_shift  = +20 um
    taper_length = 150 um
    taper_power  = 2.0

This workflow changes only the existing INNER RF boundary facing the
transport channel. The deformation is piecewise linear through fixed knots:
    s = (30, 60, 90, 120, 150) um.

The endpoint offsets are fixed to zero. The scan varies only the central
control point at s=90 um, then reruns the best candidates with diagnostics.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
from core.optimization.junction_evaluator import (
    plot_candidate_diagnostics,
    plot_candidate_layout,
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
    inner_offsets_m: tuple[float, float, float, float, float],
):
    """Create fixed 06c baseline with a connected inner RF-edge profile."""
    if len(inner_offsets_m) != len(INNER_KNOTS_M):
        raise ValueError(
            "inner_offsets_m must match INNER_KNOTS_M length."
        )

    if abs(inner_offsets_m[0]) > 1e-15:
        raise ValueError("Offset at s=30 um must be zero.")

    if abs(inner_offsets_m[-1]) > 1e-15:
        raise ValueError("Offset at s=150 um must be zero.")

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
        inner_contour_offsets_m=inner_offsets_m,
    )


def offsets_from_central_amplitude_m(
    central_amplitude_m: float,
) -> tuple[float, float, float, float, float]:
    """Return one-gene central inner-boundary deformation profile."""
    return (
        0.0,
        0.0,
        central_amplitude_m,
        0.0,
        0.0,
    )


def empty_result(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
) -> dict[str, object]:
    """Create a fixed-schema result row before physical evaluation."""
    return {
        "label": label,
        "inner_knot_30_offset_m": inner_offsets_m[0],
        "inner_knot_60_offset_m": inner_offsets_m[1],
        "inner_knot_90_offset_m": inner_offsets_m[2],
        "inner_knot_120_offset_m": inner_offsets_m[3],
        "inner_knot_150_offset_m": inner_offsets_m[4],
        "trace_valid": False,
        "path_valid": False,
        "height_peak_m": np.inf,
        "height_rms_m": np.inf,
        "barrier_ev": np.inf,
        "maximum_y_m": np.inf,
        "n_panels": 0,
        "reason": "",
        "candidate_directory": "",
    }


def evaluate_inner_contour(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
    x_values_m: np.ndarray,
    candidate_directory: Path | None = None,
    save_diagnostics: bool = False,
    show_bem_progress: bool = False,
) -> dict[str, object]:
    """Build, solve, trace, and evaluate one connected inner contour."""
    row = empty_result(
        label=label,
        inner_offsets_m=inner_offsets_m,
    )

    parameters = make_parameters(inner_offsets_m)

    try:
        manufacturability = check_x_junction_manufacturability(parameters)
    except ValueError as error:
        row["reason"] = f"invalid geometry: {error}"
        return row

    if not manufacturability.valid:
        row["reason"] = "; ".join(manufacturability.messages)
        return row

    model = build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )

    model.bem.assemble(show_progress=show_bem_progress)
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

    if save_diagnostics:
        if candidate_directory is None:
            raise ValueError(
                "candidate_directory is required when "
                "save_diagnostics=True."
            )

        candidate_directory.mkdir(parents=True, exist_ok=True)

        offsets_um = tuple(value * 1e6 for value in inner_offsets_m)
        title = (
            f"{label}: inner offsets at s=(30,60,90,120,150) um = "
            f"({offsets_um[0]:+.1f}, {offsets_um[1]:+.1f}, "
            f"{offsets_um[2]:+.1f}, {offsets_um[3]:+.1f}, "
            f"{offsets_um[4]:+.1f}) um"
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

        row["candidate_directory"] = str(candidate_directory)

    return {
        **row,
        "trace_valid": bool(trace.valid),
        "path_valid": bool(validation.valid),
        "height_peak_m": float(metrics.height_peak_deviation_m),
        "height_rms_m": float(metrics.height_rms_deviation_m),
        "barrier_ev": float(barrier.barrier_height_ev),
        "maximum_y_m": float(metrics.maximum_lateral_offset_m),
        "n_panels": int(model.n_panels),
        "reason": "; ".join(validation.messages),
    }


def print_header() -> None:
    """Print the console table header."""
    print("Step 07d: resolved connected inner RF-contour scan")
    print("=" * 138)
    print(
        "Base geometry: start=30 um, inner=-25 um, outer=+20 um, "
        "L=150 um, p=2.0"
    )
    print(
        "Inner knots [um]: 30, 60, 90, 120, 150; "
        "offsets at 30 and 150 um are fixed at zero."
    )
    print(
        f"{'index':>5} {'d_in,90 [um]':>15} {'trace':>7} {'path':>7} "
        f"{'peak dz [um]':>15} {'rms dz [um]':>14} "
        f"{'barrier [meV]':>16} {'max |y| [um]':>15} {'panels':>8}"
    )


def print_row(index: int, row: dict[str, object]) -> None:
    """Print one evaluation row."""
    print(
        f"{index:5d} "
        f"{float(row['inner_knot_90_offset_m']) * 1e6:15.2f} "
        f"{str(bool(row['trace_valid'])):>7} "
        f"{str(bool(row['path_valid'])):>7} "
        f"{float(row['height_peak_m']) * 1e6:15.4f} "
        f"{float(row['height_rms_m']) * 1e6:14.4f} "
        f"{float(row['barrier_ev']) * 1e3:16.5f} "
        f"{float(row['maximum_y_m']) * 1e6:15.4f} "
        f"{int(row['n_panels']):8d}"
    )


def write_results_csv(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Write all scan metrics to CSV."""
    fieldnames = (
        "label",
        "inner_knot_30_offset_m",
        "inner_knot_60_offset_m",
        "inner_knot_90_offset_m",
        "inner_knot_120_offset_m",
        "inner_knot_150_offset_m",
        "trace_valid",
        "path_valid",
        "height_peak_m",
        "height_rms_m",
        "barrier_ev",
        "maximum_y_m",
        "n_panels",
        "reason",
        "candidate_directory",
    )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def valid_rows_only(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep rows with a converged trace and finite physical metrics."""
    return [
        row
        for row in rows
        if bool(row["trace_valid"])
        and np.isfinite(float(row["height_peak_m"]))
        and np.isfinite(float(row["height_rms_m"]))
        and np.isfinite(float(row["barrier_ev"]))
        and np.isfinite(float(row["maximum_y_m"]))
    ]


def plot_sensitivity(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot height, barrier, and lateral displacement versus central offset."""
    valid_rows = valid_rows_only(rows)

    if not valid_rows:
        return

    valid_rows.sort(
        key=lambda row: float(row["inner_knot_90_offset_m"])
    )

    amplitude_um = np.asarray(
        [
            float(row["inner_knot_90_offset_m"]) * 1e6
            for row in valid_rows
        ],
        dtype=float,
    )
    peak_height_um = np.asarray(
        [
            float(row["height_peak_m"]) * 1e6
            for row in valid_rows
        ],
        dtype=float,
    )
    barrier_mev = np.asarray(
        [
            float(row["barrier_ev"]) * 1e3
            for row in valid_rows
        ],
        dtype=float,
    )
    maximum_y_um = np.asarray(
        [
            float(row["maximum_y_m"]) * 1e6
            for row in valid_rows
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.8, 9.4),
        sharex=True,
    )

    axes[0].plot(
        amplitude_um,
        peak_height_um,
        "o-",
        color="tab:blue",
        linewidth=2.0,
        markersize=6.5,
    )
    axes[0].axhline(
        3.0,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label=r"Target: $\Delta z_{\rm peak}=3$ um",
    )
    axes[0].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Reference inner boundary",
    )
    axes[0].set_ylabel(r"Peak $\Delta z$ [um]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        amplitude_um,
        barrier_mev,
        "o-",
        color="tab:purple",
        linewidth=2.0,
        markersize=6.5,
    )
    axes[1].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].set_ylabel("RF barrier [meV]")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        amplitude_um,
        maximum_y_um,
        "o-",
        color="tab:green",
        linewidth=2.0,
        markersize=6.5,
    )
    axes[2].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axes[2].set_xlabel(
        r"Inner-boundary offset at $s=90$ um [um]"
    )
    axes[2].set_ylabel(r"Maximum $|y|$ [um]")
    axes[2].grid(alpha=0.3)

    figure.suptitle(
        r"Step 07d: one-gene connected inner RF contour, "
        r"$\delta_{\rm in}(90)$"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def estimate_central_sensitivity(
    rows: list[dict[str, object]],
) -> tuple[float, float] | None:
    """Estimate local finite-difference height sensitivity around zero."""
    valid_rows = valid_rows_only(rows)

    by_amplitude = {
        float(row["inner_knot_90_offset_m"]): row
        for row in valid_rows
    }

    negative = [
        amplitude
        for amplitude in by_amplitude
        if amplitude < 0.0
    ]
    positive = [
        amplitude
        for amplitude in by_amplitude
        if amplitude > 0.0
    ]

    if not negative or not positive:
        return None

    negative_amplitude = max(negative)
    positive_amplitude = min(positive)

    negative_height_m = float(
        by_amplitude[negative_amplitude]["height_peak_m"]
    )
    positive_height_m = float(
        by_amplitude[positive_amplitude]["height_peak_m"]
    )

    sensitivity = (
        (positive_height_m - negative_height_m)
        / (positive_amplitude - negative_amplitude)
    )

    return sensitivity, 0.5 * (
        positive_amplitude - negative_amplitude
    )


def compromise_score(row: dict[str, object]) -> float:
    """Dimensionless score used only to select a diagnostic candidate."""
    if not bool(row["trace_valid"]):
        return np.inf

    height_um = float(row["height_peak_m"]) * 1e6
    barrier_mev = float(row["barrier_ev"]) * 1e3
    maximum_y_um = float(row["maximum_y_m"]) * 1e6

    if not np.all(
        np.isfinite(
            (
                height_um,
                barrier_mev,
                maximum_y_um,
            )
        )
    ):
        return np.inf

    return (
        (height_um / 20.0) ** 2
        + (barrier_mev / 100.0) ** 2
        + (maximum_y_um / 5.0) ** 2
    )


def print_selected(
    category: str,
    row: dict[str, object],
) -> None:
    """Print one selected candidate summary."""
    print(
        f"{category:<16} {row['label']}: "
        f"d_in,90={float(row['inner_knot_90_offset_m']) * 1e6:+.1f} um, "
        f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
        f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV, "
        f"max |y|={float(row['maximum_y_m']) * 1e6:.4f} um"
    )


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07d_inner_contour_resolved_scan"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Same range and point count as 05h, 06b, 06c, and 06d.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    # Negative offset: RF protrudes toward the channel.
    # Positive offset: RF retracts from the channel.
    central_amplitudes_um = (
        -15.0,
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
        15.0,
    )

    print_header()

    rows: list[dict[str, object]] = []

    for index, amplitude_um in enumerate(
        central_amplitudes_um,
        start=1,
    ):
        inner_offsets_m = offsets_from_central_amplitude_m(
            amplitude_um * 1e-6
        )
        label = f"inner_90_{amplitude_um:+.0f}um"

        row = evaluate_inner_contour(
            label=label,
            inner_offsets_m=inner_offsets_m,
            x_values_m=x_values_m,
            save_diagnostics=False,
            show_bem_progress=True,
        )

        rows.append(row)
        print_row(index, row)

    write_results_csv(
        rows,
        output_path=output_directory / "inner_contour_scan_results.csv",
    )
    plot_sensitivity(
        rows,
        output_path=output_directory / "inner_contour_sensitivity.png",
    )

    sensitivity_result = estimate_central_sensitivity(rows)

    if sensitivity_result is not None:
        sensitivity, half_step_m = sensitivity_result

        print()
        print(
            "Local central sensitivity: "
            f"d(peak dz)/d(d_in,90) = {sensitivity:.4f} m/m "
            f"from the nearest valid +/- {half_step_m * 1e6:.1f} um pair."
        )

    valid_rows = valid_rows_only(rows)

    if not valid_rows:
        print()
        print(
            "No valid candidates. "
            "Inspect manufacturability and trace output."
        )
        print(f"Partial results saved to: {output_directory}")
        return

    best_height = min(
        valid_rows,
        key=lambda row: float(row["height_peak_m"]),
    )
    best_barrier = min(
        valid_rows,
        key=lambda row: float(row["barrier_ev"]),
    )
    best_compromise = min(
        valid_rows,
        key=compromise_score,
    )

    selected = {
        "best_height": best_height,
        "best_barrier": best_barrier,
        "best_compromise": best_compromise,
    }

    print()
    print("Selected diagnostic reruns")
    print("=" * 138)

    for category, row in selected.items():
        print_selected(category, row)

        inner_offsets_m = (
            float(row["inner_knot_30_offset_m"]),
            float(row["inner_knot_60_offset_m"]),
            float(row["inner_knot_90_offset_m"]),
            float(row["inner_knot_120_offset_m"]),
            float(row["inner_knot_150_offset_m"]),
        )

        evaluate_inner_contour(
            label=f"{category}_{row['label']}",
            inner_offsets_m=inner_offsets_m,
            x_values_m=x_values_m,
            candidate_directory=output_directory / "selected" / category,
            save_diagnostics=True,
            show_bem_progress=True,
        )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()