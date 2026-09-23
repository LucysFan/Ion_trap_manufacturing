"""Step 07e: resolved scan of broad connected inner RF-contour profiles.

The fixed geometry is the best 06c baseline:
    start_radius = 30 um
    inner_shift  = -25 um
    outer_shift  = +20 um
    taper_length = 150 um
    taper_power  = 2.0

The existing inner RF boundary facing the transport channel is represented by
piecewise-linear offsets at s=(30, 60, 90, 120, 150) um. The endpoint offsets
at 30 and 150 um remain zero. This scan tests deliberately broad negative
profiles, because 07d showed that a single central control point becomes
physically active only for sufficiently negative displacement.
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
    """Create the fixed baseline with one connected inner RF contour."""
    if len(inner_offsets_m) != len(INNER_KNOTS_M):
        raise ValueError("inner_offsets_m must match INNER_KNOTS_M length.")
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


def profile_um_to_m(
    offsets_um: tuple[float, float, float],
) -> tuple[float, float, float, float, float]:
    """Embed free offsets at 60, 90, 120 um in zero-fixed endpoints."""
    return (
        0.0,
        offsets_um[0] * 1e-6,
        offsets_um[1] * 1e-6,
        offsets_um[2] * 1e-6,
        0.0,
    )


def empty_result(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
) -> dict[str, object]:
    """Create a stable row schema before a physical BEM evaluation."""
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


def evaluate_profile(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
    x_values_m: np.ndarray,
    candidate_directory: Path | None = None,
    save_diagnostics: bool = False,
    show_bem_progress: bool = False,
) -> dict[str, object]:
    """Build, solve, trace, and evaluate one connected inner profile."""
    row = empty_result(label=label, inner_offsets_m=inner_offsets_m)
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
                "candidate_directory is required when save_diagnostics=True."
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
    print("Step 07e: resolved broad connected inner RF-contour profile scan")
    print("=" * 152)
    print(
        "Base geometry: start=30 um, inner=-25 um, outer=+20 um, "
        "L=150 um, p=2.0"
    )
    print(
        "Free connected inner-boundary genes: d_in(60), d_in(90), "
        "d_in(120); endpoint offsets at 30 and 150 um are zero."
    )
    print(
        f"{'index':>5} {'d60 [um]':>10} {'d90 [um]':>10} {'d120 [um]':>11} "
        f"{'trace':>7} {'path':>7} {'peak dz [um]':>15} "
        f"{'rms dz [um]':>14} {'barrier [meV]':>16} "
        f"{'max |y| [um]':>15} {'panels':>8}"
    )


def print_row(index: int, row: dict[str, object]) -> None:
    """Print one physical evaluation row."""
    print(
        f"{index:5d} "
        f"{float(row['inner_knot_60_offset_m']) * 1e6:10.2f} "
        f"{float(row['inner_knot_90_offset_m']) * 1e6:10.2f} "
        f"{float(row['inner_knot_120_offset_m']) * 1e6:11.2f} "
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
    """Write scan results to a CSV file."""
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


def valid_rows_only(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Retain converged candidates with finite physical metrics."""
    return [
        row
        for row in rows
        if bool(row["trace_valid"])
        and np.isfinite(float(row["height_peak_m"]))
        and np.isfinite(float(row["height_rms_m"]))
        and np.isfinite(float(row["barrier_ev"]))
        and np.isfinite(float(row["maximum_y_m"]))
    ]


def plot_profile_comparison(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Compare valid broad profiles by peak height and barrier."""
    valid_rows = valid_rows_only(rows)
    if not valid_rows:
        return

    labels = [str(row["label"]) for row in valid_rows]
    peak_height_um = np.asarray(
        [float(row["height_peak_m"]) * 1e6 for row in valid_rows],
        dtype=float,
    )
    barrier_mev = np.asarray(
        [float(row["barrier_ev"]) * 1e3 for row in valid_rows],
        dtype=float,
    )
    maximum_y_um = np.asarray(
        [float(row["maximum_y_m"]) * 1e6 for row in valid_rows],
        dtype=float,
    )

    positions = np.arange(len(valid_rows))
    figure, axes = plt.subplots(3, 1, figsize=(11.8, 9.2), sharex=True)

    axes[0].bar(positions, peak_height_um, color="tab:blue", alpha=0.85)
    axes[0].axhline(
        3.0,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label=r"Target: $\Delta z_{\rm peak}=3$ um",
    )
    axes[0].set_ylabel(r"Peak $\Delta z$ [um]")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].bar(positions, barrier_mev, color="tab:purple", alpha=0.85)
    axes[1].set_ylabel("RF barrier [meV]")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(positions, maximum_y_um, color="tab:green", alpha=0.85)
    axes[2].set_ylabel(r"Maximum $|y|$ [um]")
    axes[2].set_xticks(positions, labels, rotation=24, ha="right")
    axes[2].grid(axis="y", alpha=0.3)

    figure.suptitle("Step 07e: broad connected inner-boundary profile scan")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compromise_score(row: dict[str, object]) -> float:
    """Dimensionless selection score for diagnostic reruns only."""
    if not bool(row["trace_valid"]):
        return np.inf

    height_um = float(row["height_peak_m"]) * 1e6
    barrier_mev = float(row["barrier_ev"]) * 1e3
    maximum_y_um = float(row["maximum_y_m"]) * 1e6

    if not np.all(np.isfinite((height_um, barrier_mev, maximum_y_um))):
        return np.inf

    return (
        (height_um / 20.0) ** 2
        + (barrier_mev / 100.0) ** 2
        + (maximum_y_um / 5.0) ** 2
    )


def print_selected(category: str, row: dict[str, object]) -> None:
    """Print a compact summary for one selected candidate."""
    print(
        f"{category:<16} {row['label']}: "
        f"d60={float(row['inner_knot_60_offset_m']) * 1e6:+.1f} um, "
        f"d90={float(row['inner_knot_90_offset_m']) * 1e6:+.1f} um, "
        f"d120={float(row['inner_knot_120_offset_m']) * 1e6:+.1f} um, "
        f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
        f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
    )


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07e_inner_contour_profile_scan"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Keep the resolved scan definition used by 05h, 06b, 06c, and 06d.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    # Entries are the free connected offsets at knots (60, 90, 120) um.
    # Negative values protrude RF toward the transport channel.
    profiles_um: dict[str, tuple[float, float, float]] = {
        "baseline": (0.0, 0.0, 0.0),
        "centre_minus_15": (0.0, -15.0, 0.0),
        "broad_minus_10": (-10.0, -10.0, -10.0),
        "broad_minus_15": (-15.0, -15.0, -15.0),
        "triangle_minus_15": (-8.0, -15.0, -8.0),
        "wide_centre_minus_15": (-12.0, -15.0, -12.0),
        "leading_minus_15": (-15.0, -12.0, -6.0),
        "trailing_minus_15": (-6.0, -12.0, -15.0),
    }

    print_header()
    rows: list[dict[str, object]] = []

    for index, (label, profile_um) in enumerate(profiles_um.items(), start=1):
        row = evaluate_profile(
            label=label,
            inner_offsets_m=profile_um_to_m(profile_um),
            x_values_m=x_values_m,
            save_diagnostics=False,
            show_bem_progress=True,
        )
        rows.append(row)
        print_row(index, row)

    write_results_csv(
        rows,
        output_path=output_directory / "inner_profile_scan_results.csv",
    )
    plot_profile_comparison(
        rows,
        output_path=output_directory / "inner_profile_comparison.png",
    )

    valid_rows = valid_rows_only(rows)
    if not valid_rows:
        print()
        print("No valid candidates. Inspect manufacturability and trace output.")
        print(f"Partial results saved to: {output_directory}")
        return

    best_height = min(valid_rows, key=lambda row: float(row["height_peak_m"]))
    best_barrier = min(valid_rows, key=lambda row: float(row["barrier_ev"]))
    best_compromise = min(valid_rows, key=compromise_score)

    selected = {
        "best_height": best_height,
        "best_barrier": best_barrier,
        "best_compromise": best_compromise,
    }

    print()
    print("Selected diagnostic reruns")
    print("=" * 152)

    for category, row in selected.items():
        print_selected(category, row)

        inner_offsets_m = (
            float(row["inner_knot_30_offset_m"]),
            float(row["inner_knot_60_offset_m"]),
            float(row["inner_knot_90_offset_m"]),
            float(row["inner_knot_120_offset_m"]),
            float(row["inner_knot_150_offset_m"]),
        )

        evaluate_profile(
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