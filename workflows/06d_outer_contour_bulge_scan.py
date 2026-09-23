"""Step 06d: resolved scan of a local outer RF-boundary deformation.

The base geometry is the best candidate from 06c:
    start_radius = 30 um
    inner_shift  = -25 um
    outer_shift  = +20 um
    taper_length = 150 um
    taper_power  = 2.0

A symmetric local Gaussian feature is applied only to the outer RF boundary:

    delta_r_out(s) = A_out * exp[-0.5 * ((s - s_c) / sigma)^2],

where s = |x| for horizontal arms and s = |y| for vertical arms.

Positive A_out locally widens the RF rail from the outside.
Negative A_out locally narrows the RF rail from the outside.
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

BULGE_CENTER_M = 70e-6
BULGE_SIGMA_M = 25e-6


def make_parameters(outer_bulge_amplitude_m: float):
    """Create the 06c best geometry plus an outer-edge local deformation."""
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=BASE_TAPER_LENGTH_M,
        inner_edge_shift_at_centre_m=BASE_INNER_SHIFT_M,
        outer_edge_shift_at_centre_m=BASE_OUTER_SHIFT_M,
        taper_power=BASE_TAPER_POWER,
        rf_start_radius_override_m=BASE_START_RADIUS_M,
        outer_bulge_amplitude_m=outer_bulge_amplitude_m,
        outer_bulge_center_m=BULGE_CENTER_M,
        outer_bulge_sigma_m=BULGE_SIGMA_M,
    )


def empty_result(
    *,
    label: str,
    outer_bulge_amplitude_m: float,
) -> dict[str, object]:
    """Create one finite-schema result row before a physical evaluation."""
    return {
        "label": label,
        "outer_bulge_amplitude_m": outer_bulge_amplitude_m,
        "outer_bulge_center_m": BULGE_CENTER_M,
        "outer_bulge_sigma_m": BULGE_SIGMA_M,
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


def evaluate_outer_bulge(
    *,
    label: str,
    outer_bulge_amplitude_m: float,
    x_values_m: np.ndarray,
    candidate_directory: Path | None = None,
    save_diagnostics: bool = False,
    show_bem_progress: bool = False,
) -> dict[str, object]:
    """Build, solve, trace, and evaluate one outer-contour deformation."""
    row = empty_result(
        label=label,
        outer_bulge_amplitude_m=outer_bulge_amplitude_m,
    )

    parameters = make_parameters(outer_bulge_amplitude_m)

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

        title = (
            f"{label}: outer bulge="
            f"{outer_bulge_amplitude_m * 1e6:+.1f} um, "
            r"$s_c=70$ um, $\sigma=25$ um"
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
    print("Step 06d: resolved local outer-contour bulge scan")
    print("=" * 124)
    print(
        "Base geometry: "
        "start=30 um, inner=-25 um, outer=+20 um, "
        "L=150 um, p=2.0"
    )
    print(
        "Local outer feature: "
        "centre=70 um, sigma=25 um"
    )
    print(
        f"{'index':>5} {'bulge [um]':>12} {'trace':>7} {'path':>7} "
        f"{'peak dz [um]':>15} {'rms dz [um]':>14} "
        f"{'barrier [meV]':>16} {'max |y| [um]':>15} {'panels':>8}"
    )


def print_row(index: int, row: dict[str, object]) -> None:
    """Print one evaluation row."""
    print(
        f"{index:5d} "
        f"{float(row['outer_bulge_amplitude_m']) * 1e6:12.2f} "
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
        "outer_bulge_amplitude_m",
        "outer_bulge_center_m",
        "outer_bulge_sigma_m",
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
    """Keep rows with finite physical metrics and a converged trace."""
    return [
        row
        for row in rows
        if bool(row["trace_valid"])
        and np.isfinite(float(row["height_peak_m"]))
        and np.isfinite(float(row["barrier_ev"]))
    ]


def plot_sensitivity(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot RF-null height and barrier versus outer-contour amplitude."""
    valid_rows = valid_rows_only(rows)

    if not valid_rows:
        return

    valid_rows.sort(
        key=lambda row: float(row["outer_bulge_amplitude_m"])
    )

    amplitude_um = np.asarray(
        [
            float(row["outer_bulge_amplitude_m"]) * 1e6
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

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 7.4),
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
        label="No local outer deformation",
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
    axes[1].set_xlabel(r"Outer-edge local deformation amplitude [um]")
    axes[1].set_ylabel("RF barrier [meV]")
    axes[1].grid(alpha=0.3)

    figure.suptitle(
        r"Step 06d: local outer RF contour around $|s|=70$ um"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compromise_score(row: dict[str, object]) -> float:
    """Dimensionless score used only to choose a diagnostic candidate."""
    if not bool(row["trace_valid"]):
        return np.inf

    height_um = float(row["height_peak_m"]) * 1e6
    barrier_mev = float(row["barrier_ev"]) * 1e3

    if not np.isfinite(height_um) or not np.isfinite(barrier_mev):
        return np.inf

    return (height_um / 20.0) ** 2 + (barrier_mev / 100.0) ** 2


def print_selected(
    category: str,
    row: dict[str, object],
) -> None:
    """Print one selected candidate summary."""
    print(
        f"{category:<16} {row['label']}: "
        f"bulge={float(row['outer_bulge_amplitude_m']) * 1e6:+.1f} um, "
        f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
        f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
    )


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "06d_outer_contour_bulge_scan"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Keep the same path range and point count used in 05h, 06b, and 06c.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    # One-dimensional sensitivity scan. Negative values locally pull the
    # outer boundary inward; positive values locally push it outward.
    amplitudes_um = (
        -30.0,
        -15.0,
        0.0,
        15.0,
        30.0,
        45.0,
    )

    print_header()

    rows: list[dict[str, object]] = []

    for index, amplitude_um in enumerate(amplitudes_um, start=1):
        label = f"outer_bulge_{amplitude_um:+.0f}um"

        row = evaluate_outer_bulge(
            label=label,
            outer_bulge_amplitude_m=amplitude_um * 1e-6,
            x_values_m=x_values_m,
            save_diagnostics=False,
            show_bem_progress=True,
        )

        rows.append(row)
        print_row(index, row)

    write_results_csv(
        rows,
        output_path=output_directory / "outer_bulge_scan_results.csv",
    )
    plot_sensitivity(
        rows,
        output_path=output_directory / "outer_bulge_sensitivity.png",
    )

    valid_rows = valid_rows_only(rows)

    if not valid_rows:
        print()
        print("No valid candidates. Inspect manufacturability and trace output.")
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
    best_compromise = min(valid_rows, key=compromise_score)

    selected = {
        "best_height": best_height,
        "best_barrier": best_barrier,
        "best_compromise": best_compromise,
    }

    print()
    print("Selected diagnostic reruns")
    print("=" * 124)

    for category, row in selected.items():
        print_selected(category, row)

        amplitude_m = float(row["outer_bulge_amplitude_m"])

        evaluate_outer_bulge(
            label=f"{category}_{row['label']}",
            outer_bulge_amplitude_m=amplitude_m,
            x_values_m=x_values_m,
            candidate_directory=output_directory / "selected" / category,
            save_diagnostics=True,
            show_bem_progress=True,
        )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()