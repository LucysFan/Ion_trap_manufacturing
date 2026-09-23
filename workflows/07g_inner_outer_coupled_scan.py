"""Step 07g: coupled inner/outer connected RF-contour local scan.

This workflow follows step 07f, where the best resolved inner-only candidate
was:

    d_in(60)  = -12 um
    d_in(90)  = -12 um
    d_in(120) =  -9 um

with:
    peak dz   = 13.3707 um
    barrier   = 100.52408 meV

and the best low-barrier resolved candidate was:

    d_in(60)  = -12 um
    d_in(90)  =  -8 um
    d_in(120) =  -9 um

with:
    peak dz   = 14.8536 um
    barrier   = 76.89695 meV

The next step is not another inner-only refinement. Instead, we keep the
connected inner profile low-dimensional and add one connected outer-profile
amplitude:

    a_out in {-10, -5, 0, +5, +10} um

applied equally at s = 60, 90, and 120 um, with endpoints fixed to zero.

The scanned inner variables are:
    d_in(60)  in {-15, -12, -9} um
    d_in(90)  in {-12, -10, -8} um
    d_in(120) fixed at -9 um

Thus the full coarse scan contains:
    3 * 3 * 5 = 45 coupled candidates
plus:
    baseline
    two resolved 07f seeds with a_out = 0
for easier comparison.

The coarse scan uses 41 transport-path points. Selected unique candidates are
rerun at 81 points and saved with layout and path diagnostics.
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


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

CONTOUR_KNOTS_M = (
    30e-6,
    60e-6,
    90e-6,
    120e-6,
    150e-6,
)

TARGET_HEIGHT_PEAK_M = 3e-6
GO_HEIGHT_PEAK_M = 10e-6
BORDERLINE_HEIGHT_PEAK_M = 13e-6
MAX_GO_BARRIER_RATIO = 1.0
MAX_BORDERLINE_BARRIER_RATIO = 1.10


def complete_inner_offsets_m(
    middle_offsets_um: tuple[float, float, float],
) -> tuple[float, float, float, float, float]:
    """Convert inner offsets at 60, 90, 120 um into the full five-knot profile."""
    if len(middle_offsets_um) != 3:
        raise ValueError(
            "middle_offsets_um must contain inner offsets at 60, 90, 120 um."
        )

    return (
        0.0,
        float(middle_offsets_um[0]) * 1e-6,
        float(middle_offsets_um[1]) * 1e-6,
        float(middle_offsets_um[2]) * 1e-6,
        0.0,
    )


def complete_outer_offsets_m(
    outer_amplitude_um: float,
) -> tuple[float, float, float, float, float]:
    """Create one connected outer profile with equal offsets at 60, 90, 120 um."""
    amplitude_m = float(outer_amplitude_um) * 1e-6
    return (
        0.0,
        amplitude_m,
        amplitude_m,
        amplitude_m,
        0.0,
    )


def make_parameters(
    inner_offsets_m: tuple[float, float, float, float, float],
    outer_offsets_m: tuple[float, float, float, float, float],
):
    """Create the fixed 06c geometry with connected inner and outer contours."""
    if len(inner_offsets_m) != len(CONTOUR_KNOTS_M):
        raise ValueError(
            "inner_offsets_m must contain one value per contour knot."
        )

    if len(outer_offsets_m) != len(CONTOUR_KNOTS_M):
        raise ValueError(
            "outer_offsets_m must contain one value per contour knot."
        )

    if abs(inner_offsets_m[0]) > 1e-15 or abs(inner_offsets_m[-1]) > 1e-15:
        raise ValueError(
            "Inner offsets at 30 and 150 um must be zero."
        )

    if abs(outer_offsets_m[0]) > 1e-15 or abs(outer_offsets_m[-1]) > 1e-15:
        raise ValueError(
            "Outer offsets at 30 and 150 um must be zero."
        )

    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=BASE_TAPER_LENGTH_M,
        inner_edge_shift_at_centre_m=BASE_INNER_SHIFT_M,
        outer_edge_shift_at_centre_m=BASE_OUTER_SHIFT_M,
        taper_power=BASE_TAPER_POWER,
        rf_start_radius_override_m=BASE_START_RADIUS_M,
        inner_contour_knots_m=CONTOUR_KNOTS_M,
        inner_contour_offsets_m=inner_offsets_m,
        outer_contour_knots_m=CONTOUR_KNOTS_M,
        outer_contour_offsets_m=outer_offsets_m,
    )


def make_scan_profiles(
) -> list[tuple[str, tuple[float, float, float], float]]:
    """Create all coupled scan candidates."""
    profiles: list[
        tuple[str, tuple[float, float, float], float]
    ] = [
        (
            "baseline",
            (0.0, 0.0, 0.0),
            0.0,
        ),
        (
            "seed_best_height_07f",
            (-12.0, -12.0, -9.0),
            0.0,
        ),
        (
            "seed_best_barrier_07f",
            (-12.0, -8.0, -9.0),
            0.0,
        ),
    ]

    d60_values_um = (
        -15.0,
        -12.0,
        -9.0,
    )
    d90_values_um = (
        -12.0,
        -10.0,
        -8.0,
    )
    d120_fixed_um = -9.0
    outer_amplitudes_um = (
        -10.0,
        -5.0,
        0.0,
        +5.0,
        +10.0,
    )

    for d60_um, d90_um, outer_um in itertools.product(
        d60_values_um,
        d90_values_um,
        outer_amplitudes_um,
    ):
        label = (
            f"grid_"
            f"in60_{d60_um:+.0f}_"
            f"in90_{d90_um:+.0f}_"
            f"in120_{d120_fixed_um:+.0f}_"
            f"out_{outer_um:+.0f}"
        )

        profiles.append(
            (
                label,
                (d60_um, d90_um, d120_fixed_um),
                outer_um,
            )
        )

    return profiles


def empty_result(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
    outer_offsets_m: tuple[float, float, float, float, float],
    path_point_count: int,
) -> dict[str, object]:
    """Create one stable-schema result row."""
    return {
        "label": label,
        "inner_knot_30_offset_m": inner_offsets_m[0],
        "inner_knot_60_offset_m": inner_offsets_m[1],
        "inner_knot_90_offset_m": inner_offsets_m[2],
        "inner_knot_120_offset_m": inner_offsets_m[3],
        "inner_knot_150_offset_m": inner_offsets_m[4],
        "outer_knot_30_offset_m": outer_offsets_m[0],
        "outer_knot_60_offset_m": outer_offsets_m[1],
        "outer_knot_90_offset_m": outer_offsets_m[2],
        "outer_knot_120_offset_m": outer_offsets_m[3],
        "outer_knot_150_offset_m": outer_offsets_m[4],
        "outer_amplitude_m": outer_offsets_m[2],
        "path_point_count": path_point_count,
        "trace_valid": False,
        "path_valid": False,
        "height_peak_m": np.inf,
        "height_rms_m": np.inf,
        "barrier_ev": np.inf,
        "maximum_y_m": np.inf,
        "peak_improvement_m": np.nan,
        "barrier_change_ev": np.nan,
        "barrier_ratio": np.nan,
        "compromise_score": np.inf,
        "pareto_optimal": False,
        "n_panels": 0,
        "runtime_s": 0.0,
        "reason": "",
        "candidate_directory": "",
    }


def evaluate_coupled_profile(
    *,
    label: str,
    inner_offsets_m: tuple[float, float, float, float, float],
    outer_offsets_m: tuple[float, float, float, float, float],
    x_values_m: np.ndarray,
    candidate_directory: Path | None = None,
    save_diagnostics: bool = False,
    show_bem_progress: bool = False,
) -> dict[str, object]:
    """Build, solve, trace, and evaluate one coupled inner/outer contour."""
    started = time.perf_counter()

    row = empty_result(
        label=label,
        inner_offsets_m=inner_offsets_m,
        outer_offsets_m=outer_offsets_m,
        path_point_count=len(x_values_m),
    )

    try:
        parameters = make_parameters(
            inner_offsets_m,
            outer_offsets_m,
        )
        manufacturability = check_x_junction_manufacturability(
            parameters
        )
    except ValueError as error:
        row["reason"] = f"invalid geometry: {error}"
        row["runtime_s"] = time.perf_counter() - started
        return row

    if not manufacturability.valid:
        row["reason"] = "; ".join(manufacturability.messages)
        row["runtime_s"] = time.perf_counter() - started
        return row

    try:
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

    except Exception as error:
        row["reason"] = (
            f"BEM failure: {type(error).__name__}: {error}"
        )
        row["runtime_s"] = time.perf_counter() - started
        return row

    try:
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

        barrier = compute_barrier_metrics(
            energies_ev,
            trace,
        )

        validation = validate_rf_transport_path(metrics)

    except Exception as error:
        row["reason"] = (
            f"path evaluation failure: {type(error).__name__}: {error}"
        )
        row["n_panels"] = int(model.n_panels)
        row["runtime_s"] = time.perf_counter() - started
        return row

    if save_diagnostics:
        if candidate_directory is None:
            raise ValueError(
                "candidate_directory is required when save_diagnostics=True."
            )

        candidate_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        title = (
            f"{label}: "
            f"in=({inner_offsets_m[1] * 1e6:+.1f}, "
            f"{inner_offsets_m[2] * 1e6:+.1f}, "
            f"{inner_offsets_m[3] * 1e6:+.1f}) um, "
            f"a_out={outer_offsets_m[2] * 1e6:+.1f} um, "
            f"Npath={len(x_values_m)}"
        )

        plot_candidate_layout(
            model,
            trace,
            output_path=(
                candidate_directory
                / "layout_and_trace.png"
            ),
            title=title,
        )

        plot_candidate_diagnostics(
            trace,
            metrics,
            barrier,
            output_path=(
                candidate_directory
                / "path_diagnostics.png"
            ),
            title=title,
        )

        row["candidate_directory"] = str(candidate_directory)

    row.update(
        {
            "trace_valid": bool(trace.valid),
            "path_valid": bool(validation.valid),
            "height_peak_m": float(metrics.height_peak_deviation_m),
            "height_rms_m": float(metrics.height_rms_deviation_m),
            "barrier_ev": float(barrier.barrier_height_ev),
            "maximum_y_m": float(metrics.maximum_lateral_offset_m),
            "n_panels": int(model.n_panels),
            "runtime_s": time.perf_counter() - started,
            "reason": "; ".join(validation.messages),
        }
    )

    return row


def valid_rows_only(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep only candidates with finite physically meaningful metrics."""
    return [
        row
        for row in rows
        if bool(row["trace_valid"])
        and np.isfinite(float(row["height_peak_m"]))
        and np.isfinite(float(row["height_rms_m"]))
        and np.isfinite(float(row["barrier_ev"]))
        and np.isfinite(float(row["maximum_y_m"]))
    ]


def annotate_results(
    rows: list[dict[str, object]],
) -> None:
    """Add baseline-relative metrics, scores, and a Pareto label."""
    valid_rows = valid_rows_only(rows)

    baseline_candidates = [
        row
        for row in valid_rows
        if row["label"] == "baseline"
    ]

    if not baseline_candidates:
        return

    baseline = baseline_candidates[0]

    baseline_peak_m = float(baseline["height_peak_m"])
    baseline_rms_m = float(baseline["height_rms_m"])
    baseline_barrier_ev = float(baseline["barrier_ev"])

    for row in valid_rows:
        peak_m = float(row["height_peak_m"])
        rms_m = float(row["height_rms_m"])
        barrier_ev = float(row["barrier_ev"])
        maximum_y_m = float(row["maximum_y_m"])

        row["peak_improvement_m"] = baseline_peak_m - peak_m
        row["barrier_change_ev"] = barrier_ev - baseline_barrier_ev
        row["barrier_ratio"] = (
            barrier_ev
            / max(abs(baseline_barrier_ev), 1e-30)
        )

        row["compromise_score"] = (
            (peak_m / baseline_peak_m) ** 2
            + 0.35 * (rms_m / baseline_rms_m) ** 2
            + 0.55 * (barrier_ev / baseline_barrier_ev) ** 2
            + 0.10 * (maximum_y_m / 5e-6) ** 2
        )

    for row in valid_rows:
        row_peak = float(row["height_peak_m"])
        row_barrier = float(row["barrier_ev"])
        dominated = False

        for other in valid_rows:
            if other is row:
                continue

            other_peak = float(other["height_peak_m"])
            other_barrier = float(other["barrier_ev"])

            no_worse = (
                other_peak <= row_peak
                and other_barrier <= row_barrier
            )
            strictly_better = (
                other_peak < row_peak
                or other_barrier < row_barrier
            )

            if no_worse and strictly_better:
                dominated = True
                break

        row["pareto_optimal"] = not dominated


def result_fieldnames() -> tuple[str, ...]:
    """Return the fixed CSV column order."""
    return (
        "label",
        "inner_knot_30_offset_m",
        "inner_knot_60_offset_m",
        "inner_knot_90_offset_m",
        "inner_knot_120_offset_m",
        "inner_knot_150_offset_m",
        "outer_knot_30_offset_m",
        "outer_knot_60_offset_m",
        "outer_knot_90_offset_m",
        "outer_knot_120_offset_m",
        "outer_knot_150_offset_m",
        "outer_amplitude_m",
        "path_point_count",
        "trace_valid",
        "path_valid",
        "height_peak_m",
        "height_rms_m",
        "barrier_ev",
        "maximum_y_m",
        "peak_improvement_m",
        "barrier_change_ev",
        "barrier_ratio",
        "compromise_score",
        "pareto_optimal",
        "n_panels",
        "runtime_s",
        "reason",
        "candidate_directory",
    )


def write_results_csv(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Write all scan results to CSV."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result_fieldnames(),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def print_header(
    candidate_count: int,
) -> None:
    """Print workflow configuration and the console table header."""
    print("Step 07g: coupled inner/outer connected RF-contour scan")
    print("=" * 192)
    print(
        "Base geometry: start=30 um, inner=-25 um, outer=+20 um, "
        "L=150 um, p=2.0"
    )
    print(
        f"Candidates: {candidate_count}; coarse transport trace: 41 points."
    )
    print(
        "Inner scan: d60 in {-15,-12,-9}, d90 in {-12,-10,-8}, "
        "d120 fixed at -9 um."
    )
    print(
        "Outer scan: a_out in {-10,-5,0,+5,+10} um, applied equally at "
        "60, 90, 120 um; endpoints at 30 and 150 um are zero."
    )
    print(
        f"{'index':>5} "
        f"{'label':<44} "
        f"{'d60':>7} "
        f"{'d90':>7} "
        f"{'d120':>7} "
        f"{'a_out':>8} "
        f"{'trace':>7} "
        f"{'path':>7} "
        f"{'peak dz':>11} "
        f"{'rms dz':>11} "
        f"{'barrier':>13} "
        f"{'panels':>8} "
        f"{'time':>8}"
    )
    print(
        f"{'':>5} "
        f"{'':<44} "
        f"{'[um]':>7} "
        f"{'[um]':>7} "
        f"{'[um]':>7} "
        f"{'[um]':>8} "
        f"{'':>7} "
        f"{'':>7} "
        f"{'[um]':>11} "
        f"{'[um]':>11} "
        f"{'[meV]':>13} "
        f"{'':>8} "
        f"{'[s]':>8}"
    )


def format_result_row(
    index: int,
    row: dict[str, object],
) -> str:
    """Format one result row for console output."""
    return (
        f"{index:5d} "
        f"{str(row['label']):<44} "
        f"{float(row['inner_knot_60_offset_m']) * 1e6:7.1f} "
        f"{float(row['inner_knot_90_offset_m']) * 1e6:7.1f} "
        f"{float(row['inner_knot_120_offset_m']) * 1e6:7.1f} "
        f"{float(row['outer_amplitude_m']) * 1e6:8.1f} "
        f"{str(bool(row['trace_valid'])):>7} "
        f"{str(bool(row['path_valid'])):>7} "
        f"{float(row['height_peak_m']) * 1e6:11.4f} "
        f"{float(row['height_rms_m']) * 1e6:11.4f} "
        f"{float(row['barrier_ev']) * 1e3:13.5f} "
        f"{int(row['n_panels']):8d} "
        f"{float(row['runtime_s']):8.2f}"
    )


def plot_peak_barrier_pareto(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot peak-height versus barrier with Pareto highlighting."""
    valid_rows = valid_rows_only(rows)

    if not valid_rows:
        return

    figure, axis = plt.subplots(figsize=(10.0, 7.6))

    for row in valid_rows:
        peak_um = float(row["height_peak_m"]) * 1e6
        barrier_mev = float(row["barrier_ev"]) * 1e3

        if row["label"] == "baseline":
            color = "black"
            marker = "s"
            size = 90
            zorder = 5
        elif bool(row["pareto_optimal"]):
            color = "tab:red"
            marker = "o"
            size = 72
            zorder = 4
        else:
            color = "tab:blue"
            marker = "o"
            size = 40
            zorder = 2

        axis.scatter(
            peak_um,
            barrier_mev,
            color=color,
            marker=marker,
            s=size,
            alpha=0.85,
            zorder=zorder,
        )

        if row["label"] == "baseline" or bool(row["pareto_optimal"]):
            axis.annotate(
                str(row["label"]),
                (peak_um, barrier_mev),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.2,
            )

    axis.axvline(
        3.0,
        color="tab:green",
        linestyle=":",
        linewidth=1.5,
        label=r"Target peak $\Delta z=3$ um",
    )

    axis.axvline(
        10.0,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label=r"GO threshold: 10 um",
    )

    axis.set_xlabel(r"Peak $\Delta z$ [um]")
    axis.set_ylabel("RF barrier [meV]")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_outer_sensitivity(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot the best peak and barrier reached for each outer amplitude."""
    valid_rows = [
        row
        for row in valid_rows_only(rows)
        if row["label"] != "baseline"
    ]

    if not valid_rows:
        return

    grouped: dict[float, list[dict[str, object]]] = {}

    for row in valid_rows:
        key_um = float(row["outer_amplitude_m"]) * 1e6
        grouped.setdefault(key_um, []).append(row)

    outer_values_um = sorted(grouped.keys())

    best_peak_um = np.asarray(
        [
            min(
                float(row["height_peak_m"]) * 1e6
                for row in grouped[value_um]
            )
            for value_um in outer_values_um
        ]
    )

    best_barrier_mev = np.asarray(
        [
            min(
                float(row["barrier_ev"]) * 1e3
                for row in grouped[value_um]
            )
            for value_um in outer_values_um
        ]
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 8.4),
        sharex=True,
    )

    axes[0].plot(
        outer_values_um,
        best_peak_um,
        marker="o",
        linewidth=2.0,
        color="tab:blue",
    )
    axes[0].axhline(
        10.0,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
    )
    axes[0].set_ylabel(r"Best peak $\Delta z$ [um]")
    axes[0].set_title("07g outer-amplitude sensitivity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        outer_values_um,
        best_barrier_mev,
        marker="o",
        linewidth=2.0,
        color="tab:purple",
    )
    axes[1].set_xlabel(r"Coupled outer amplitude $a_{\rm out}$ [um]")
    axes[1].set_ylabel("Best barrier [meV]")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_inner_outer_map(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot d60 vs d90 colored by peak, grouped across outer amplitudes."""
    valid_rows = [
        row
        for row in valid_rows_only(rows)
        if row["label"] != "baseline"
    ]

    if not valid_rows:
        return

    d60_um = np.asarray(
        [float(row["inner_knot_60_offset_m"]) * 1e6 for row in valid_rows]
    )
    d90_um = np.asarray(
        [float(row["inner_knot_90_offset_m"]) * 1e6 for row in valid_rows]
    )
    peak_um = np.asarray(
        [float(row["height_peak_m"]) * 1e6 for row in valid_rows]
    )
    outer_um = np.asarray(
        [float(row["outer_amplitude_m"]) * 1e6 for row in valid_rows]
    )

    marker_sizes = 45.0 + 8.0 * (np.max(outer_um) - outer_um + 10.0)

    figure, axis = plt.subplots(figsize=(9.2, 7.5))

    scatter = axis.scatter(
        d60_um,
        d90_um,
        c=peak_um,
        s=marker_sizes,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.85,
    )

    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(r"Peak $\Delta z$ [um]")

    axis.set_xlabel(r"$\delta_{\rm in}(60)$ [um]")
    axis.set_ylabel(r"$\delta_{\rm in}(90)$ [um]")
    axis.set_title(
        "07g inner-control map\n"
        "marker size increases with more-negative outer amplitude"
    )
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def row_profiles(
    row: dict[str, object],
) -> tuple[
    tuple[float, float, float, float, float],
    tuple[float, float, float, float, float],
]:
    """Extract the full inner and outer profiles from a result row."""
    inner_offsets_m = (
        float(row["inner_knot_30_offset_m"]),
        float(row["inner_knot_60_offset_m"]),
        float(row["inner_knot_90_offset_m"]),
        float(row["inner_knot_120_offset_m"]),
        float(row["inner_knot_150_offset_m"]),
    )

    outer_offsets_m = (
        float(row["outer_knot_30_offset_m"]),
        float(row["outer_knot_60_offset_m"]),
        float(row["outer_knot_90_offset_m"]),
        float(row["outer_knot_120_offset_m"]),
        float(row["outer_knot_150_offset_m"]),
    )

    return inner_offsets_m, outer_offsets_m


def print_selected(
    category: str,
    row: dict[str, object],
) -> None:
    """Print one selected candidate."""
    print(
        f"{category:<20} "
        f"{str(row['label']):<44} "
        f"in=("
        f"{float(row['inner_knot_60_offset_m']) * 1e6:+.1f}, "
        f"{float(row['inner_knot_90_offset_m']) * 1e6:+.1f}, "
        f"{float(row['inner_knot_120_offset_m']) * 1e6:+.1f}) um, "
        f"a_out={float(row['outer_amplitude_m']) * 1e6:+.1f} um, "
        f"peak={float(row['height_peak_m']) * 1e6:.4f} um, "
        f"rms={float(row['height_rms_m']) * 1e6:.4f} um, "
        f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV, "
        f"score={float(row['compromise_score']):.5f}"
    )


def print_decision(
    coarse_rows: list[dict[str, object]],
    resolved_rows: list[dict[str, object]],
) -> None:
    """Print the next-step decision after the resolved coupled scan."""
    coarse_valid = valid_rows_only(coarse_rows)
    resolved_valid = valid_rows_only(resolved_rows)

    baseline_candidates = [
        row
        for row in resolved_valid
        if row["label"] == "baseline"
    ]

    modified_resolved = [
        row
        for row in resolved_valid
        if row["label"] != "baseline"
    ]

    print()
    print("07g coupled-family assessment")
    print("=" * 120)

    if not baseline_candidates:
        print("No valid resolved baseline candidate.")
        return

    if not modified_resolved:
        print("No valid resolved modified candidate.")
        return

    baseline = baseline_candidates[0]
    best_height = min(
        modified_resolved,
        key=lambda row: float(row["height_peak_m"]),
    )
    best_barrier = min(
        modified_resolved,
        key=lambda row: float(row["barrier_ev"]),
    )

    baseline_peak_m = float(baseline["height_peak_m"])
    baseline_barrier_ev = float(baseline["barrier_ev"])

    best_height_peak_m = float(best_height["height_peak_m"])
    best_height_barrier_ev = float(best_height["barrier_ev"])
    best_height_barrier_ratio = (
        best_height_barrier_ev
        / max(abs(baseline_barrier_ev), 1e-30)
    )

    print(
        f"Resolved baseline peak dz: {baseline_peak_m * 1e6:.4f} um"
    )
    print(
        f"Best resolved candidate: {best_height['label']}"
    )
    print(
        f"Best resolved peak dz: {best_height_peak_m * 1e6:.4f} um"
    )
    print(
        f"Peak improvement relative to resolved baseline: "
        f"{(baseline_peak_m - best_height_peak_m) * 1e6:+.4f} um"
    )
    print(
        f"Best-height barrier: {best_height_barrier_ev * 1e3:.5f} meV"
    )
    print(
        f"Best-height barrier ratio: {best_height_barrier_ratio:.4f}"
    )
    print(
        f"Lowest resolved barrier candidate: {best_barrier['label']}, "
        f"{float(best_barrier['barrier_ev']) * 1e3:.5f} meV"
    )

    if (
        best_height_peak_m <= GO_HEIGHT_PEAK_M
        and best_height_barrier_ratio <= MAX_GO_BARRIER_RATIO
    ):
        print(
            "Decision: GO. Start a constrained joint optimizer from this "
            "inner/outer family."
        )
    elif (
        best_height_peak_m <= BORDERLINE_HEIGHT_PEAK_M
        and best_height_barrier_ratio <= MAX_BORDERLINE_BARRIER_RATIO
    ):
        print(
            "Decision: BORDERLINE. The coupled family has leverage, but still "
            "cannot reach the 10 um screening threshold."
        )
    else:
        print(
            "Decision: NO-GO. This low-dimensional coupled family is not enough; "
            "move to richer outer control or central-topology change."
        )


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07g_inner_outer_coupled_scan"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    coarse_x_values_m = np.linspace(
        -350e-6,
        350e-6,
        41,
    )

    resolved_x_values_m = np.linspace(
        -350e-6,
        350e-6,
        81,
    )

    profiles = make_scan_profiles()

    print_header(candidate_count=len(profiles))

    coarse_rows: list[dict[str, object]] = []

    progress = tqdm(
        profiles,
        total=len(profiles),
        desc="07g coarse coupled scan",
        unit="candidate",
        dynamic_ncols=True,
    )

    for index, (label, inner_middle_um, outer_um) in enumerate(progress, start=1):
        progress.set_postfix_str(label)

        row = evaluate_coupled_profile(
            label=label,
            inner_offsets_m=complete_inner_offsets_m(inner_middle_um),
            outer_offsets_m=complete_outer_offsets_m(outer_um),
            x_values_m=coarse_x_values_m,
            save_diagnostics=False,
            show_bem_progress=False,
        )

        coarse_rows.append(row)

        tqdm.write(format_result_row(index, row))

        if row["reason"] and not bool(row["trace_valid"]):
            tqdm.write(f"      reason: {row['reason']}")

    annotate_results(coarse_rows)

    write_results_csv(
        coarse_rows,
        output_path=output_directory / "coarse_coupled_results.csv",
    )

    plot_peak_barrier_pareto(
        coarse_rows,
        output_path=output_directory / "coarse_peak_barrier_pareto.png",
        title=(
            "07g coarse coupled scan: height-barrier trade-off\n"
            "red = non-dominated candidate"
        ),
    )

    plot_outer_sensitivity(
        coarse_rows,
        output_path=output_directory / "coarse_outer_sensitivity.png",
    )

    plot_inner_outer_map(
        coarse_rows,
        output_path=output_directory / "coarse_inner_outer_map.png",
    )

    valid_rows = valid_rows_only(coarse_rows)

    if not valid_rows:
        print()
        print(
            "No valid candidates. Inspect geometry validation, BEM solve, "
            "and RF-null tracing."
        )
        print(f"Partial results saved to: {output_directory}")
        return

    modified_rows = [
        row
        for row in valid_rows
        if row["label"] != "baseline"
    ]

    if not modified_rows:
        print()
        print("Only the baseline candidate is valid.")
        return

    best_height = min(
        modified_rows,
        key=lambda row: float(row["height_peak_m"]),
    )

    best_barrier = min(
        modified_rows,
        key=lambda row: float(row["barrier_ev"]),
    )

    best_rms = min(
        modified_rows,
        key=lambda row: float(row["height_rms_m"]),
    )

    best_compromise = min(
        modified_rows,
        key=lambda row: float(row["compromise_score"]),
    )

    pareto_rows = [
        row
        for row in modified_rows
        if bool(row["pareto_optimal"])
    ]

    best_pareto = (
        min(
            pareto_rows,
            key=lambda row: float(row["compromise_score"]),
        )
        if pareto_rows
        else best_compromise
    )

    selected = {
        "best_height": best_height,
        "best_barrier": best_barrier,
        "best_rms": best_rms,
        "best_compromise": best_compromise,
        "best_pareto": best_pareto,
    }

    print()
    print("Selected coarse candidates")
    print("=" * 192)

    for category, row in selected.items():
        print_selected(category, row)

    baseline_row = next(
        (
            row
            for row in valid_rows
            if row["label"] == "baseline"
        ),
        None,
    )

    unique_selected: dict[str, dict[str, object]] = {}

    if baseline_row is not None:
        unique_selected["baseline"] = baseline_row

    for row in selected.values():
        unique_selected.setdefault(str(row["label"]), row)

    resolved_rows: list[dict[str, object]] = []

    resolved_progress = tqdm(
        unique_selected.values(),
        total=len(unique_selected),
        desc="07g 81-point reruns",
        unit="candidate",
        dynamic_ncols=True,
    )

    for row in resolved_progress:
        source_label = str(row["label"])
        resolved_progress.set_postfix_str(source_label)

        inner_offsets_m, outer_offsets_m = row_profiles(row)

        rerun = evaluate_coupled_profile(
            label=f"resolved_{source_label}",
            inner_offsets_m=inner_offsets_m,
            outer_offsets_m=outer_offsets_m,
            x_values_m=resolved_x_values_m,
            candidate_directory=(
                output_directory
                / "resolved"
                / source_label
            ),
            save_diagnostics=True,
            show_bem_progress=False,
        )

        resolved_rows.append(rerun)

        tqdm.write(
            format_result_row(
                len(resolved_rows),
                rerun,
            )
        )

    for row in resolved_rows:
        if row["label"] == "resolved_baseline":
            row["label"] = "baseline"
            break

    annotate_results(resolved_rows)

    write_results_csv(
        resolved_rows,
        output_path=output_directory / "resolved_81_point_results.csv",
    )

    plot_peak_barrier_pareto(
        resolved_rows,
        output_path=output_directory / "resolved_peak_barrier_pareto.png",
        title=(
            "07g resolved coupled scan: height-barrier trade-off\n"
            "red = non-dominated candidate"
        ),
    )

    print_decision(
        coarse_rows,
        resolved_rows,
    )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()