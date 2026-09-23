"""Step 5c: scan longitudinal RF-rail start radius in an X-junction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import build_x_junction_bem
from core.visualization.geometry_scan import plot_height_profiles


def evaluate_start_radius(
    start_radius_m: float,
    x_values_m: np.ndarray,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Build and evaluate one RF-rail start-radius candidate."""
    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        rf_start_radius_override_m=start_radius_m,
    )

    try:
        manufacturability = check_x_junction_manufacturability(parameters)
    except ValueError as error:
        return (
            {
                "start_radius_m": start_radius_m,
                "trace_valid": False,
                "height_peak_m": np.inf,
                "height_rms_m": np.inf,
                "barrier_ev": np.inf,
                "n_panels": 0,
                "reason": f"Invalid geometry: {error}",
            },
            None,
        )

    if not manufacturability.valid:
        return (
            {
                "start_radius_m": start_radius_m,
                "trace_valid": False,
                "height_peak_m": np.inf,
                "height_rms_m": np.inf,
                "barrier_ev": np.inf,
                "n_panels": 0,
                "reason": "; ".join(manufacturability.messages),
            },
            None,
        )

    model = build_x_junction_bem(
        parameters,
        edge_aligned=False,
        n_central=3,
        n_ground=3,
        n_rf=6,
        n_arm=5,
        n_outer=2,
    )
    model.bem.assemble(show_progress=False)
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

    row: dict[str, object] = {
        "start_radius_m": start_radius_m,
        "trace_valid": trace.valid,
        "height_peak_m": metrics.height_peak_deviation_m,
        "height_rms_m": metrics.height_rms_deviation_m,
        "barrier_ev": barrier.barrier_height_ev,
        "maximum_y_m": metrics.maximum_lateral_offset_m,
        "n_panels": model.n_panels,
        "reason": "ok" if trace.valid else trace.messages[-1],
    }

    profile: dict[str, object] | None = None

    if trace.valid:
        profile = {
            "x_m": trace.x_m,
            "deviation_m": trace.z_m - metrics.arm_reference_height_m,
            "label": (
                f"start={start_radius_m * 1e6:.1f} um, "
                f"peak={metrics.height_peak_deviation_m * 1e6:.1f} um"
            ),
        }

    return row, profile


def main() -> None:
    output_directory = Path("reports") / "figures" / "05c_start_radius_scan"
    output_directory.mkdir(parents=True, exist_ok=True)

    x_values_m = np.linspace(-350e-6, 350e-6, 51)

    baseline_start_radius_m = 0.5 * 0.83 * TARGET_ION_HEIGHT_M
    start_radii_m = np.array(
        [
            baseline_start_radius_m,
            35e-6,
            30e-6,
            25e-6,
            20e-6,
            15e-6,
        ]
    )

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []

    print("Longitudinal RF-rail start-radius scan")
    print("-" * 96)
    print(
        f"{'start [um]':>12} {'trace':>8} {'peak dz [um]':>15} "
        f"{'rms dz [um]':>14} {'barrier [meV]':>16} {'panels':>8}"
    )

    for start_radius_m in start_radii_m:
        print(f"Evaluating start={start_radius_m * 1e6:.2f} um...")

        row, profile = evaluate_start_radius(start_radius_m, x_values_m)
        results.append(row)

        if profile is not None:
            profiles.append(profile)

        print(
            f"{float(row['start_radius_m']) * 1e6:12.3f} "
            f"{str(bool(row['trace_valid'])):>8} "
            f"{float(row['height_peak_m']) * 1e6:15.4f} "
            f"{float(row['height_rms_m']) * 1e6:14.4f} "
            f"{float(row['barrier_ev']) * 1e3:16.5f} "
            f"{int(row['n_panels']):8d}"
        )

        if not bool(row["trace_valid"]):
            print(f"  reason: {row['reason']}")

    valid_results = [
        row
        for row in results
        if bool(row["trace_valid"]) and np.isfinite(float(row["height_peak_m"]))
    ]
    valid_results.sort(
        key=lambda row: (
            float(row["height_peak_m"]),
            float(row["barrier_ev"]),
        )
    )

    print()
    print("Best valid start-radius candidates")
    print("-" * 96)

    for rank, row in enumerate(valid_results, start=1):
        print(
            f"{rank}. start={float(row['start_radius_m']) * 1e6:.2f} um: "
            f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
            f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
        )

    if profiles:
        plot_height_profiles(
            profiles,
            output_path=output_directory / "start_radius_height_profiles.png",
        )

    np.savez(
        output_directory / "start_radius_scan_results.npz",
        start_radius_m=np.asarray(
            [float(row["start_radius_m"]) for row in results]
        ),
        trace_valid=np.asarray([bool(row["trace_valid"]) for row in results]),
        height_peak_m=np.asarray(
            [float(row["height_peak_m"]) for row in results]
        ),
        height_rms_m=np.asarray(
            [float(row["height_rms_m"]) for row in results]
        ),
        barrier_ev=np.asarray([float(row["barrier_ev"]) for row in results]),
    )

    print()
    print(f"Saved results to: {output_directory}")


if __name__ == "__main__":
    main()