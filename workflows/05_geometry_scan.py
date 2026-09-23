"""Step 5: controlled scan of square and circular central ground islands."""

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
from core.geometry.junction_templates import (
    ISLAND_CIRCLE,
    ISLAND_NONE,
    ISLAND_SQUARE,
    make_house_style_x_junction,
)
from core.geometry.mask_builder import build_x_junction_bem
from core.visualization.geometry_scan import (
    plot_height_profiles,
    plot_island_scan_summary,
)


def evaluate_candidate(
    *,
    island_kind: str,
    island_radius_m: float,
    x_values_m: np.ndarray,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Build, solve, trace, and summarize one controlled candidate."""
    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        central_island_radius_m=island_radius_m,
        central_island_kind=island_kind,
    )

    model = build_x_junction_bem(
        parameters,
        edge_aligned=False,
        n_central=2,
        n_ground=2,
        n_rf=4,
        n_arm=4,
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
        "kind": island_kind,
        "radius_m": island_radius_m,
        "trace_valid": trace.valid,
        "height_peak_m": metrics.height_peak_deviation_m,
        "height_rms_m": metrics.height_rms_deviation_m,
        "barrier_ev": barrier.barrier_height_ev,
        "maximum_y_m": metrics.maximum_lateral_offset_m,
        "n_panels": model.n_panels,
    }

    profile: dict[str, object] | None = None
    if trace.valid:
        profile = {
            "x_m": trace.x_m,
            "deviation_m": trace.z_m - metrics.arm_reference_height_m,
            "label": (
                f"{island_kind}, r={island_radius_m * 1e6:.0f} um, "
                f"peak={metrics.height_peak_deviation_m * 1e6:.1f} um"
            ),
        }

    return row, profile


def main() -> None:
    output_directory = Path("reports") / "figures" / "05_geometry_scan"
    output_directory.mkdir(parents=True, exist_ok=True)

    x_values_m = np.linspace(-350e-6, 350e-6, 51)

    scan_plan: list[tuple[str, float]] = [
        (ISLAND_NONE, 0.0),
        (ISLAND_SQUARE, 20e-6),
        (ISLAND_SQUARE, 40e-6),
        (ISLAND_SQUARE, 60e-6),
        (ISLAND_SQUARE, 80e-6),
        (ISLAND_SQUARE, 100e-6),
        (ISLAND_CIRCLE, 20e-6),
        (ISLAND_CIRCLE, 40e-6),
        (ISLAND_CIRCLE, 60e-6),
        (ISLAND_CIRCLE, 80e-6),
        (ISLAND_CIRCLE, 100e-6),
    ]

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []

    print("Controlled central-island scan")
    print("-" * 92)
    print(
        f"{'kind':<10} {'r [um]':>8} {'trace':>8} "
        f"{'peak dz [um]':>15} {'rms dz [um]':>14} "
        f"{'barrier [meV]':>16} {'panels':>8}"
    )

    for index, (kind, radius_m) in enumerate(scan_plan, start=1):
        print(
            f"[{index:02d}/{len(scan_plan):02d}] "
            f"evaluating {kind}, r={radius_m * 1e6:.1f} um..."
        )

        row, profile = evaluate_candidate(
            island_kind=kind,
            island_radius_m=radius_m,
            x_values_m=x_values_m,
        )
        results.append(row)

        if profile is not None:
            profiles.append(profile)

        print(
            f"{str(row['kind']):<10} "
            f"{float(row['radius_m']) * 1e6:8.1f} "
            f"{str(bool(row['trace_valid'])):>8} "
            f"{float(row['height_peak_m']) * 1e6:15.4f} "
            f"{float(row['height_rms_m']) * 1e6:14.4f} "
            f"{float(row['barrier_ev']) * 1e3:16.5f} "
            f"{int(row['n_panels']):8d}"
        )

    valid_results = [row for row in results if bool(row["trace_valid"])]
    valid_results.sort(key=lambda row: float(row["height_peak_m"]))

    print()
    print("Best candidates by peak RF-null height deviation")
    print("-" * 92)

    for rank, row in enumerate(valid_results[:5], start=1):
        print(
            f"{rank}. {str(row['kind'])}, "
            f"r={float(row['radius_m']) * 1e6:.1f} um: "
            f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
            f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
        )

    plot_island_scan_summary(
        results,
        output_path=output_directory / "island_scan_summary.png",
    )

    selected_profiles: list[dict[str, object]] = []

    if profiles:
        baseline_profiles = [
            profile
            for profile in profiles
            if profile["label"].startswith("none")
        ]
        selected_profiles.extend(baseline_profiles[:1])

        for row in valid_results[:4]:
            target_label_start = (
                f"{str(row['kind'])}, r={float(row['radius_m']) * 1e6:.0f} um"
            )
            matches = [
                profile
                for profile in profiles
                if str(profile["label"]).startswith(target_label_start)
            ]
            if matches:
                selected_profiles.append(matches[0])

    if selected_profiles:
        plot_height_profiles(
            selected_profiles,
            output_path=output_directory / "selected_height_profiles.png",
        )

    np.savez(
        output_directory / "scan_results.npz",
        kind=np.asarray([str(row["kind"]) for row in results]),
        radius_m=np.asarray([float(row["radius_m"]) for row in results]),
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
    print(f"Saved scan data and figures to: {output_directory}")


if __name__ == "__main__":
    main()