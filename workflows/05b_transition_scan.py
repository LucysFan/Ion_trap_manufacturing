"""Step 5b: controlled scan of smooth RF-rail transition profiles."""

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
from core.visualization.geometry_scan import (
    plot_height_profiles,
    plot_transition_scan_summary,
)


def evaluate_transition(
    *,
    label: str,
    taper_length_m: float,
    inner_shift_m: float,
    outer_shift_m: float,
    taper_power: float,
    x_values_m: np.ndarray,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Evaluate one taper geometry with BEM and transverse tracing."""
    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=taper_length_m,
        inner_edge_shift_at_centre_m=inner_shift_m,
        outer_edge_shift_at_centre_m=outer_shift_m,
        taper_power=taper_power,
    )

    try:
        manufacturability = check_x_junction_manufacturability(parameters)
    except ValueError as error:
        return (
            {
                "label": label,
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
                "label": label,
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
        "label": label,
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
                f"{label}: "
                f"peak={metrics.height_peak_deviation_m * 1e6:.1f} um"
            ),
        }

    return row, profile


def main() -> None:
    output_directory = Path("reports") / "figures" / "05b_transition_scan"
    output_directory.mkdir(parents=True, exist_ok=True)

    x_values_m = np.linspace(-350e-6, 350e-6, 51)
    taper_length_m = 150e-6
    taper_power = 2.0

    scan_plan_um: list[tuple[str, float, float]] = [
        ("baseline", 0.0, 0.0),
        ("inner_bulge_20", -20.0, 0.0),
        ("inner_bulge_40", -40.0, 0.0),
        ("outer_expand_20", 0.0, 20.0),
        ("outer_expand_40", 0.0, 40.0),
        ("inner_pinch_20", 20.0, 0.0),
        ("inner_pinch_40", 40.0, 0.0),
        ("whole_rail_in_20", -20.0, -20.0),
        ("whole_rail_out_20", 20.0, 20.0),
        ("wide_bulge", -20.0, 20.0),
        ("wide_bulge_strong", -40.0, 40.0),
        ("narrow_pinch", 20.0, -20.0),
        ("narrow_pinch_strong", 40.0, -40.0),
    ]

    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []

    print("Controlled RF-rail transition scan")
    print("-" * 108)
    print(
        f"{'label':<22} {'din [um]':>10} {'dout [um]':>11} "
        f"{'trace':>8} {'peak dz [um]':>15} {'rms dz [um]':>14} "
        f"{'barrier [meV]':>16} {'panels':>8}"
    )

    for index, (label, inner_shift_um, outer_shift_um) in enumerate(
        scan_plan_um,
        start=1,
    ):
        print(f"[{index:02d}/{len(scan_plan_um):02d}] evaluating {label}...")

        row, profile = evaluate_transition(
            label=label,
            taper_length_m=taper_length_m,
            inner_shift_m=inner_shift_um * 1e-6,
            outer_shift_m=outer_shift_um * 1e-6,
            taper_power=taper_power,
            x_values_m=x_values_m,
        )
        row["inner_shift_m"] = inner_shift_um * 1e-6
        row["outer_shift_m"] = outer_shift_um * 1e-6

        results.append(row)

        if profile is not None:
            profiles.append(profile)

        print(
            f"{label:<22} "
            f"{inner_shift_um:10.1f} "
            f"{outer_shift_um:11.1f} "
            f"{str(bool(row['trace_valid'])):>8} "
            f"{float(row['height_peak_m']) * 1e6:15.4f} "
            f"{float(row['height_rms_m']) * 1e6:14.4f} "
            f"{float(row['barrier_ev']) * 1e3:16.5f} "
            f"{int(row.get('n_panels', 0)):8d}"
        )

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
    print("Best controlled transition candidates")
    print("-" * 108)

    for rank, row in enumerate(valid_results[:5], start=1):
        print(
            f"{rank}. {row['label']}: "
            f"peak dz={float(row['height_peak_m']) * 1e6:.4f} um, "
            f"rms dz={float(row['height_rms_m']) * 1e6:.4f} um, "
            f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
        )

    plot_transition_scan_summary(
        results,
        output_path=output_directory / "transition_tradeoff.png",
    )

    selected_profiles: list[dict[str, object]] = []

    for row in valid_results[:5]:
        target_prefix = f"{row['label']}:"
        matching = [
            profile
            for profile in profiles
            if str(profile["label"]).startswith(target_prefix)
        ]

        if matching:
            selected_profiles.append(matching[0])

    if selected_profiles:
        plot_height_profiles(
            selected_profiles,
            output_path=output_directory / "best_height_profiles.png",
        )

    np.savez(
        output_directory / "transition_scan_results.npz",
        label=np.asarray([str(row["label"]) for row in results]),
        trace_valid=np.asarray([bool(row["trace_valid"]) for row in results]),
        inner_shift_m=np.asarray(
            [float(row["inner_shift_m"]) for row in results]
        ),
        outer_shift_m=np.asarray(
            [float(row["outer_shift_m"]) for row in results]
        ),
        height_peak_m=np.asarray(
            [float(row["height_peak_m"]) for row in results]
        ),
        height_rms_m=np.asarray(
            [float(row["height_rms_m"]) for row in results]
        ),
        barrier_ev=np.asarray([float(row["barrier_ev"]) for row in results]),
    )

    print()
    print(f"Saved figures and data to: {output_directory}")


if __name__ == "__main__":
    main()