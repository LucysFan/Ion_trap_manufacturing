"""Step 4: trace RF transverse minimum through baseline X-junction."""

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
from core.analysis.validation import validate_rf_transport_path
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.mask_builder import build_x_junction_bem
from core.visualization.rf_null_path import (
    plot_height_deviation,
    plot_path_metrics,
    plot_trace_on_geometry,
)


def main() -> None:
    output_directory = Path("reports") / "figures" / "04_rf_null_trace"
    output_directory.mkdir(parents=True, exist_ok=True)

    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        central_island_radius_m=0.0,
    )

    model = build_x_junction_bem(
        parameters,
        edge_aligned=False,
        n_central=2,
        n_ground=2,
        n_rf=5,
        n_arm=5,
        n_outer=2,
    )

    print("Building BEM for RF-null trace...")
    print(f"Panel count: {model.n_panels}")

    model.bem.assemble(show_progress=True)
    model.bem.solve(show_progress=False)

    x_values_m = np.linspace(-350e-6, 350e-6, 71)

    print("Tracing transverse RF minimum...")
    trace = trace_rf_transverse_minimum(
        model.bem,
        x_values_m,
        initial_y_m=0.0,
        initial_z_m=TARGET_ION_HEIGHT_M,
        residual_tolerance_v_m=1e-3,
        max_transverse_shift_m=25e-6,
    )

    metrics = compute_path_metrics(trace)
    report = validate_rf_transport_path(metrics)

    energies_ev = pseudopotential_profile_ev(
        model.bem,
        trace,
        rf_voltage_peak_v=100.0,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )
    barrier = compute_barrier_metrics(energies_ev, trace)

    print()
    print("RF-null trace summary")
    print("-" * 68)
    print(f"Trace complete              : {trace.valid}")
    print(
        f"Converged path points       : "
        f"{metrics.n_converged}/{metrics.n_points}"
    )
    print(
        f"Arm reference height        : "
        f"{metrics.arm_reference_height_m * 1e6:.4f} um"
    )
    print(
        f"Peak height deviation       : "
        f"{metrics.height_peak_deviation_m * 1e6:.4f} um"
    )
    print(
        f"RMS height deviation        : "
        f"{metrics.height_rms_deviation_m * 1e6:.4f} um"
    )
    print(
        f"Peak-to-peak height change  : "
        f"{metrics.height_peak_to_peak_m * 1e6:.4f} um"
    )
    print(
        f"Maximum |y|                 : "
        f"{metrics.maximum_lateral_offset_m * 1e6:.4f} um"
    )
    print(
        f"Maximum |E_yz|              : "
        f"{metrics.maximum_transverse_residual_v_m:.3e} V/m"
    )
    print(
        f"Maximum |E|                 : "
        f"{metrics.maximum_full_field_norm_v_m:.3e} V/m"
    )
    print(
        f"RF pseudo-barrier @100 V    : "
        f"{barrier.barrier_height_ev:.3e} eV"
    )
    print(f"Path passes nominal checks  : {report.valid}")

    if report.messages != ("ok",):
        print("Validation notes:")
        for message in report.messages:
            print(f"  - {message}")

    if not trace.valid:
        failure_index = trace.first_failure_index
        if failure_index is not None:
            print()
            print("First trace failure:")
            print(trace.messages[failure_index])

    print()
    print("Saving figures...")

    plot_trace_on_geometry(
        model,
        trace,
        output_path=output_directory / "trace_on_geometry.png",
    )
    plot_height_deviation(
        trace,
        metrics,
        output_path=output_directory / "height_deviation.png",
    )
    plot_path_metrics(
        trace,
        metrics,
        barrier,
        output_path=output_directory / "path_diagnostics.png",
    )

    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    main()