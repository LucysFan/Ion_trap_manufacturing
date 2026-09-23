"""Step 3: build and visualize the first symmetric X-junction baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.targets import (
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
)
from core.analysis.curvature import secular_frequencies_at_null_hz
from core.analysis.rf_null import find_rf_null
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import build_x_junction_bem
from core.visualization.junction import (
    plot_field_squared_map,
    plot_x_junction_mask,
    plot_xz_field_squared_slice,
)


def main() -> None:
    output_directory = Path("reports") / "figures" / "03_single_junction"
    output_directory.mkdir(parents=True, exist_ok=True)

    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        central_island_radius_m=0.0,
    )

    report = check_x_junction_manufacturability(parameters)
    if not report.valid:
        raise RuntimeError("Invalid geometry: " + "; ".join(report.messages))

    print("Single X-junction baseline")
    print("-" * 64)
    print(
        f"Ground rail width : {parameters.ground_rail_width_m * 1e6:.3f} um"
    )
    print(f"RF rail width     : {parameters.rf_rail_width_m * 1e6:.3f} um")
    print(f"Arm length        : {parameters.arm_length_m * 1e6:.3f} um")
    print(f"BEM outer extent  : {parameters.outer_extent_m * 1e6:.3f} um")
    print(f"Central island    : {parameters.central_island_kind}")
    print()

    model = build_x_junction_bem(
        parameters,
        edge_aligned=False,
        n_central=2,
        n_ground=2,
        n_rf=5,
        n_arm=5,
        n_outer=2,
    )

    print(f"BEM panel count   : {model.n_panels}")
    print("Assembling BEM matrix...")
    model.bem.assemble(show_progress=True)

    print("Solving BEM system...")
    model.bem.solve(show_progress=False)
    print()

    centre_null = find_rf_null(
        model.bem,
        initial_guess_m=np.array([0.0, 0.0, TARGET_ION_HEIGHT_M]),
    )
    arm_null = find_rf_null(
        model.bem,
        initial_guess_m=np.array([300e-6, 0.0, TARGET_ION_HEIGHT_M]),
    )

    print("RF-null results")
    print("-" * 64)

    for name, result in (
        ("Junction centre", centre_null),
        ("Horizontal arm", arm_null),
    ):
        print(f"{name}:")
        print(f"  converged     : {result.converged}")
        print(
            "  position [um] : "
            f"({result.x_m * 1e6:.4f}, "
            f"{result.y_m * 1e6:.4f}, "
            f"{result.z_m * 1e6:.4f})"
        )
        print(f"  |E_RF|        : {result.residual_field_v_m:.3e} V/m")
        print(f"  function evals: {result.n_function_evaluations}")
        print(f"  message       : {result.message}")

    if centre_null.converged:
        frequencies_hz, eigenvalues, _ = secular_frequencies_at_null_hz(
            model.bem,
            centre_null.position_m,
            rf_voltage_peak_v=100.0,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        )

        print()
        print("Secular frequencies at junction centre for V_RF = 100 V")
        print(
            "  "
            + ", ".join(
                f"{frequency_hz / 1e6:.4f} MHz"
                for frequency_hz in frequencies_hz
            )
        )
        print(
            "  Hessian eigenvalues: "
            + ", ".join(f"{value:.3e}" for value in eigenvalues)
        )

    if centre_null.converged and arm_null.converged:
        delta_z_m = centre_null.z_m - arm_null.z_m
        print()
        print(
            "Two-point volcano estimate "
            f"(centre minus arm): {delta_z_m * 1e6:+.4f} um"
        )
        print(
            "This is only a preliminary diagnostic. "
            "The next workflow traces the full branch."
        )

    print()
    print("Creating figures...")

    plot_x_junction_mask(
        model,
        output_path=output_directory / "junction_mask.png",
    )

    plot_field_squared_map(
        model,
        z_m=TARGET_ION_HEIGHT_M,
        output_path=output_directory / "field_squared_z90um.png",
        x_limits_m=(-450e-6, 450e-6),
        y_limits_m=(-450e-6, 450e-6),
        n_points=51,
    )

    plot_xz_field_squared_slice(
        model,
        output_path=output_directory / "field_squared_xz.png",
        y_m=0.0,
        x_limits_m=(-450e-6, 450e-6),
        z_limits_m=(15e-6, 180e-6),
        n_x=71,
        n_z=51,
    )

    print("Figures saved to:")
    print(f"  {output_directory / 'junction_mask.png'}")
    print(f"  {output_directory / 'field_squared_z90um.png'}")
    print(f"  {output_directory / 'field_squared_xz.png'}")


if __name__ == "__main__":
    main()