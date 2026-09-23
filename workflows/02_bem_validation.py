"""Step 2: compare finite BEM strip model against the analytic 2D baseline."""

from __future__ import annotations

import numpy as np

from config.targets import TARGET_ION_HEIGHT_M
from core.electrostatics.bem import BEM2D
from core.electrostatics.bem_mesh import rectangular_strip_panels
from core.electrostatics.surface_2d import build_linear_surface_trap


def relative_error(
    reference: float,
    value: float,
    *,
    near_zero_threshold_v_m: float = 50.0,
) -> float | None:
    """Return relative error unless the reference field is near zero."""
    if abs(reference) < near_zero_threshold_v_m:
        return None
    return 100.0 * (value - reference) / abs(reference)


def main() -> None:
    h_m = TARGET_ION_HEIGHT_M
    ground_width_m = 0.83 * h_m
    rf_width_m = 1.99 * h_m

    analytic = build_linear_surface_trap(
        ground_width_m=ground_width_m,
        rf_width_m=rf_width_m,
    )

    outer_x_m = 15.0 * (ground_width_m + rf_width_m)
    x_edges_m = np.array(
        [
            -outer_x_m,
            -0.5 * ground_width_m - rf_width_m,
            -0.5 * ground_width_m,
            0.5 * ground_width_m,
            0.5 * ground_width_m + rf_width_m,
            outer_x_m,
        ]
    )
    strip_voltages_v = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

    half_length_y_m = 40.0 * h_m

    panels_m, voltages_v = rectangular_strip_panels(
        x_edges_m=x_edges_m,
        electrode_voltages_v=strip_voltages_v,
        y_min_m=-half_length_y_m,
        y_max_m=half_length_y_m,
        panels_per_strip=12,
        graded=True,
        clustering=2.5,
    )

    bem = BEM2D(panels_m, voltages_v)
    bem.solve(show_progress=False)

    print("BEM validation against analytic 2D strip model")
    print("-" * 66)
    print(f"Ground width           : {ground_width_m * 1e6:.3f} um")
    print(f"RF rail width          : {rf_width_m * 1e6:.3f} um")
    print(f"Finite BEM length      : {2.0 * half_length_y_m * 1e6:.1f} um")
    print(f"Number of BEM panels   : {bem.n_panels}")
    print()

    print(
        f"{'z [um]':>9}  "
        f"{'Ez analytic [V/m]':>20}  "
        f"{'Ez BEM [V/m]':>16}  "
        f"{'difference [%]':>16}"
    )

    for z_m in (70e-6, 85e-6, 90e-6, 95e-6, 110e-6):
        _, e_z_analytic = analytic.electric_field(0.0, z_m)
        _, _, e_z_bem = bem.electric_field(0.0, 0.0, z_m)

        error_percent = relative_error(e_z_analytic, e_z_bem)
        error_text = (
            f"{error_percent:16.3f}"
            if error_percent is not None
            else f"{'near RF-null':>16}"
        )

        print(
            f"{z_m * 1e6:9.1f}  "
            f"{e_z_analytic:20.6f}  "
            f"{e_z_bem:16.6f}  "
            f"{error_text}"
        )

    print()
    print("Symmetry check at the nominal height:")
    e_x, e_y, e_z = bem.electric_field(0.0, 0.0, h_m)
    print(f"  E_x(0,0,h) = {e_x:.6e} V/m")
    print(f"  E_y(0,0,h) = {e_y:.6e} V/m")
    print(f"  E_z(0,0,h) = {e_z:.6e} V/m")

    z_grid_m = np.linspace(70e-6, 110e-6, 401)
    e_z_grid_v_m = np.array(
        [bem.electric_field(0.0, 0.0, z_m)[2] for z_m in z_grid_m]
    )

    sign_change_indices = np.where(
        e_z_grid_v_m[:-1] * e_z_grid_v_m[1:] <= 0.0
    )[0]

    if len(sign_change_indices) == 0:
        print("  BEM RF-null: not found in [70, 110] um")
        return

    index = int(sign_change_indices[0])
    z_left_m = z_grid_m[index]
    z_right_m = z_grid_m[index + 1]
    e_left = e_z_grid_v_m[index]
    e_right = e_z_grid_v_m[index + 1]

    z_null_bem_m = z_left_m - e_left * (
        z_right_m - z_left_m
    ) / (e_right - e_left)

    print(f"  BEM RF-null: z = {z_null_bem_m * 1e6:.4f} um")
    print(
        "  Offset from nominal height: "
        f"{(z_null_bem_m - h_m) * 1e6:+.4f} um"
    )


if __name__ == "__main__":
    main()