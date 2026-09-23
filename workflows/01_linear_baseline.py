"""Step 1: analytic baseline for a straight 40Ca+ surface-electrode trap."""

from __future__ import annotations

from config.targets import (
    MAX_RF_VOLTAGE_PEAK_V,
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
    TARGET_RADIAL_FREQUENCY_HZ,
)
from core.electrostatics.linear_trap import (
    find_rf_null_on_axis,
    required_rf_voltage_peak_v,
    secular_frequencies_hz,
)
from core.electrostatics.surface_2d import build_linear_surface_trap


def main() -> None:
    ion_height_m = TARGET_ION_HEIGHT_M

    ground_width_m = 0.83 * ion_height_m
    rf_width_m = 1.99 * ion_height_m

    trap = build_linear_surface_trap(
        ground_width_m=ground_width_m,
        rf_width_m=rf_width_m,
    )

    x_null_m, z_null_m = find_rf_null_on_axis(trap)

    print("Straight surface-electrode trap baseline")
    print("-" * 52)
    print(f"Target ion height : {ion_height_m * 1e6:.3f} um")
    print(f"Ground width      : {ground_width_m * 1e6:.3f} um")
    print(f"RF rail width     : {rf_width_m * 1e6:.3f} um")
    print(
        f"RF-null position  : x={x_null_m * 1e6:.6f} um, "
        f"z={z_null_m * 1e6:.6f} um"
    )
    print()

    print(f"{'V_RF peak [V]':>14}  {'f_1 [MHz]':>12}  {'f_2 [MHz]':>12}")

    for rf_voltage_v in (50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
        frequencies_hz, _ = secular_frequencies_hz(
            trap=trap,
            x_m=x_null_m,
            z_m=z_null_m,
            rf_voltage_peak_v=rf_voltage_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        )
        values_mhz = frequencies_hz / 1e6

        print(
            f"{rf_voltage_v:14.1f}  "
            f"{values_mhz[0]:12.6f}  "
            f"{values_mhz[1]:12.6f}"
        )

    required_voltage_v, _ = required_rf_voltage_peak_v(
        trap=trap,
        x_m=x_null_m,
        z_m=z_null_m,
        target_frequency_hz=TARGET_RADIAL_FREQUENCY_HZ,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )

    print()
    print(
        "Required RF peak voltage for "
        f"{TARGET_RADIAL_FREQUENCY_HZ / 1e6:.3f} MHz: "
        f"{required_voltage_v:.3f} V"
    )
    print(
        f"Within {MAX_RF_VOLTAGE_PEAK_V:.0f} V limit: "
        f"{required_voltage_v <= MAX_RF_VOLTAGE_PEAK_V}"
    )


if __name__ == "__main__":
    main()