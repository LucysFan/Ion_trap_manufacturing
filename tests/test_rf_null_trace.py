"""Tests for transverse RF-minimum continuation."""

from __future__ import annotations

import numpy as np

from core.analysis.rf_null_trace import trace_rf_transverse_minimum


class StraightChannelModel:
    """Synthetic channel with transverse RF minimum at y=0, z=z0."""

    def __init__(self, z0_m: float) -> None:
        self.z0_m = z0_m

    def electric_field(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> tuple[float, float, float]:
        # E_x is nonzero because x is prescribed in the transverse solver.
        return (0.5 * x_m, y_m, z_m - self.z0_m)


def test_trace_straight_channel() -> None:
    x_values_m = np.linspace(-100e-6, 100e-6, 21)
    z0_m = 90e-6

    trace = trace_rf_transverse_minimum(
        StraightChannelModel(z0_m),
        x_values_m,
        initial_y_m=0.5e-6,
        initial_z_m=95e-6,
        residual_tolerance_v_m=1e-8,
        max_transverse_shift_m=20e-6,
    )

    assert trace.valid
    assert np.allclose(trace.y_m, 0.0, atol=1e-10)
    assert np.allclose(trace.z_m, z0_m, atol=1e-10)
    assert np.all(trace.transverse_residual_v_m < 1e-8)