"""Tests for RF pseudopotential-barrier metrics."""

from __future__ import annotations

import numpy as np

from core.analysis.barrier import compute_barrier_metrics
from core.analysis.rf_null_trace import RFNullTrace


def _complete_trace(n_points: int) -> RFNullTrace:
    return RFNullTrace(
        x_m=np.linspace(-100e-6, 100e-6, n_points),
        y_m=np.zeros(n_points),
        z_m=np.full(n_points, 90e-6),
        transverse_residual_v_m=np.zeros(n_points),
        full_field_norm_v_m=np.zeros(n_points),
        converged=np.ones(n_points, dtype=bool),
        messages=tuple("ok" for _ in range(n_points)),
    )


def test_constant_energy_profile_has_zero_barrier() -> None:
    trace = _complete_trace(21)
    energies_ev = np.full(21, 1.5e-9)

    metrics = compute_barrier_metrics(energies_ev, trace)

    assert metrics.valid
    assert np.isclose(metrics.barrier_height_ev, 0.0, atol=1e-20)


def test_central_energy_peak_is_detected() -> None:
    trace = _complete_trace(21)
    energies_ev = np.zeros(21)
    energies_ev[10] = 2e-6

    metrics = compute_barrier_metrics(energies_ev, trace)

    assert metrics.valid
    assert np.isclose(metrics.barrier_height_ev, 2e-6, atol=1e-15)