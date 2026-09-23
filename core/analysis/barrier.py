"""RF pseudopotential barrier proxies along traced transport paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.analysis.rf_null_trace import RFNullTrace
from core.electrostatics.pseudopotential import pseudopotential_energy_ev


@dataclass(frozen=True)
class BarrierMetrics:
    """Pseudopotential-energy metrics along a traced path."""

    valid: bool
    reference_energy_ev: float
    maximum_energy_ev: float
    barrier_height_ev: float
    rms_energy_above_reference_ev: float
    energies_ev: np.ndarray


def pseudopotential_profile_ev(
    field_model: object,
    trace: RFNullTrace,
    *,
    rf_voltage_peak_v: float,
    rf_angular_frequency_rad_s: float,
) -> np.ndarray:
    """Evaluate RF pseudopotential energy at every converged trace point."""
    energies_ev = np.full(len(trace.x_m), np.nan, dtype=float)

    for index, (x_m, y_m, z_m, is_converged) in enumerate(
        zip(trace.x_m, trace.y_m, trace.z_m, trace.converged)
    ):
        if not is_converged:
            continue

        field = np.asarray(
            field_model.electric_field(float(x_m), float(y_m), float(z_m)),
            dtype=float,
        )

        energies_ev[index] = pseudopotential_energy_ev(
            field,
            rf_voltage_peak_v=rf_voltage_peak_v,
            rf_angular_frequency_rad_s=rf_angular_frequency_rad_s,
        )

    return energies_ev


def compute_barrier_metrics(
    energies_ev: np.ndarray,
    trace: RFNullTrace,
    *,
    edge_fraction: float = 0.1,
) -> BarrierMetrics:
    """Calculate barrier relative to median energy at both arm ends."""
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be in (0, 0.5].")

    valid_energies_ev = np.asarray(energies_ev, dtype=float)[trace.converged]

    if len(valid_energies_ev) < 2 or not np.all(np.isfinite(valid_energies_ev)):
        return BarrierMetrics(
            valid=False,
            reference_energy_ev=np.nan,
            maximum_energy_ev=np.nan,
            barrier_height_ev=np.inf,
            rms_energy_above_reference_ev=np.inf,
            energies_ev=np.asarray(energies_ev, dtype=float),
        )

    n_edge = max(1, int(np.ceil(edge_fraction * len(valid_energies_ev))))
    reference_samples_ev = np.concatenate(
        [valid_energies_ev[:n_edge], valid_energies_ev[-n_edge:]]
    )

    reference_energy_ev = float(np.median(reference_samples_ev))
    maximum_energy_ev = float(np.max(valid_energies_ev))

    excess_ev = valid_energies_ev - reference_energy_ev
    barrier_height_ev = float(max(0.0, np.max(excess_ev)))
    rms_energy_above_reference_ev = float(
        np.sqrt(np.mean(np.maximum(excess_ev, 0.0) ** 2))
    )

    return BarrierMetrics(
        valid=trace.valid,
        reference_energy_ev=reference_energy_ev,
        maximum_energy_ev=maximum_energy_ev,
        barrier_height_ev=barrier_height_ev,
        rms_energy_above_reference_ev=rms_energy_above_reference_ev,
        energies_ev=np.asarray(energies_ev, dtype=float),
    )