"""Diagnostics for retained v22 BEM evaluation; no independent 3-D sign-off.

All coordinates are metres, fields V/m and potentials eV. The v22 charge
solution is retained: this module NEVER treats a finer sampling path as a
newly refined BEM mesh. Reports distinguish historical barrier from excursion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.signal import find_peaks

Q = 1.602176634e-19
M_CA40 = 39.96259098 * 1.66053906660e-27


@dataclass(frozen=True)
class ValidationTargets:
    height_m: float = 90e-6
    height_tolerance_m: float = 2e-6
    barrier_limit_ev: float = 10e-3
    search_margin_ev: float = 7e-3
    excursion_limit_ev: float = 10e-3
    residual_limit_v_m: float = 1e-3


def path_points(level: str) -> np.ndarray:
    """Nested longitudinal sampling, not a BEM panel-size specification."""
    if level not in ('L1', 'L2', 'L3'):
        raise ValueError('level must be L1, L2 or L3')
    outer = {'L1': 7e-6, 'L2': 3e-6, 'L3': 1e-6}[level]
    centre = {'L1': 2e-6, 'L2': 1e-6, 'L3': 0.5e-6}[level]
    return np.unique(np.r_[np.arange(-350e-6, 350.1e-6, outer),
                              np.arange(-50e-6, 50.1e-6, centre),
                              -320e-6, -260e-6, -160e-6, -45e-6, 0.,
                              45e-6, 160e-6, 260e-6, 320e-6])


def fields_at(field, xyz: np.ndarray, chunk: int = 256) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    blocks = []
    for group in np.array_split(xyz, max(1, int(np.ceil(len(xyz) / chunk)))):
        raw = field.bem.field_batch(group[:, 0][None, :],
                                    group[:, 1][None, :],
                                    group[:, 2][None, :],
                                    field.charge, candidate_chunk=1)
        blocks.append(np.asarray(field.bem.asnumpy(raw)[0], dtype=float))
    return np.concatenate(blocks, axis=0)


def potential_ev(field, xyz: np.ndarray, rf_peak_v: float, omega: float) -> np.ndarray:
    # v22 field is computed for the unit-voltage BEM electrode mask.
    e = fields_at(field, xyz)
    return Q * rf_peak_v**2 * np.einsum('ij,ij->i', e, e) / (4 * M_CA40 * omega**2)


def analyse_retained(result: dict, data: tuple, cfg, omega: float,
                     targets: ValidationTargets = ValidationTargets(),
                     level: str = 'L2', max_seconds: float = 120.) -> dict:
    """Inspect the actual retained fine field; raise on incomplete checks."""
    if not result.get('valid') or data is None:
        return {'complete': False, 'reason': result.get('reason', 'no retained data')}
    t0 = perf_counter()
    model, trace, u_ev, field, geometry, zone = data
    x0 = np.asarray(trace.x_m, dtype=float)
    y0 = np.asarray(trace.y_m, dtype=float)
    z0 = np.asarray(trace.z_m, dtype=float)
    if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(y0)) and
            np.all(np.isfinite(z0)) and np.all(trace.converged)):
        return {'complete': False, 'reason': 'invalid retained RF-null trace'}
    if level not in ('L1', 'L2', 'L3'):
        raise ValueError(level)
    xs = path_points(level)
    # Interpolation only proposes seeds; each position is corrected by the
    # project's exact RF-null tracer, not scored from interpolated potentials.
    from core.analysis.rf_null_trace import trace_rf_transverse_minimum
    checked = trace_rf_transverse_minimum(
        field, xs, initial_y_m=float(np.interp(xs[0], x0, y0)),
        initial_z_m=float(np.interp(xs[0], x0, z0)),
        residual_tolerance_v_m=targets.residual_limit_v_m,
        max_transverse_shift_m=25e-6)
    if not checked.valid or not np.all(checked.converged):
        return {'complete': False, 'reason': 'dense RF-null tracing failed',
                'panels': int(model.n_panels), 'level': level}
    xyz = np.column_stack((checked.x_m, checked.y_m, checked.z_m))
    u = potential_ev(field, xyz, cfg.rf_peak_v, omega)
    if not np.all(np.isfinite(u)):
        return {'complete': False, 'reason': 'non-finite pseudopotential'}
    if perf_counter() - t0 > max_seconds:
        return {'complete': False, 'reason': 'time budget exceeded after trace'}
    ref = float(zone['u_reference_ev'])
    reference_mask = (np.abs(xs) >= 260e-6) & (np.abs(xs) <= 320e-6)
    ref_height = float(np.median(np.asarray(checked.z_m)[reference_mask]))
    excursions = np.abs(u - ref)
    # Positive barrier relative to the historical v22 reference, not max-min.
    barrier = float(max(0., np.max(u - ref)))
    peaks, _ = find_peaks(u, prominence=1e-5)
    wells, _ = find_peaks(-u, prominence=1e-5)
    rf_residual = np.asarray(checked.full_field_norm_v_m, dtype=float)
    errors = np.abs(np.asarray(checked.z_m, dtype=float) - targets.height_m)
    report = {
        'complete': True, 'level': level, 'panels': int(model.n_panels),
        'path_points': int(len(xs)), 'time_s': float(perf_counter() - t0),
        'target_height_um': targets.height_m * 1e6,
        'reference_height_um': ref_height * 1e6,
        'height_min_um': float(np.min(checked.z_m) * 1e6),
        'height_max_um': float(np.max(checked.z_m) * 1e6),
        'height_max_abs_error_um': float(np.max(errors) * 1e6),
        'height_relative_to_arm_max_um': float(np.max(np.abs(checked.z_m - ref_height)) * 1e6),
        'barrier_mev': barrier * 1e3,
        'historical_v22_barrier_mev': float(result['barrier_ev']) * 1e3,
        'excursion_mev': float(np.max(excursions) * 1e3),
        'rf_full_field_max_v_m': float(np.max(rf_residual)),
        'axial_peaks_um': (xs[peaks] * 1e6).tolist(),
        'axial_wells_um': (xs[wells] * 1e6).tolist(),
        'c4v_other_axis_inferred': False,
        'lateral_extrema_3d_checked': False,
        'surface_field_checked': False,
        'secular_frequency_checked': False,
        'mesh_convergence_checked': False,
        'targets': asdict(targets),
    }
    report['passes_axial_targets'] = bool(
        np.max(errors) <= targets.height_tolerance_m and
        barrier < targets.barrier_limit_ev and
        np.max(excursions) < targets.excursion_limit_ev and
        np.all(np.isfinite(rf_residual)))
    report['search_margin_pass'] = bool(barrier < targets.search_margin_ev)
    return report
