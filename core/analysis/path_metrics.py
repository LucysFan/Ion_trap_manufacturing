"""Metrics for a traced RF-minimum transport path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.analysis.rf_null_trace import RFNullTrace


@dataclass(frozen=True)
class PathMetrics:
    """Geometric quality metrics for one traced transport path."""

    valid: bool
    n_points: int
    n_converged: int
    arm_reference_height_m: float
    height_peak_deviation_m: float
    height_rms_deviation_m: float
    height_peak_to_peak_m: float
    maximum_slope: float
    maximum_curvature_m_inv: float
    maximum_lateral_offset_m: float
    maximum_transverse_residual_v_m: float
    maximum_full_field_norm_v_m: float


def _estimate_arm_reference_height(
    trace: RFNullTrace,
    *,
    edge_fraction: float,
) -> float:
    """Estimate straight-arm height from both outer ends of complete trace."""
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be in (0, 0.5].")

    valid_z_m = trace.z_m[trace.converged]
    if len(valid_z_m) == 0:
        return np.nan

    n_edge = max(1, int(np.ceil(edge_fraction * len(valid_z_m))))
    samples = np.concatenate([valid_z_m[:n_edge], valid_z_m[-n_edge:]])
    return float(np.median(samples))


def compute_path_metrics(
    trace: RFNullTrace,
    *,
    arm_reference_height_m: float | None = None,
    edge_fraction: float = 0.1,
) -> PathMetrics:
    """Compute height, smoothness, residual, and lateral-offset metrics."""
    converged = trace.converged
    n_converged = int(np.sum(converged))

    if n_converged < 2:
        return PathMetrics(
            valid=False,
            n_points=len(trace.x_m),
            n_converged=n_converged,
            arm_reference_height_m=np.nan,
            height_peak_deviation_m=np.inf,
            height_rms_deviation_m=np.inf,
            height_peak_to_peak_m=np.inf,
            maximum_slope=np.inf,
            maximum_curvature_m_inv=np.inf,
            maximum_lateral_offset_m=np.inf,
            maximum_transverse_residual_v_m=np.inf,
            maximum_full_field_norm_v_m=np.inf,
        )

    x_m = trace.x_m[converged]
    y_m = trace.y_m[converged]
    z_m = trace.z_m[converged]
    residual_m = trace.transverse_residual_v_m[converged]
    full_norm_m = trace.full_field_norm_v_m[converged]

    reference_m = (
        _estimate_arm_reference_height(trace, edge_fraction=edge_fraction)
        if arm_reference_height_m is None
        else float(arm_reference_height_m)
    )

    height_deviation_m = z_m - reference_m
    slope = np.gradient(z_m, x_m, edge_order=1)
    curvature_m_inv = np.gradient(slope, x_m, edge_order=1)

    return PathMetrics(
        valid=trace.valid,
        n_points=len(trace.x_m),
        n_converged=n_converged,
        arm_reference_height_m=reference_m,
        height_peak_deviation_m=float(np.max(np.abs(height_deviation_m))),
        height_rms_deviation_m=float(
            np.sqrt(np.mean(height_deviation_m**2))
        ),
        height_peak_to_peak_m=float(np.max(z_m) - np.min(z_m)),
        maximum_slope=float(np.max(np.abs(slope))),
        maximum_curvature_m_inv=float(np.max(np.abs(curvature_m_inv))),
        maximum_lateral_offset_m=float(np.max(np.abs(y_m))),
        maximum_transverse_residual_v_m=float(np.max(residual_m)),
        maximum_full_field_norm_v_m=float(np.max(full_norm_m)),
    )