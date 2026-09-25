from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.curvature import secular_frequencies_at_null_hz
from core.analysis.path_metrics import compute_path_metrics
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.validation import validate_rf_transport_path
from core.geometry.rf_mask import rf_mask
from core.optimization.objectives import build_evaluation


@dataclass(frozen=True)
class CandidateProfiles:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    transverse_residual_v_m: np.ndarray
    full_field_norm_v_m: np.ndarray
    converged: np.ndarray
    frequency_low_hz: np.ndarray
    frequency_high_hz: np.ndarray
    pseudopotential_ev: np.ndarray


class SolvedFieldModel:
    def __init__(self, solver, sigma: np.ndarray) -> None:
        self._solver = solver
        self._sigma = np.asarray(sigma, dtype=float)

    def electric_field(self, x_m: float, y_m: float, z_m: float) -> np.ndarray:
        return np.asarray(
            self._solver.field(float(x_m), float(y_m), float(z_m), self._sigma),
            dtype=float,
        )


def _solve_candidate_sigma(candidate, solver, cfg) -> np.ndarray:
    centre_x = np.asarray(solver.centres_m[:, 0], dtype=float)
    centre_y = np.asarray(solver.centres_m[:, 1], dtype=float)
    mask = rf_mask(candidate, centre_x, centre_y, cfg)
    return np.asarray(solver.solve_masks(mask[None, :])[0], dtype=float)


def _route_x_positions(cfg) -> np.ndarray:
    return np.linspace(0.0, cfg.route_extent_m, cfg.path_points, dtype=float)


def calculate_candidate_profiles(
    *,
    candidate,
    solver,
    cfg,
    sigma: np.ndarray | None = None,
) -> CandidateProfiles:
    sigma_value = _solve_candidate_sigma(candidate, solver, cfg) if sigma is None else sigma
    field_model = SolvedFieldModel(solver, sigma_value)

    x_m = _route_x_positions(cfg)
    trace = trace_rf_transverse_minimum(
        field_model,
        x_m,
        initial_y_m=0.0,
        initial_z_m=TARGET_ION_HEIGHT_M,
        residual_tolerance_v_m=cfg.rf_null_residual_tolerance_v_m,
        min_height_m=cfg.z_min_m,
        max_height_m=cfg.z_max_m,
        max_transverse_shift_m=cfg.max_transverse_shift_m,
    )

    frequencies_low = np.full(len(x_m), np.nan, dtype=float)
    frequencies_high = np.full(len(x_m), np.nan, dtype=float)

    for index, (position_m, is_converged) in enumerate(
        zip(trace.positions_m, trace.converged)
    ):
        if not is_converged:
            continue

        frequencies_hz, _, _ = secular_frequencies_at_null_hz(
            field_model,
            position_m,
            rf_voltage_peak_v=cfg.rf_peak_voltage_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        )
        frequencies_sorted_hz = np.sort(np.asarray(frequencies_hz, dtype=float))
        frequencies_low[index] = float(frequencies_sorted_hz[0])
        frequencies_high[index] = float(frequencies_sorted_hz[-1])

    pseudopotential_ev = pseudopotential_profile_ev(
        field_model,
        trace,
        rf_voltage_peak_v=cfg.rf_peak_voltage_v,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )

    return CandidateProfiles(
        x_m=np.asarray(trace.x_m, dtype=float),
        y_m=np.asarray(trace.y_m, dtype=float),
        z_m=np.asarray(trace.z_m, dtype=float),
        transverse_residual_v_m=np.asarray(
            trace.transverse_residual_v_m,
            dtype=float,
        ),
        full_field_norm_v_m=np.asarray(trace.full_field_norm_v_m, dtype=float),
        converged=np.asarray(trace.converged, dtype=bool),
        frequency_low_hz=frequencies_low,
        frequency_high_hz=frequencies_high,
        pseudopotential_ev=np.asarray(pseudopotential_ev, dtype=float),
    )


def evaluate_candidate(
    candidate,
    solver,
    cfg,
):
    sigma = _solve_candidate_sigma(candidate, solver, cfg)
    field_model = SolvedFieldModel(solver, sigma)

    x_m = _route_x_positions(cfg)
    trace = trace_rf_transverse_minimum(
        field_model,
        x_m,
        initial_y_m=0.0,
        initial_z_m=TARGET_ION_HEIGHT_M,
        residual_tolerance_v_m=cfg.rf_null_residual_tolerance_v_m,
        min_height_m=cfg.z_min_m,
        max_height_m=cfg.z_max_m,
        max_transverse_shift_m=cfg.max_transverse_shift_m,
    )

    path_metrics = compute_path_metrics(
        trace,
        arm_reference_height_m=TARGET_ION_HEIGHT_M,
    )
    validation_report = validate_rf_transport_path(path_metrics)

    pseudopotential_ev = pseudopotential_profile_ev(
        field_model,
        trace,
        rf_voltage_peak_v=cfg.rf_peak_voltage_v,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )
    barrier_metrics = compute_barrier_metrics(pseudopotential_ev, trace)

    frequencies_low = np.full(len(x_m), np.nan, dtype=float)
    frequencies_high = np.full(len(x_m), np.nan, dtype=float)

    for index, (position_m, is_converged) in enumerate(
        zip(trace.positions_m, trace.converged)
    ):
        if not is_converged:
            continue

        frequencies_hz, _, _ = secular_frequencies_at_null_hz(
            field_model,
            position_m,
            rf_voltage_peak_v=cfg.rf_peak_voltage_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        )
        frequencies_sorted_hz = np.sort(np.asarray(frequencies_hz, dtype=float))
        frequencies_low[index] = float(frequencies_sorted_hz[0])
        frequencies_high[index] = float(frequencies_sorted_hz[-1])

    return build_evaluation(
        candidate=candidate,
        cfg=cfg,
        path_metrics=path_metrics,
        barrier_metrics=barrier_metrics,
        validation_report=validation_report,
        frequencies_low_hz=frequencies_low[trace.converged],
        frequencies_high_hz=frequencies_high[trace.converged],
    )


def evaluate_population(
    population: list,
    solver,
    cfg,
    progress=None,
    stage: str | None = None,
):
    evaluations = []

    if progress is not None and stage:
        progress.set_description_str(stage)
        progress.refresh()

    for candidate in population:
        evaluations.append(evaluate_candidate(candidate, solver, cfg))
        if progress is not None:
            progress.update(1)

    return evaluations