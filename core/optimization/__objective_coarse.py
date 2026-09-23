from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from core.analysis.barrier import compute_barrier_metrics
from core.analysis.path_metrics import compute_path_metrics
from core.analysis.rf_null_trace import trace_rf_null_path
from core.analysis.validation import validate_rf_transport_path
from core.geometry.bem_builder import build_x_junction_bem
from core.optimization.constraints import GeometryConstraintResult, evaluate_geometry_constraints
from core.optimization.genome import JunctionGenome


@dataclass(frozen=True)
class CoarseObjectiveConfig:
    """Numerical and physical settings for the fast GA evaluation backend."""

    x_start_um: float = -300.0
    x_stop_um: float = 300.0
    n_path_samples: int = 81

    panel_pitch_um: float = 8.0
    domain_half_width_um: float = 400.0
    ground_width_um: float = 80.0
    rf_rail_width_um: float = 80.0
    dc_segment_length_um: float = 80.0

    ion_mass_kg: float = 40.0 * 1.66053906660e-27
    ion_charge_c: float = 1.602176634e-19
    rf_amplitude_v: float = 100.0
    rf_angular_frequency_rad_s: float = 2.0 * np.pi * 40.0e6

    initial_height_um: float = 90.0
    minimum_height_um: float = 20.0
    maximum_height_um: float = 250.0
    transverse_bound_um: float = 250.0

    trace_tolerance_v_m: float = 1.0e-3
    trace_max_iterations: int = 40
    validation_residual_limit_v_m: float = 50.0

    target_height_variation_um: float = 10.0
    target_rf_barrier_mev: float = 10.0

    failed_height_variation_um: float = 1.0e4
    failed_rf_barrier_mev: float = 1.0e6
    failed_penalty: float = 1.0e6


@dataclass(frozen=True)
class CoarseObjectiveResult:
    """Complete result of one coarse field evaluation.

    objectives are minimized and ordered as:
      [peak RF-null height excursion in um,
       RF pseudopotential barrier in meV,
       constraint/trace/residual penalty].
    """

    objectives: np.ndarray
    feasible: bool
    metrics: dict[str, float]
    diagnostics: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "objective_height_variation_um": float(self.objectives[0]),
            "objective_rf_barrier_mev": float(self.objectives[1]),
            "objective_penalty": float(self.objectives[2]),
            "feasible": bool(self.feasible),
        }
        record.update(self.metrics)
        return record


def _genome_to_geometry_kwargs(genome: GeometryGenome) -> dict[str, float]:
    return {
        "start_radius_m": genome.start_radius_um * 1.0e-6,
        "inner_shift_m": genome.inner_shift_um * 1.0e-6,
        "outer_shift_m": genome.outer_shift_um * 1.0e-6,
        "taper_length_m": genome.taper_length_um * 1.0e-6,
        "taper_power": genome.taper_power,
    }


def _constraint_result_dict(result: GeometryConstraintResult) -> dict[str, Any]:
    return {
        "constraint_feasible": bool(result.feasible),
        "constraint_penalty": float(result.penalty),
        "constraint_messages": tuple(result.messages),
    }


def _failure_result(
    genome: GeometryGenome,
    constraint: GeometryConstraintResult,
    config: CoarseObjectiveConfig,
    reason: str,
    exception: Exception | None = None,
) -> CoarseObjectiveResult:
    penalty = float(config.failed_penalty + constraint.penalty)
    metrics = {
        "peak_height_variation_um": float(config.failed_height_variation_um),
        "rf_barrier_mev": float(config.failed_rf_barrier_mev),
        "max_lateral_displacement_um": float(config.failed_height_variation_um),
        "arm_reference_height_um": float("nan"),
        "max_transverse_residual_v_m": float("inf"),
        "max_field_norm_v_m": float("inf"),
        "trace_converged_fraction": 0.0,
        "n_panels": 0.0,
        "constraint_penalty": float(constraint.penalty),
        "trace_penalty": float(config.failed_penalty),
        "residual_penalty": 0.0,
        "soft_target_penalty": 0.0,
        "total_penalty": penalty,
    }
    diagnostics: dict[str, Any] = {
        "genome": asdict(genome),
        "config": asdict(config),
        "failure_reason": reason,
        **_constraint_result_dict(constraint),
    }
    if exception is not None:
        diagnostics["exception_type"] = type(exception).__name__
        diagnostics["exception_message"] = str(exception)

    return CoarseObjectiveResult(
        objectives=np.array(
            [
                config.failed_height_variation_um,
                config.failed_rf_barrier_mev,
                penalty,
            ],
            dtype=float,
        ),
        feasible=False,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def evaluate_coarse_objective(
    genome: JunctionGenome,
    *,
    config: CoarseObjectiveConfig | None = None,
) -> CoarseObjectiveResult:
    """Evaluate one X-junction geometry using the fast coarse BEM backend.

    Geometry constraints are evaluated first.  Valid candidates are built with
    the coarse edge-aligned BEM, the transverse RF null is traced across the
    junction, then path and pseudopotential-barrier metrics are calculated.
    """
    if config is None:
        config = CoarseObjectiveConfig()

    constraint = evaluate_geometry_constraints(genome)
    if not constraint.feasible:
        return _failure_result(
            genome,
            constraint,
            config,
            reason="geometry_constraints_failed",
        )

    x_values_m = np.linspace(
        config.x_start_um * 1.0e-6,
        config.x_stop_um * 1.0e-6,
        config.n_path_samples,
    )

    try:
        model = build_x_junction_bem(
            **_genome_to_geometry_kwargs(genome),
            panel_pitch_m=config.panel_pitch_um * 1.0e-6,
            domain_half_width_m=config.domain_half_width_um * 1.0e-6,
            ground_width_m=config.ground_width_um * 1.0e-6,
            rf_rail_width_m=config.rf_rail_width_um * 1.0e-6,
            dc_segment_length_m=config.dc_segment_length_um * 1.0e-6,
            edge_aligned=False,
        )

        trace = trace_rf_null_path(
            model,
            x_values_m=x_values_m,
            initial_y_m=0.0,
            initial_z_m=config.initial_height_um * 1.0e-6,
            minimum_z_m=config.minimum_height_um * 1.0e-6,
            maximum_z_m=config.maximum_height_um * 1.0e-6,
            transverse_bound_m=config.transverse_bound_um * 1.0e-6,
            tolerance_v_m=config.trace_tolerance_v_m,
            max_iterations=config.trace_max_iterations,
        )

        metrics = compute_path_metrics(trace)
        energies_ev = model.pseudopotential_ev(
            np.column_stack((trace.x_m, trace.y_m, trace.z_m)),
            ion_mass_kg=config.ion_mass_kg,
            ion_charge_c=config.ion_charge_c,
            rf_amplitude_v=config.rf_amplitude_v,
            rf_angular_frequency_rad_s=config.rf_angular_frequency_rad_s,
        )
        barrier = compute_barrier_metrics(energies_ev, trace)
        validation = validate_rf_transport_path(
            metrics,
            maximum_transverse_residual_v_m=config.validation_residual_limit_v_m,
        )

    except Exception as exception:
        return _failure_result(
            genome,
            constraint,
            config,
            reason="coarse_bem_or_trace_exception",
            exception=exception,
        )

    converged_fraction = float(np.mean(trace.converged))
    max_transverse_residual = float(np.max(trace.transverse_residual_v_m))
    max_field_norm = float(np.max(trace.full_field_norm_v_m))
    peak_height_variation_um = float(metrics.peak_height_variation_m * 1.0e6)
    max_lateral_displacement_um = float(metrics.max_lateral_displacement_m * 1.0e6)
    arm_reference_height_um = float(metrics.arm_reference_height_m * 1.0e6)
    rf_barrier_mev = float(barrier.barrier_ev * 1.0e3)

    trace_penalty = 0.0
    if converged_fraction < 1.0:
        trace_penalty += 1.0e3 * (1.0 - converged_fraction)

    if not validation.valid:
        trace_penalty += 100.0

    residual_penalty = 0.0
    if max_transverse_residual > config.validation_residual_limit_v_m:
        residual_penalty = 10.0 * np.log1p(
            max_transverse_residual / config.validation_residual_limit_v_m
        )

    soft_target_penalty = (
        0.25
        * max(0.0, peak_height_variation_um - config.target_height_variation_um)
        + 0.05 * max(0.0, rf_barrier_mev - config.target_rf_barrier_mev)
    )

    total_penalty = float(
        constraint.penalty + trace_penalty + residual_penalty + soft_target_penalty
    )
    feasible = bool(constraint.feasible and validation.valid and np.all(trace.converged))

    result_metrics = {
        "peak_height_variation_um": peak_height_variation_um,
        "rf_barrier_mev": rf_barrier_mev,
        "max_lateral_displacement_um": max_lateral_displacement_um,
        "arm_reference_height_um": arm_reference_height_um,
        "max_transverse_residual_v_m": max_transverse_residual,
        "max_field_norm_v_m": max_field_norm,
        "trace_converged_fraction": converged_fraction,
        "n_panels": float(model.n_panels),
        "constraint_penalty": float(constraint.penalty),
        "trace_penalty": float(trace_penalty),
        "residual_penalty": float(residual_penalty),
        "soft_target_penalty": float(soft_target_penalty),
        "total_penalty": total_penalty,
    }

    diagnostics = {
        "genome": asdict(genome),
        "config": asdict(config),
        **_constraint_result_dict(constraint),
        "validation_valid": bool(validation.valid),
        "validation_messages": tuple(validation.messages),
        "trace_converged": trace.converged.copy(),
        "trace_x_m": trace.x_m.copy(),
        "trace_y_m": trace.y_m.copy(),
        "trace_z_m": trace.z_m.copy(),
        "trace_transverse_residual_v_m": trace.transverse_residual_v_m.copy(),
        "trace_full_field_norm_v_m": trace.full_field_norm_v_m.copy(),
        "pseudopotential_energies_ev": np.asarray(energies_ev, dtype=float).copy(),
        "barrier_reference_energy_ev": float(barrier.reference_energy_ev),
        "barrier_peak_energy_ev": float(barrier.peak_energy_ev),
    }

    return CoarseObj