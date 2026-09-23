from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import (
    FEATURE_OP_ADD_RF,
    FEATURE_OP_REMOVE_RF,
    FEATURE_RECTANGLE,
    FEATURE_NONE,
    Genome,
    N_CTRL,
    N_FEATURES,
    TOPOLOGY_NAMES,
)


@dataclass(frozen=True)
class ConstraintReport:
    valid: bool
    violations: tuple[str, ...]


def validate_genome(
    genome: Genome,
    *,
    min_rf_width_m: float = 18e-6,
    fix_terminal_controls: bool = True,
    base_inner_m: float = 37.35e-6,
    base_outer_m: float = 216.45e-6,
) -> ConstraintReport:
    violations: list[str] = []

    if genome.inner_control_m.shape != (N_CTRL,):
        violations.append(f"inner_control_m must have shape {(N_CTRL,)}")

    if genome.outer_control_m.shape != (N_CTRL,):
        violations.append(f"outer_control_m must have shape {(N_CTRL,)}")

    if genome.feature_kind.shape != (N_FEATURES,):
        violations.append(f"feature_kind must have shape {(N_FEATURES,)}")

    if genome.feature_operation.shape != (N_FEATURES,):
        violations.append(f"feature_operation must have shape {(N_FEATURES,)}")

    if genome.feature_radius_m.shape != (N_FEATURES,):
        violations.append(f"feature_radius_m must have shape {(N_FEATURES,)}")

    if genome.feature_theta_rad.shape != (N_FEATURES,):
        violations.append(f"feature_theta_rad must have shape {(N_FEATURES,)}")

    if genome.feature_p1_m.shape != (N_FEATURES,):
        violations.append(f"feature_p1_m must have shape {(N_FEATURES,)}")

    if genome.feature_p2_m.shape != (N_FEATURES,):
        violations.append(f"feature_p2_m must have shape {(N_FEATURES,)}")

    if genome.feature_angle_rad.shape != (N_FEATURES,):
        violations.append(f"feature_angle_rad must have shape {(N_FEATURES,)}")

    arrays = (
        ("inner_control_m", genome.inner_control_m),
        ("outer_control_m", genome.outer_control_m),
        ("feature_radius_m", genome.feature_radius_m),
        ("feature_theta_rad", genome.feature_theta_rad),
        ("feature_p1_m", genome.feature_p1_m),
        ("feature_p2_m", genome.feature_p2_m),
        ("feature_angle_rad", genome.feature_angle_rad),
    )
    for name, values in arrays:
        if not np.all(np.isfinite(values)):
            violations.append(f"{name} contains non-finite values")

    if not np.isfinite(genome.topology_size_m):
        violations.append("topology_size_m is not finite")

    if not np.isfinite(genome.topology_width_m):
        violations.append("topology_width_m is not finite")

    if np.any(genome.outer_control_m - genome.inner_control_m < min_rf_width_m):
        violations.append("outer_control_m - inner_control_m violates minimum RF width")

    if genome.topology < 0 or genome.topology >= len(TOPOLOGY_NAMES):
        violations.append("topology index is out of range")

    if np.any((genome.feature_kind < FEATURE_NONE) | (genome.feature_kind > FEATURE_RECTANGLE)):
        violations.append("feature_kind contains invalid values")

    if np.any(
        (genome.feature_operation != FEATURE_OP_ADD_RF)
        & (genome.feature_operation != FEATURE_OP_REMOVE_RF)
    ):
        violations.append("feature_operation contains invalid values")

    if fix_terminal_controls:
        if not np.isclose(genome.inner_control_m[-1], base_inner_m):
            violations.append("last inner control point must be fixed to baseline")
        if not np.isclose(genome.outer_control_m[-1], base_outer_m):
            violations.append("last outer control point must be fixed to baseline")

    if genome.topology_size_m < Genome.TOPOLOGY_SIZE_MIN_M or genome.topology_size_m > Genome.TOPOLOGY_SIZE_MAX_M:
        violations.append("topology_size_m is outside allowed range")

    if genome.topology_width_m < Genome.TOPOLOGY_WIDTH_MIN_M or genome.topology_width_m > Genome.TOPOLOGY_WIDTH_MAX_M:
        violations.append("topology_width_m is outside allowed range")

    if np.any(genome.inner_control_m < Genome.INNER_MIN_M) or np.any(genome.inner_control_m > Genome.INNER_MAX_M):
        violations.append("inner_control_m is outside allowed range")

    if np.any(genome.outer_control_m < Genome.OUTER_MIN_M) or np.any(genome.outer_control_m > Genome.OUTER_MAX_M):
        violations.append("outer_control_m is outside allowed range")

    if np.any(genome.feature_radius_m < Genome.FEATURE_RADIUS_MIN_M) or np.any(
        genome.feature_radius_m > Genome.FEATURE_RADIUS_MAX_M
    ):
        violations.append("feature_radius_m is outside allowed range")

    if np.any(genome.feature_p1_m < Genome.FEATURE_P_MIN_M) or np.any(
        genome.feature_p1_m > Genome.FEATURE_P_MAX_M
    ):
        violations.append("feature_p1_m is outside allowed range")

    if np.any(genome.feature_p2_m < Genome.FEATURE_P_MIN_M) or np.any(
        genome.feature_p2_m > Genome.FEATURE_P_MAX_M
    ):
        violations.append("feature_p2_m is outside allowed range")

    return ConstraintReport(
        valid=len(violations) == 0,
        violations=tuple(violations),
    )


def repair_genome(
    genome: Genome,
    *,
    min_rf_width_m: float = 18e-6,
    fix_terminal_controls: bool = True,
    base_inner_m: float = 37.35e-6,
    base_outer_m: float = 216.45e-6,
) -> Genome:
    repaired = genome.clip()

    repaired.outer_control_m = np.maximum(
        repaired.outer_control_m,
        repaired.inner_control_m + min_rf_width_m,
    )

    if fix_terminal_controls:
        repaired.inner_control_m[-1] = base_inner_m
        repaired.outer_control_m[-1] = max(base_outer_m, base_inner_m + min_rf_width_m)

    repaired.feature_kind = np.clip(
        repaired.feature_kind,
        FEATURE_NONE,
        FEATURE_RECTANGLE,
    ).astype(np.int8)

    repaired.feature_operation = np.where(
        repaired.feature_operation >= 0,
        FEATURE_OP_ADD_RF,
        FEATURE_OP_REMOVE_RF,
    ).astype(np.int8)

    repaired.feature_theta_rad = np.mod(repaired.feature_theta_rad, np.pi / 4.0)
    repaired.feature_angle_rad = (repaired.feature_angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    return repaired


def check_hard_constraints(
    metrics: dict[str, float],
    *,
    max_height_peak_m: float = 25e-6,
    max_lateral_peak_m: float = 25e-6,
    max_field_residual_v_m: float = 5e3,
    min_frequency_hz: float = 0.15e6,
) -> tuple[bool, str]:
    required_keys = (
        "height_peak_m",
        "lateral_peak_m",
        "field_residual_max_v_m",
        "frequency_min_hz",
    )
    for key in required_keys:
        if key not in metrics:
            return False, f"missing metric: {key}"
        if not np.isfinite(metrics[key]):
            return False, f"non-finite metric: {key}"

    if metrics["height_peak_m"] > max_height_peak_m:
        return False, "height peak exceeds hard limit"

    if metrics["lateral_peak_m"] > max_lateral_peak_m:
        return False, "lateral peak exceeds hard limit"

    if metrics["field_residual_max_v_m"] > max_field_residual_v_m:
        return False, "field residual exceeds hard limit"

    if metrics["frequency_min_hz"] < min_frequency_hz:
        return False, "minimum frequency is below hard limit"

    return True, ""