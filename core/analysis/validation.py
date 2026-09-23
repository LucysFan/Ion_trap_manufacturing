"""Hard-constraint validation for X-junction RF transport paths."""

from __future__ import annotations

from dataclasses import dataclass

from config.targets import (
    MAX_NOMINAL_NULL_HEIGHT_DEVIATION_M,
    RF_NULL_HEIGHT_MAX_M,
    RF_NULL_HEIGHT_MIN_M,
)
from core.analysis.path_metrics import PathMetrics


@dataclass(frozen=True)
class ValidationReport:
    """Result of hard physical checks for a path."""

    valid: bool
    messages: tuple[str, ...]


def validate_rf_transport_path(
    metrics: PathMetrics,
    *,
    max_height_deviation_m: float = MAX_NOMINAL_NULL_HEIGHT_DEVIATION_M,
    min_height_m: float = RF_NULL_HEIGHT_MIN_M,
    max_height_m: float = RF_NULL_HEIGHT_MAX_M,
) -> ValidationReport:
    """Validate basic RF-path conditions before optimisation."""
    issues: list[str] = []

    if not metrics.valid:
        issues.append("RF transverse-minimum trace is incomplete.")

    if not min_height_m <= metrics.arm_reference_height_m <= max_height_m:
        issues.append("Reference RF-null height is outside allowed range.")

    if metrics.height_peak_deviation_m > max_height_deviation_m:
        issues.append(
            "Peak RF-null height deviation exceeds nominal limit: "
            f"{metrics.height_peak_deviation_m * 1e6:.3f} um > "
            f"{max_height_deviation_m * 1e6:.3f} um."
        )

    if metrics.maximum_lateral_offset_m > 5e-6:
        issues.append(
            "Trace leaves nominal transport axis by more than 5 um."
        )

    return ValidationReport(
        valid=not issues,
        messages=tuple(issues) if issues else ("ok",),
    )