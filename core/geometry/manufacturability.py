"""Basic manufacturability checks for idealized X-junction geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.junction_templates import XJunctionParameters


@dataclass(frozen=True)
class ManufacturabilityReport:
    """Result of basic manufacturability checks."""

    valid: bool
    messages: tuple[str, ...]


def check_x_junction_manufacturability(
    parameters: XJunctionParameters,
    *,
    min_feature_size_m: float = 5e-6,
    min_island_radius_m: float = 5e-6,
    n_taper_samples: int = 101,
) -> ManufacturabilityReport:
    """Check widths and simple geometric constraints along taper profiles."""
    parameters.validate()

    if min_feature_size_m <= 0.0:
        raise ValueError("min_feature_size_m must be positive.")
    if min_island_radius_m <= 0.0:
        raise ValueError("min_island_radius_m must be positive.")
    if n_taper_samples < 2:
        raise ValueError("n_taper_samples must be at least two.")

    issues: list[str] = []

    if parameters.ground_rail_width_m < min_feature_size_m:
        issues.append("Ground transport rail is narrower than min feature size.")

    if parameters.rf_rail_width_m < min_feature_size_m:
        issues.append("Straight RF rail is narrower than min feature size.")

    if (
        parameters.central_island_radius_m > 0.0
        and parameters.central_island_radius_m < min_island_radius_m
    ):
        issues.append("Central island is below minimum manufacturable size.")

    if parameters.arm_length_m <= parameters.rf_outer_radius_m:
        issues.append("Arm length is too short to contain a straight RF section.")

    if parameters.outer_extent_m <= parameters.rf_outer_radius_m:
        issues.append("Outer BEM extent does not include the RF rail.")

    contour_extent_m = (
        max(parameters.outer_contour_knots_m)
        if parameters.outer_contour_knots_m
        else 0.0
    )

    profile_extent_m = max(
        parameters.taper_length_m,
        parameters.outer_bulge_center_m
        + 4.0 * parameters.outer_bulge_sigma_m,
        contour_extent_m,
    )
    profile_extent_m = min(profile_extent_m, parameters.arm_length_m)

    if profile_extent_m > 0.0:
        longitudinal_m = np.linspace(
            0.0,
            profile_extent_m,
            n_taper_samples,
        )
        inner_m, outer_m = parameters.rail_boundaries_m(longitudinal_m)
        widths_m = outer_m - inner_m

        if np.min(inner_m) < 0.5 * min_feature_size_m:
            issues.append(
                "RF inner edge is too close to the transport axis."
            )

        if np.min(widths_m) < min_feature_size_m:
            issues.append(
                "RF rail is narrower than the minimum feature size."
            )

        if np.any(outer_m <= inner_m):
            issues.append("RF boundaries cross along the rail profile.")

    return ManufacturabilityReport(
        valid=not issues,
        messages=tuple(issues) if issues else ("ok",),
    )