"""Polygonal, fourfold-symmetric RF geometry for planar X-junctions.

This module is intentionally separate from junction_templates.py. The existing
baseline mask is a union of perpendicular RF strips; the present family adds a
dedicated polygonal transition region so junction boundaries can be chamfered
or wedge-shaped rather than purely rectangular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChamferedXJunctionParameters:
    """Parameters of a symmetric X-junction with polygonal RF transitions.

    Dimensions are in metres.

    The straight RF rails occupy, far from the centre:

        ground_half_width <= |transverse| <= rf_outer_radius

    for longitudinal coordinate between transition_radius and arm_length.

    Inside the transition radius, each RF rail becomes a trapezoidal/polygonal
    wedge. The pair of wedge edges is controlled independently by their
    centre and arm-side radii.
    """

    ground_rail_width_m: float
    rf_rail_width_m: float
    arm_length_m: float
    outer_extent_m: float

    transition_radius_m: float
    rf_start_radius_m: float

    inner_radius_at_centre_m: float
    outer_radius_at_centre_m: float

    inner_radius_at_transition_m: float
    outer_radius_at_transition_m: float

    def validate(self) -> None:
        """Validate basic geometry and non-crossing RF boundaries."""
        values = (
            self.ground_rail_width_m,
            self.rf_rail_width_m,
            self.arm_length_m,
            self.outer_extent_m,
            self.transition_radius_m,
            self.rf_start_radius_m,
            self.inner_radius_at_centre_m,
            self.outer_radius_at_centre_m,
            self.inner_radius_at_transition_m,
            self.outer_radius_at_transition_m,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("All chamfered-junction parameters must be finite.")

        if self.ground_rail_width_m <= 0.0:
            raise ValueError("ground_rail_width_m must be positive.")
        if self.rf_rail_width_m <= 0.0:
            raise ValueError("rf_rail_width_m must be positive.")
        if self.arm_length_m <= 0.0:
            raise ValueError("arm_length_m must be positive.")
        if self.outer_extent_m <= self.arm_length_m:
            raise ValueError("outer_extent_m must exceed arm_length_m.")

        if self.rf_start_radius_m < 0.0:
            raise ValueError("rf_start_radius_m must be non-negative.")
        if self.transition_radius_m <= self.rf_start_radius_m:
            raise ValueError(
                "transition_radius_m must exceed rf_start_radius_m."
            )
        if self.transition_radius_m >= self.arm_length_m:
            raise ValueError(
                "transition_radius_m must lie inside the arm length."
            )

        if self.inner_radius_at_centre_m <= 0.0:
            raise ValueError("inner_radius_at_centre_m must be positive.")
        if self.outer_radius_at_centre_m <= self.inner_radius_at_centre_m:
            raise ValueError(
                "Outer RF boundary must lie outside inner boundary at centre."
            )
        if (
            self.outer_radius_at_transition_m
            <= self.inner_radius_at_transition_m
        ):
            raise ValueError(
                "Outer RF boundary must lie outside inner boundary at transition."
            )

    @property
    def half_ground_width_m(self) -> float:
        return 0.5 * self.ground_rail_width_m

    @property
    def rf_outer_radius_m(self) -> float:
        return self.half_ground_width_m + self.rf_rail_width_m

    def rail_boundaries_m(
        self,
        longitudinal_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return linearly chamfered RF boundaries in a given arm.

        For s >= transition_radius_m, this exactly returns the straight-arm
        boundaries. For rf_start_radius_m <= s < transition_radius_m, the
        inner and outer boundaries linearly connect their central and
        arm-transition values.
        """
        s_m = np.asarray(longitudinal_m, dtype=float)
        absolute_s_m = np.abs(s_m)

        denominator_m = (
            self.transition_radius_m - self.rf_start_radius_m
        )
        fraction = np.clip(
            (absolute_s_m - self.rf_start_radius_m) / denominator_m,
            0.0,
            1.0,
        )

        inner_transition_m = (
            self.inner_radius_at_centre_m
            + fraction
            * (
                self.inner_radius_at_transition_m
                - self.inner_radius_at_centre_m
            )
        )
        outer_transition_m = (
            self.outer_radius_at_centre_m
            + fraction
            * (
                self.outer_radius_at_transition_m
                - self.outer_radius_at_centre_m
            )
        )

        inner_m = np.where(
            absolute_s_m >= self.transition_radius_m,
            self.half_ground_width_m,
            inner_transition_m,
        )
        outer_m = np.where(
            absolute_s_m >= self.transition_radius_m,
            self.rf_outer_radius_m,
            outer_transition_m,
        )

        return inner_m, outer_m


def make_chamfered_house_x_junction(
    *,
    ion_height_m: float,
    arm_length_m: float = 600e-6,
    outer_extent_m: float = 900e-6,
    rf_start_radius_m: float = 30e-6,
    transition_radius_m: float = 160e-6,
    inner_radius_at_centre_m: float = 20e-6,
    outer_radius_at_centre_m: float = 150e-6,
    inner_radius_at_transition_m: float | None = None,
    outer_radius_at_transition_m: float | None = None,
) -> ChamferedXJunctionParameters:
    """Build a House-ratio straight arm with a polygonal central transition."""
    ground_rail_width_m = 0.83 * ion_height_m
    rf_rail_width_m = 1.99 * ion_height_m
    half_ground_width_m = 0.5 * ground_rail_width_m
    rf_outer_radius_m = half_ground_width_m + rf_rail_width_m

    return ChamferedXJunctionParameters(
        ground_rail_width_m=ground_rail_width_m,
        rf_rail_width_m=rf_rail_width_m,
        arm_length_m=arm_length_m,
        outer_extent_m=outer_extent_m,
        transition_radius_m=transition_radius_m,
        rf_start_radius_m=rf_start_radius_m,
        inner_radius_at_centre_m=inner_radius_at_centre_m,
        outer_radius_at_centre_m=outer_radius_at_centre_m,
        inner_radius_at_transition_m=(
            half_ground_width_m
            if inner_radius_at_transition_m is None
            else inner_radius_at_transition_m
        ),
        outer_radius_at_transition_m=(
            rf_outer_radius_m
            if outer_radius_at_transition_m is None
            else outer_radius_at_transition_m
        ),
    )


def chamfered_x_junction_rf_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: ChamferedXJunctionParameters,
) -> np.ndarray:
    """Return RF mask for a fourfold-symmetric polygonal/chamfered X-junction.

    Each arm has linearly varying inner/outer boundaries inside the transition
    zone. The exact mask is still constructed from four arm regions, but those
    regions have non-rectangular trapezoidal boundaries in the junction.
    """
    parameters.validate()

    x_values_m = np.asarray(x_m, dtype=float)
    y_values_m = np.asarray(y_m, dtype=float)

    if x_values_m.shape != y_values_m.shape:
        raise ValueError("x_m and y_m must have identical shapes.")

    abs_x_m = np.abs(x_values_m)
    abs_y_m = np.abs(y_values_m)

    inner_x_m, outer_x_m = parameters.rail_boundaries_m(abs_x_m)
    horizontal_rf = (
        (abs_x_m >= parameters.rf_start_radius_m)
        & (abs_x_m <= parameters.arm_length_m)
        & (abs_y_m >= inner_x_m)
        & (abs_y_m <= outer_x_m)
    )

    inner_y_m, outer_y_m = parameters.rail_boundaries_m(abs_y_m)
    vertical_rf = (
        (abs_y_m >= parameters.rf_start_radius_m)
        & (abs_y_m <= parameters.arm_length_m)
        & (abs_x_m >= inner_y_m)
        & (abs_x_m <= outer_y_m)
    )

    return horizontal_rf | vertical_rf