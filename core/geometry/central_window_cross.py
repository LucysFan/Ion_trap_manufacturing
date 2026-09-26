"""Central-window RF-cross geometry family and BEM builders.

The family preserves a C4v-symmetric four-arm RF cross around a grounded
central window.  The window may be square, circular, or a superellipse
(rounded square).  RF rails begin at the window boundary and extend along
both coordinate axes.

All electrodes are represented as one planar RF/ground mask:
    1.0 -> RF electrode
    0.0 -> grounded/DC plane

This makes the result directly compatible with ``BEM2D`` and the existing
mesh tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.electrostatics.bem import BEM2D
from core.electrostatics.bem_mesh import (
    edge_aligned_axis_edges,
    geometry_aware_quadtree_rectangular_panels,
    uniform_rectangular_grid,
)

WINDOW_SQUARE = "square"
WINDOW_CIRCLE = "circle"
WINDOW_ROUNDED_SQUARE = "rounded_square"

WindowKind = Literal["square", "circle", "rounded_square"]

VALID_WINDOW_KINDS = frozenset(
    {
        WINDOW_SQUARE,
        WINDOW_CIRCLE,
        WINDOW_ROUNDED_SQUARE,
    }
)


@dataclass(frozen=True)
class CentralWindowCrossParameters:
    """C4v RF cross surrounding a grounded central opening.

    Parameters
    ----------
    window_half_size_m:
        Characteristic half-size of the grounded central window.  For a
        square this is its half-side; for a circle it is its radius; for a
        rounded square it is the superellipse semi-axis.

    window_kind:
        ``"square"``, ``"circle"``, or ``"rounded_square"``.

    superellipse_exponent:
        Only used by ``"rounded_square"``.  ``2`` gives a circle; increasing
        the exponent approaches a square.  Values from 3 to 10 are a useful
        smooth design space.

    rf_rail_width_m:
        Nominal width of every RF rail measured transverse to its arm.

    arm_length_m:
        Maximum distance from the origin where RF rails exist.

    outer_extent_m:
        Extent of the grounded plane / BEM computational square.

    transition_length_m:
        Length over which the inner and outer RF boundaries interpolate from
        their central values to their nominal straight-arm values.

    inner_edge_shift_at_window_m:
        Offset of the inner rail edge at the window. Positive values make
        the central grounded gap wider; negative values make rails approach
        the window.

    outer_edge_shift_at_window_m:
        Offset of the outer rail edge at the window. Positive values widen
        RF rails locally; negative values narrow them.

    taper_power:
        Shape exponent for the transition envelope.
    """

    window_half_size_m: float
    rf_rail_width_m: float
    arm_length_m: float
    outer_extent_m: float

    window_kind: WindowKind = WINDOW_SQUARE
    superellipse_exponent: float = 4.0

    transition_length_m: float = 0.0
    inner_edge_shift_at_window_m: float = 0.0
    outer_edge_shift_at_window_m: float = 0.0
    taper_power: float = 2.0

    min_ground_gap_m: float = 2e-6
    min_rf_width_m: float = 12e-6

    def validate(self) -> None:
        """Validate physical dimensions and topology-preserving bounds."""
        values = (
            self.window_half_size_m,
            self.rf_rail_width_m,
            self.arm_length_m,
            self.outer_extent_m,
            self.superellipse_exponent,
            self.transition_length_m,
            self.inner_edge_shift_at_window_m,
            self.outer_edge_shift_at_window_m,
            self.taper_power,
            self.min_ground_gap_m,
            self.min_rf_width_m,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("All central-window geometry parameters must be finite.")

        if self.window_kind not in VALID_WINDOW_KINDS:
            raise ValueError(
                f"Unknown window_kind={self.window_kind!r}; "
                f"choose one of {sorted(VALID_WINDOW_KINDS)}."
            )

        if self.window_half_size_m <= 0.0:
            raise ValueError("window_half_size_m must be positive.")
        if self.rf_rail_width_m <= 0.0:
            raise ValueError("rf_rail_width_m must be positive.")
        if self.arm_length_m <= self.window_half_size_m:
            raise ValueError(
                "arm_length_m must exceed window_half_size_m."
            )
        if self.outer_extent_m <= self.arm_length_m:
            raise ValueError(
                "outer_extent_m must exceed arm_length_m."
            )

        if self.window_kind == WINDOW_ROUNDED_SQUARE:
            if self.superellipse_exponent < 2.0:
                raise ValueError(
                    "superellipse_exponent must be at least 2 for a rounded square."
                )

        if self.transition_length_m < 0.0:
            raise ValueError("transition_length_m must be non-negative.")
        if self.transition_length_m > self.arm_length_m:
            raise ValueError(
                "transition_length_m cannot exceed arm_length_m."
            )
        if self.taper_power <= 0.0:
            raise ValueError("taper_power must be positive.")
        if self.min_ground_gap_m < 0.0:
            raise ValueError("min_ground_gap_m must be non-negative.")
        if self.min_rf_width_m <= 0.0:
            raise ValueError("min_rf_width_m must be positive.")

        sample_end_m = min(
            self.arm_length_m,
            self.window_half_size_m + self.transition_length_m,
        )
        samples_m = np.linspace(
            self.window_half_size_m,
            max(sample_end_m, self.window_half_size_m),
            101,
        )
        inner_m, outer_m = self.rail_boundaries_m(samples_m)

        if np.min(inner_m) <= self.min_ground_gap_m:
            raise ValueError(
                "Inner RF boundary reaches the transport axis / window gap."
            )
        if np.any(outer_m - inner_m < self.min_rf_width_m):
            raise ValueError(
                "RF rail becomes narrower than min_rf_width_m."
            )

    @property
    def nominal_inner_boundary_m(self) -> float:
        """Nominal straight-arm inner RF boundary."""
        return self.window_half_size_m

    @property
    def nominal_outer_boundary_m(self) -> float:
        """Nominal straight-arm outer RF boundary."""
        return self.window_half_size_m + self.rf_rail_width_m

    @property
    def rf_start_radius_m(self) -> float:
        """Conservative earliest longitudinal RF coordinate."""
        return self.window_half_size_m

    @property
    def maximum_rf_outer_radius_m(self) -> float:
        """Conservative outer RF radius for refinement."""
        return max(
            self.nominal_outer_boundary_m,
            self.nominal_outer_boundary_m
            + self.outer_edge_shift_at_window_m,
        )

    def transition_envelope(
        self,
        longitudinal_m: np.ndarray | float,
    ) -> np.ndarray:
        """One at the window, zero after the transition zone."""
        s_m = np.asarray(longitudinal_m, dtype=float)

        if self.transition_length_m <= 0.0:
            return np.zeros_like(s_m, dtype=float)

        normalized = np.clip(
            1.0
            - (np.abs(s_m) - self.window_half_size_m)
            / self.transition_length_m,
            0.0,
            1.0,
        )
        return normalized**self.taper_power

    def rail_boundaries_m(
        self,
        longitudinal_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return inner and outer RF boundaries for an arm.

        The values are transverse coordinates, not radial distances.  Far
        from the window, the rails are straight and have nominal width.
        """
        envelope = self.transition_envelope(longitudinal_m)

        inner_m = (
            self.nominal_inner_boundary_m
            + self.inner_edge_shift_at_window_m * envelope
        )
        outer_m = (
            self.nominal_outer_boundary_m
            + self.outer_edge_shift_at_window_m * envelope
        )

        return (
            np.asarray(inner_m, dtype=float),
            np.asarray(outer_m, dtype=float),
        )


@dataclass
class CentralWindowCrossBEMModel:
    """BEM model together with central-window geometry metadata."""

    bem: BEM2D
    parameters: CentralWindowCrossParameters
    rf_mask: np.ndarray
    panel_centres_x_m: np.ndarray
    panel_centres_y_m: np.ndarray
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray

    @property
    def n_panels(self) -> int:
        """Total number of rectangular BEM panels."""
        return self.bem.n_panels


def central_window_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: CentralWindowCrossParameters,
) -> np.ndarray:
    """Return the grounded central-window mask."""
    parameters.validate()

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x_m and y_m must have identical shapes.")

    a_m = parameters.window_half_size_m

    if parameters.window_kind == WINDOW_SQUARE:
        return (np.abs(x) <= a_m) & (np.abs(y) <= a_m)

    if parameters.window_kind == WINDOW_CIRCLE:
        return x**2 + y**2 <= a_m**2

    exponent = parameters.superellipse_exponent
    normalized = (
        np.abs(x / a_m) ** exponent
        + np.abs(y / a_m) ** exponent
    )
    return normalized <= 1.0


def central_window_cross_rf_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: CentralWindowCrossParameters,
) -> np.ndarray:
    """Return RF mask for the four-arm central-window cross.

    Horizontal RF rails lie above and below the x-axis.  Vertical RF rails
    lie left and right of the y-axis.  The central window is explicitly
    grounded, even if a rail predicate would otherwise include a point.
    """
    parameters.validate()

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x_m and y_m must have identical shapes.")

    abs_x = np.abs(x)
    abs_y = np.abs(y)

    inner_x, outer_x = parameters.rail_boundaries_m(abs_x)
    horizontal_rf = (
        (abs_x >= parameters.rf_start_radius_m)
        & (abs_x <= parameters.arm_length_m)
        & (abs_y >= inner_x)
        & (abs_y <= outer_x)
    )

    inner_y, outer_y = parameters.rail_boundaries_m(abs_y)
    vertical_rf = (
        (abs_y >= parameters.rf_start_radius_m)
        & (abs_y <= parameters.arm_length_m)
        & (abs_x >= inner_y)
        & (abs_x <= outer_y)
    )

    window = central_window_mask(x, y, parameters)
    return (horizontal_rf | vertical_rf) & ~window


def _critical_positions_m(
    parameters: CentralWindowCrossParameters,
    *,
    n_transition_samples: int,
) -> np.ndarray:
    """Return positive coordinates that should become mesh boundaries."""
    if n_transition_samples < 2:
        raise ValueError("n_transition_samples must be at least two.")

    if parameters.transition_length_m > 0.0:
        longitudinal_m = np.linspace(
            parameters.window_half_size_m,
            min(
                parameters.arm_length_m,
                parameters.window_half_size_m
                + parameters.transition_length_m,
            ),
            n_transition_samples,
        )
    else:
        longitudinal_m = np.asarray(
            [
                parameters.window_half_size_m,
                parameters.arm_length_m,
            ],
            dtype=float,
        )

    inner_m, outer_m = parameters.rail_boundaries_m(longitudinal_m)

    positions = np.concatenate(
        (
            np.asarray(
                [
                    0.0,
                    parameters.window_half_size_m,
                    parameters.nominal_outer_boundary_m,
                    parameters.arm_length_m,
                ],
                dtype=float,
            ),
            np.asarray(inner_m, dtype=float),
            np.asarray(outer_m, dtype=float),
        )
    )

    return np.unique(positions[np.isfinite(positions)])


def _rectangle_intersects_window(
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
    parameters: CentralWindowCrossParameters,
) -> bool:
    """Conservative test for a cell crossing the central-window boundary."""
    a_m = parameters.window_half_size_m

    if not (
        x_left_m <= a_m
        and x_right_m >= -a_m
        and y_bottom_m <= a_m
        and y_top_m >= -a_m
    ):
        return False

    if parameters.window_kind == WINDOW_SQUARE:
        return True

    x_closest = np.clip(0.0, x_left_m, x_right_m)
    y_closest = np.clip(0.0, y_bottom_m, y_top_m)
    x_farthest = max(abs(x_left_m), abs(x_right_m))
    y_farthest = max(abs(y_bottom_m), abs(y_top_m))

    if parameters.window_kind == WINDOW_CIRCLE:
        min_radius_sq = x_closest**2 + y_closest**2
        max_radius_sq = x_farthest**2 + y_farthest**2
        return min_radius_sq <= a_m**2 <= max_radius_sq

    exponent = parameters.superellipse_exponent
    min_value = (
        abs(x_closest / a_m) ** exponent
        + abs(y_closest / a_m) ** exponent
    )
    max_value = (
        abs(x_farthest / a_m) ** exponent
        + abs(y_farthest / a_m) ** exponent
    )
    return min_value <= 1.0 <= max_value


def _rail_boundary_intersects_cell(
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
    parameters: CentralWindowCrossParameters,
) -> bool:
    """Check horizontal RF rail boundaries against a rectangular cell."""
    abs_x_min = (
        0.0
        if x_left_m <= 0.0 <= x_right_m
        else min(abs(x_left_m), abs(x_right_m))
    )
    abs_x_max = max(abs(x_left_m), abs(x_right_m))

    start_m = parameters.rf_start_radius_m
    end_m = parameters.arm_length_m

    if abs_x_max < start_m or abs_x_min > end_m:
        return False

    left_m = max(abs_x_min, start_m)
    right_m = min(abs_x_max, end_m)

    if right_m < left_m:
        return False

    samples_m = np.linspace(left_m, right_m, 9)
    inner_m, outer_m = parameters.rail_boundaries_m(samples_m)

    for boundary_m in (
        float(np.min(inner_m)),
        float(np.max(inner_m)),
        float(np.min(outer_m)),
        float(np.max(outer_m)),
    ):
        if y_bottom_m <= boundary_m <= y_top_m:
            return True
        if y_bottom_m <= -boundary_m <= y_top_m:
            return True

    for coordinate_m in (-end_m, -start_m, start_m, end_m):
        if x_left_m <= coordinate_m <= x_right_m:
            if (
                y_bottom_m <= parameters.maximum_rf_outer_radius_m
                and y_top_m >= -parameters.maximum_rf_outer_radius_m
            ):
                return True

    return False


def central_window_cross_boundary_intersects_cell(
    x_left_m: float,
    x_right_m: float,
    y_bottom_m: float,
    y_top_m: float,
    parameters: CentralWindowCrossParameters,
) -> bool:
    """Return whether a physical RF/window boundary crosses a cell."""
    parameters.validate()

    if _rectangle_intersects_window(
        x_left_m,
        x_right_m,
        y_bottom_m,
        y_top_m,
        parameters,
    ):
        return True

    if _rail_boundary_intersects_cell(
        x_left_m,
        x_right_m,
        y_bottom_m,
        y_top_m,
        parameters,
    ):
        return True

    return _rail_boundary_intersects_cell(
        y_bottom_m,
        y_top_m,
        x_left_m,
        x_right_m,
        parameters,
    )


def build_central_window_cross_bem(
    parameters: CentralWindowCrossParameters,
    *,
    n_transition_samples: int = 17,
    max_cell_size_centre_m: float = 12.5e-6,
    max_cell_size_outer_m: float = 75e-6,
    centre_refinement_radius_m: float = 250e-6,
) -> CentralWindowCrossBEMModel:
    """Build an edge-aligned regular-grid BEM model."""
    parameters.validate()

    critical_positions_m = _critical_positions_m(
        parameters,
        n_transition_samples=n_transition_samples,
    )

    axis_edges_m = edge_aligned_axis_edges(
        outer_extent_m=parameters.outer_extent_m,
        critical_positions_m=critical_positions_m,
        max_cell_size_centre_m=max_cell_size_centre_m,
        max_cell_size_outer_m=max_cell_size_outer_m,
        centre_refinement_radius_m=centre_refinement_radius_m,
    )

    x_centres_m = 0.5 * (axis_edges_m[:-1] + axis_edges_m[1:])
    y_centres_m = 0.5 * (axis_edges_m[:-1] + axis_edges_m[1:])
    x_grid_m, y_grid_m = np.meshgrid(
        x_centres_m,
        y_centres_m,
        indexing="xy",
    )

    rf_mask = central_window_cross_rf_mask(
        x_grid_m,
        y_grid_m,
        parameters,
    )

    panels_m, voltages_v = uniform_rectangular_grid(
        x_edges_m=axis_edges_m,
        y_edges_m=axis_edges_m,
        cell_values=rf_mask.astype(float),
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    return CentralWindowCrossBEMModel(
        bem=bem,
        parameters=parameters,
        rf_mask=rf_mask,
        panel_centres_x_m=x_grid_m,
        panel_centres_y_m=y_grid_m,
        x_edges_m=axis_edges_m,
        y_edges_m=axis_edges_m,
    )


def build_geometry_aware_quadtree_central_window_cross_bem(
    parameters: CentralWindowCrossParameters,
    *,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 28e-6,
    boundary_max_cell_m: float = 10e-6,
    outer_max_cell_m: float = 180e-6,
    min_cell_m: float = 5e-6,
) -> CentralWindowCrossBEMModel:
    """Build a geometry-aware adaptive-quadtree BEM model.

    This is the preferred L1 screening builder for special central-window
    islands. It refines every panel crossed by an RF edge or the window
    boundary, while retaining large far-field grounded panels.
    """
    parameters.validate()

    def classify_point(x_m: float, y_m: float) -> float:
        mask = central_window_cross_rf_mask(
            np.asarray([x_m], dtype=float),
            np.asarray([y_m], dtype=float),
            parameters,
        )
        return float(mask[0])

    def intersects_boundary(
        x_left_m: float,
        x_right_m: float,
        y_bottom_m: float,
        y_top_m: float,
    ) -> bool:
        return central_window_cross_boundary_intersects_cell(
            x_left_m,
            x_right_m,
            y_bottom_m,
            y_top_m,
            parameters,
        )

    panels_m, voltages_v = geometry_aware_quadtree_rectangular_panels(
        outer_extent_m=parameters.outer_extent_m,
        classify_point=classify_point,
        intersects_boundary=intersects_boundary,
        central_half_extent_m=central_half_extent_m,
        central_max_cell_m=central_max_cell_m,
        boundary_max_cell_m=boundary_max_cell_m,
        outer_max_cell_m=outer_max_cell_m,
        min_cell_m=min_cell_m,
    )

    bem = BEM2D(
        panels_m=panels_m,
        electrode_voltages_v=voltages_v,
    )

    centres_m = bem.panel_centres_m

    return CentralWindowCrossBEMModel(
        bem=bem,
        parameters=parameters,
        rf_mask=voltages_v > 0.5,
        panel_centres_x_m=centres_m[:, 0],
        panel_centres_y_m=centres_m[:, 1],
        x_edges_m=np.asarray([], dtype=float),
        y_edges_m=np.asarray([], dtype=float),
    )


__all__ = [
    "CentralWindowCrossBEMModel",
    "CentralWindowCrossParameters",
    "VALID_WINDOW_KINDS",
    "WINDOW_CIRCLE",
    "WINDOW_ROUNDED_SQUARE",
    "WINDOW_SQUARE",
    "build_central_window_cross_bem",
    "build_geometry_aware_quadtree_central_window_cross_bem",
    "central_window_cross_boundary_intersects_cell",
    "central_window_cross_rf_mask",
    "central_window_mask",
]