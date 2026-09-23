"""Connected polygonal RF rails for fourfold-symmetric planar X-junctions.

The RF electrode is represented as connected polygons. Local triangular
features modify the RF/DC boundary itself; they are not disconnected islands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolygonalXJunctionParameters:
    """Physical parameters for a symmetric polygonal RF-junction layout."""

    ground_rail_width_m: float
    rf_rail_width_m: float
    arm_length_m: float
    rf_start_radius_m: float

    transition_radius_m: float
    inner_radius_at_start_m: float
    outer_radius_at_start_m: float

    bump_longitudinal_positions_m: tuple[float, ...] = ()
    bump_heights_m: tuple[float, ...] = ()
    bump_half_widths_m: tuple[float, ...] = ()

    def validate(self) -> None:
        """Validate dimensions and triangular boundary-feature arrays."""
        all_scalars = (
            self.ground_rail_width_m,
            self.rf_rail_width_m,
            self.arm_length_m,
            self.rf_start_radius_m,
            self.transition_radius_m,
            self.inner_radius_at_start_m,
            self.outer_radius_at_start_m,
        )

        if not np.all(np.isfinite(all_scalars)):
            raise ValueError("Junction scalar parameters must be finite.")

        if self.ground_rail_width_m <= 0.0:
            raise ValueError("ground_rail_width_m must be positive.")
        if self.rf_rail_width_m <= 0.0:
            raise ValueError("rf_rail_width_m must be positive.")
        if self.arm_length_m <= 0.0:
            raise ValueError("arm_length_m must be positive.")
        if self.rf_start_radius_m < 0.0:
            raise ValueError("rf_start_radius_m must be non-negative.")
        if self.transition_radius_m <= self.rf_start_radius_m:
            raise ValueError(
                "transition_radius_m must exceed rf_start_radius_m."
            )
        if self.transition_radius_m >= self.arm_length_m:
            raise ValueError(
                "transition_radius_m must be smaller than arm_length_m."
            )

        if self.inner_radius_at_start_m <= 0.0:
            raise ValueError("inner_radius_at_start_m must be positive.")
        if (
            self.outer_radius_at_start_m
            <= self.inner_radius_at_start_m
        ):
            raise ValueError(
                "outer_radius_at_start_m must exceed inner_radius_at_start_m."
            )

        positions_m = np.asarray(
            self.bump_longitudinal_positions_m,
            dtype=float,
        )
        heights_m = np.asarray(
            self.bump_heights_m,
            dtype=float,
        )
        widths_m = np.asarray(
            self.bump_half_widths_m,
            dtype=float,
        )

        if not (
            positions_m.size == heights_m.size == widths_m.size
        ):
            raise ValueError(
                "Bump position, height, and half-width arrays must "
                "have equal length."
            )

        if positions_m.size == 0:
            return

        if not (
            np.all(np.isfinite(positions_m))
            and np.all(np.isfinite(heights_m))
            and np.all(np.isfinite(widths_m))
        ):
            raise ValueError("Bump arrays must be finite.")

        if np.any(positions_m <= self.rf_start_radius_m):
            raise ValueError(
                "Each bump must lie strictly beyond rf_start_radius_m."
            )
        if np.any(positions_m >= self.transition_radius_m):
            raise ValueError(
                "Each bump must lie inside the transition region."
            )
        if np.any(widths_m <= 0.0):
            raise ValueError("Each bump half-width must be positive.")

        if np.any(
            positions_m - widths_m <= self.rf_start_radius_m
        ):
            raise ValueError(
                "A bump overlaps the central RF-start boundary."
            )
        if np.any(
            positions_m + widths_m >= self.transition_radius_m
        ):
            raise ValueError(
                "A bump overlaps the straight-arm transition boundary."
            )

        order = np.argsort(positions_m)
        sorted_positions_m = positions_m[order]
        sorted_widths_m = widths_m[order]

        if sorted_positions_m.size > 1:
            separations_m = np.diff(sorted_positions_m)
            required_m = (
                sorted_widths_m[:-1] + sorted_widths_m[1:]
            )

            if np.any(separations_m <= required_m):
                raise ValueError("Triangular bumps overlap.")

    @property
    def half_ground_width_m(self) -> float:
        """Straight-arm inner RF radius."""
        return 0.5 * self.ground_rail_width_m

    @property
    def straight_outer_radius_m(self) -> float:
        """Straight-arm outer RF radius."""
        return self.half_ground_width_m + self.rf_rail_width_m

    def rail_boundaries_m(
        self,
        longitudinal_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return unbumped inner/outer boundaries of one RF rail."""
        s_m = np.abs(np.asarray(longitudinal_m, dtype=float))

        span_m = self.transition_radius_m - self.rf_start_radius_m
        fraction = np.clip(
            (s_m - self.rf_start_radius_m) / span_m,
            0.0,
            1.0,
        )

        inner_transition_m = (
            self.inner_radius_at_start_m
            + fraction
            * (
                self.half_ground_width_m
                - self.inner_radius_at_start_m
            )
        )
        outer_transition_m = (
            self.outer_radius_at_start_m
            + fraction
            * (
                self.straight_outer_radius_m
                - self.outer_radius_at_start_m
            )
        )

        inner_m = np.where(
            s_m >= self.transition_radius_m,
            self.half_ground_width_m,
            inner_transition_m,
        )
        outer_m = np.where(
            s_m >= self.transition_radius_m,
            self.straight_outer_radius_m,
            outer_transition_m,
        )

        return inner_m, outer_m


def make_polygonal_house_x_junction(
    *,
    ion_height_m: float,
    arm_length_m: float = 600e-6,
    rf_start_radius_m: float = 30e-6,
    transition_radius_m: float = 180e-6,
    inner_radius_at_start_m: float = 16e-6,
    outer_radius_at_start_m: float = 135e-6,
    bump_longitudinal_positions_m: tuple[float, ...] = (),
    bump_heights_m: tuple[float, ...] = (),
    bump_half_widths_m: tuple[float, ...] = (),
) -> PolygonalXJunctionParameters:
    """Create a House-ratio RF rail layout with optional contour bumps."""
    return PolygonalXJunctionParameters(
        ground_rail_width_m=0.83 * ion_height_m,
        rf_rail_width_m=1.99 * ion_height_m,
        arm_length_m=arm_length_m,
        rf_start_radius_m=rf_start_radius_m,
        transition_radius_m=transition_radius_m,
        inner_radius_at_start_m=inner_radius_at_start_m,
        outer_radius_at_start_m=outer_radius_at_start_m,
        bump_longitudinal_positions_m=bump_longitudinal_positions_m,
        bump_heights_m=bump_heights_m,
        bump_half_widths_m=bump_half_widths_m,
    )


def _outer_edge_chain_m(
    parameters: PolygonalXJunctionParameters,
    *,
    samples_per_segment: int = 12,
) -> np.ndarray:
    """Build a positive-s outer boundary chain with connected triangular bumps.

    Every bump replaces a local base segment by:
        (s-hw, r_out) -> (s, r_out + height) -> (s+hw, r_out).

    It is therefore part of the electrode perimeter, not a separate polygon.
    """
    parameters.validate()

    start_m = parameters.rf_start_radius_m
    transition_m = parameters.transition_radius_m

    positions_m = np.asarray(
        parameters.bump_longitudinal_positions_m,
        dtype=float,
    )
    heights_m = np.asarray(
        parameters.bump_heights_m,
        dtype=float,
    )
    widths_m = np.asarray(
        parameters.bump_half_widths_m,
        dtype=float,
    )

    order = np.argsort(positions_m)
    positions_m = positions_m[order]
    heights_m = heights_m[order]
    widths_m = widths_m[order]

    breakpoints_m = [start_m]

    for position_m, width_m in zip(positions_m, widths_m):
        breakpoints_m.extend(
            [position_m - width_m, position_m + width_m]
        )

    breakpoints_m.append(transition_m)

    chain: list[tuple[float, float]] = []

    def append_base_segment(left_m: float, right_m: float) -> None:
        if right_m <= left_m:
            return

        segment_s_m = np.linspace(
            left_m,
            right_m,
            samples_per_segment,
            endpoint=True,
        )
        _, segment_outer_m = parameters.rail_boundaries_m(segment_s_m)

        for s_m, outer_m in zip(
            segment_s_m,
            segment_outer_m,
        ):
            point = (float(s_m), float(outer_m))

            if not chain or point != chain[-1]:
                chain.append(point)

    previous_m = start_m

    for position_m, height_m, width_m in zip(
        positions_m,
        heights_m,
        widths_m,
    ):
        left_m = position_m - width_m
        right_m = position_m + width_m

        append_base_segment(previous_m, left_m)

        _, base_outer_m = parameters.rail_boundaries_m(
            np.asarray([position_m], dtype=float)
        )

        left_point = (left_m, float(base_outer_m[0]))
        apex_point = (
            float(position_m),
            float(base_outer_m[0] + height_m),
        )
        right_point = (right_m, float(base_outer_m[0]))

        if not chain or left_point != chain[-1]:
            chain.append(left_point)

        chain.append(apex_point)
        chain.append(right_point)

        previous_m = right_m

    append_base_segment(previous_m, transition_m)

    return np.asarray(chain, dtype=float)


def _positive_horizontal_rail_polygon_m(
    parameters: PolygonalXJunctionParameters,
) -> np.ndarray:
    """Create one connected upper-right RF rail polygon.

    The polygon begins at the inner boundary at the RF start, follows the
    inner edge to the transition, closes across the straight arm width, and
    returns along a possibly bumped outer edge.
    """
    parameters.validate()

    s_m = np.linspace(
        parameters.rf_start_radius_m,
        parameters.transition_radius_m,
        160,
    )
    inner_m, outer_m = parameters.rail_boundaries_m(s_m)

    inner_chain = np.column_stack((s_m, inner_m))
    outer_chain = _outer_edge_chain_m(parameters)

    outer_end_s_m = outer_chain[-1, 0]
    outer_end_y_m = outer_chain[-1, 1]

    if not np.isclose(
        outer_end_s_m,
        parameters.transition_radius_m,
    ):
        raise RuntimeError(
            "Outer contour chain does not reach transition radius."
        )

    polygon_m = np.vstack(
        (
            inner_chain,
            np.array(
                [
                    (
                        parameters.transition_radius_m,
                        outer_m[-1],
                    )
                ],
                dtype=float,
            ),
            outer_chain[::-1],
        )
    )

    return polygon_m


def _rotate_points_90_m(
    points_m: np.ndarray,
    quarter_turns: int,
) -> np.ndarray:
    """Rotate points counter-clockwise by an integer number of 90° turns."""
    points = np.asarray(points_m, dtype=float)
    turns = quarter_turns % 4

    if turns == 0:
        return points.copy()
    if turns == 1:
        return np.column_stack((-points[:, 1], points[:, 0]))
    if turns == 2:
        return -points
    return np.column_stack((points[:, 1], -points[:, 0]))


def fourfold_rf_rail_polygons_m(
    parameters: PolygonalXJunctionParameters,
) -> tuple[np.ndarray, ...]:
    """Return four connected RF-rail polygons with exact C4 symmetry.

    A positive horizontal rail from +x toward +y is reflected and rotated
    to obtain all arms.
    """
    base = _positive_horizontal_rail_polygon_m(parameters)

    upper_right = base
    upper_left = np.column_stack((-base[:, 0], base[:, 1]))
    lower_left = -base
    lower_right = np.column_stack((base[:, 0], -base[:, 1]))

    return (
        upper_right,
        upper_left,
        lower_left,
        lower_right,
        _rotate_points_90_m(upper_right, 1),
        _rotate_points_90_m(upper_left, 1),
        _rotate_points_90_m(lower_left, 1),
        _rotate_points_90_m(lower_right, 1),
    )


def point_in_polygon_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    polygon_m: np.ndarray,
) -> np.ndarray:
    """Return whether points lie inside a simple closed polygon.

    Uses vectorized ray casting. Boundary points are treated as inside for
    raster-mask construction.
    """
    x_values_m = np.asarray(x_m, dtype=float)
    y_values_m = np.asarray(y_m, dtype=float)
    polygon = np.asarray(polygon_m, dtype=float)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon_m must have shape (n, 2).")
    if polygon.shape[0] < 3:
        raise ValueError("A polygon needs at least three vertices.")

    vertices_x = polygon[:, 0]
    vertices_y = polygon[:, 1]

    inside = np.zeros_like(x_values_m, dtype=bool)

    previous = polygon.shape[0] - 1

    for current in range(polygon.shape[0]):
        x_current = vertices_x[current]
        y_current = vertices_y[current]
        x_previous = vertices_x[previous]
        y_previous = vertices_y[previous]

        crosses = (
            (y_current > y_values_m)
            != (y_previous > y_values_m)
        )

        x_intersection = (
            (x_previous - x_current)
            * (y_values_m - y_current)
            / (y_previous - y_current + 1e-30)
            + x_current
        )

        inside ^= crosses & (x_values_m < x_intersection)
        previous = current

    return inside


def polygonal_x_junction_rf_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: PolygonalXJunctionParameters,
) -> np.ndarray:
    """Return RF mask made from connected polygonal RF rails.

    No disconnected wedges are added: triangular features are inserted into
    the outer contour chain of each RF rail polygon.
    """
    parameters.validate()

    x_values_m = np.asarray(x_m, dtype=float)
    y_values_m = np.asarray(y_m, dtype=float)

    if x_values_m.shape != y_values_m.shape:
        raise ValueError("x_m and y_m must have identical shapes.")

    rf_mask = np.zeros_like(x_values_m, dtype=bool)

    for polygon_m in fourfold_rf_rail_polygons_m(parameters):
        rf_mask |= point_in_polygon_mask(
            x_values_m,
            y_values_m,
            polygon_m,
        )

    return rf_mask