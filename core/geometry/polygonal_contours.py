"""Connected polygonal RF-boundary features for planar X-junction masks.

The utilities here do not create isolated RF islands. They describe local
boundary displacements of an already connected RF rail. A triangular feature
is represented as three consecutive vertices of the rail boundary polygon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriangularBoundaryFeature:
    """A triangular displacement of one continuous RF boundary.

    ``centre_s_m`` is the longitudinal position along a rail.
    ``normal_offset_m`` is the signed displacement of the apex along the
    outward normal. Positive values expand RF into neighbouring non-RF area;
    negative values create an inward notch.

    The base half-width determines how much of the original straight boundary
    is replaced by the two sloped polygon segments.
    """

    centre_s_m: float
    normal_offset_m: float
    half_base_m: float

    def validate(self) -> None:
        values = (
            self.centre_s_m,
            self.normal_offset_m,
            self.half_base_m,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Triangular boundary feature values must be finite.")
        if self.centre_s_m < 0.0:
            raise ValueError("centre_s_m must be non-negative.")
        if self.half_base_m <= 0.0:
            raise ValueError("half_base_m must be positive.")


def validate_nonoverlapping_features(
    features: tuple[TriangularBoundaryFeature, ...],
    *,
    minimum_spacing_m: float = 0.0,
) -> None:
    """Ensure triangular boundary replacements do not overlap."""
    if minimum_spacing_m < 0.0:
        raise ValueError("minimum_spacing_m must be non-negative.")

    ordered = sorted(features, key=lambda feature: feature.centre_s_m)

    for feature in ordered:
        feature.validate()

    for left, right in zip(ordered[:-1], ordered[1:]):
        left_end_m = left.centre_s_m + left.half_base_m
        right_start_m = right.centre_s_m - right.half_base_m

        if right_start_m < left_end_m + minimum_spacing_m:
            raise ValueError(
                "Triangular boundary features overlap or violate minimum spacing."
            )


def outer_boundary_profile_m(
    longitudinal_s_m: np.ndarray | float,
    *,
    base_outer_m,
    features: tuple[TriangularBoundaryFeature, ...],
) -> np.ndarray:
    """Return outer boundary with piecewise-linear triangular modifications.

    ``base_outer_m`` is a callable which accepts a NumPy array of nonnegative
    longitudinal coordinates and returns the original outer RF boundary.

    Each feature applies a triangular signed offset. The resulting profile is
    continuous and represents one connected RF electrode boundary.
    """
    validate_nonoverlapping_features(features)

    s_m = np.asarray(longitudinal_s_m, dtype=float)
    s_abs_m = np.abs(s_m)

    boundary_m = np.asarray(base_outer_m(s_abs_m), dtype=float).copy()

    for feature in features:
        left_m = feature.centre_s_m - feature.half_base_m
        right_m = feature.centre_s_m + feature.half_base_m

        inside = (s_abs_m >= left_m) & (s_abs_m <= right_m)

        if not np.any(inside):
            continue

        local_distance_m = np.abs(
            s_abs_m[inside] - feature.centre_s_m
        )

        triangular_weight = 1.0 - local_distance_m / feature.half_base_m

        boundary_m[inside] += (
            feature.normal_offset_m * triangular_weight
        )

    return boundary_m


def point_in_polygon_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    vertices_m: np.ndarray,
) -> np.ndarray:
    """Vectorized ray-casting test for points inside one simple polygon.

    Points exactly on an edge are treated as inside to avoid holes between
    contiguous RF pieces in a discretized mask.
    """
    x_values_m = np.asarray(x_m, dtype=float)
    y_values_m = np.asarray(y_m, dtype=float)
    vertices = np.asarray(vertices_m, dtype=float)

    if x_values_m.shape != y_values_m.shape:
        raise ValueError("x_m and y_m must have identical shapes.")
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices_m must have shape (n_vertices, 2).")
    if vertices.shape[0] < 3:
        raise ValueError("A polygon needs at least three vertices.")

    inside = np.zeros_like(x_values_m, dtype=bool)

    x_previous_m, y_previous_m = vertices[-1]

    for x_current_m, y_current_m in vertices:
        crosses = (
            (y_current_m > y_values_m)
            != (y_previous_m > y_values_m)
        )

        x_intersection_m = (
            (x_previous_m - x_current_m)
            * (y_values_m - y_current_m)
            / (y_previous_m - y_current_m + 1e-30)
            + x_current_m
        )

        inside ^= crosses & (x_values_m < x_intersection_m)

        x_previous_m = x_current_m
        y_previous_m = y_current_m

    return inside


def _polygon_for_upper_horizontal_rail(
    *,
    start_m: float,
    end_m: float,
    inner_boundary_m,
    outer_boundary_m,
    n_boundary_samples: int,
) -> np.ndarray:
    """Return one closed upper-horizontal RF rail polygon."""
    s_m = np.linspace(start_m, end_m, n_boundary_samples)

    inner_m = np.asarray(inner_boundary_m(s_m), dtype=float)
    outer_m = np.asarray(outer_boundary_m(s_m), dtype=float)

    inner_chain = np.column_stack((s_m, inner_m))
    outer_chain = np.column_stack((s_m[::-1], outer_m[::-1]))

    return np.vstack((inner_chain, outer_chain))


def rotate_vertices_m(
    vertices_m: np.ndarray,
    quarter_turns: int,
) -> np.ndarray:
    """Rotate planar polygon vertices by integer multiples of 90 degrees."""
    vertices = np.asarray(vertices_m, dtype=float)
    turns = quarter_turns % 4

    if turns == 0:
        return vertices.copy()
    if turns == 1:
        return np.column_stack((-vertices[:, 1], vertices[:, 0]))
    if turns == 2:
        return -vertices
    return np.column_stack((vertices[:, 1], -vertices[:, 0]))


def fourfold_connected_rf_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    start_m: float,
    end_m: float,
    inner_boundary_m,
    base_outer_boundary_m,
    features: tuple[TriangularBoundaryFeature, ...],
    n_boundary_samples: int = 801,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return RF mask from four connected polygonal rail copies.

    A single upper-horizontal rail is constructed as a closed polygon.
    Its outer boundary contains triangular piecewise-linear contour features.
    Rotating that one polygon produces the other three rails. The result
    has exact fourfold symmetry and each feature is attached to a rail,
    never floating as an isolated RF island.
    """
    if start_m < 0.0:
        raise ValueError("start_m must be non-negative.")
    if end_m <= start_m:
        raise ValueError("end_m must exceed start_m.")
    if n_boundary_samples < 8:
        raise ValueError("n_boundary_samples must be at least eight.")

    validate_nonoverlapping_features(features)

    for feature in features:
        if feature.centre_s_m - feature.half_base_m < start_m:
            raise ValueError(
                "A triangular feature begins before the RF rail start."
            )
        if feature.centre_s_m + feature.half_base_m > end_m:
            raise ValueError(
                "A triangular feature extends past the RF rail end."
            )

    def featured_outer_boundary_m(s_m: np.ndarray) -> np.ndarray:
        return outer_boundary_profile_m(
            s_m,
            base_outer_m=base_outer_boundary_m,
            features=features,
        )

    upper_polygon_m = _polygon_for_upper_horizontal_rail(
        start_m=start_m,
        end_m=end_m,
        inner_boundary_m=inner_boundary_m,
        outer_boundary_m=featured_outer_boundary_m,
        n_boundary_samples=n_boundary_samples,
    )

    polygons_m = tuple(
        rotate_vertices_m(upper_polygon_m, turn)
        for turn in range(4)
    )

    rf_mask = np.zeros_like(np.asarray(x_m, dtype=float), dtype=bool)

    for polygon_m in polygons_m:
        rf_mask |= point_in_polygon_mask(
            x_m,
            y_m,
            polygon_m,
        )

    return rf_mask, polygons_m