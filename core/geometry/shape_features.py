"""Rasterization of simple local geometry features on a BEM mesh."""

from __future__ import annotations

import numpy as np

from core.optimization.genome import (
    FEATURE_CIRCLE,
    FEATURE_NONE,
    FEATURE_RECTANGLE,
    FEATURE_OP_ADD_RF,
    FEATURE_OP_REMOVE_RF,
    Genome,
    N_FEATURES,
)


def _points_in_triangle(
    x_m: np.ndarray,
    y_m: np.ndarray,
    vertices_m: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for points inside a triangle."""
    vertex_a, vertex_b, vertex_c = vertices_m

    def signed_area(
        point_x: np.ndarray,
        point_y: np.ndarray,
        point_a: np.ndarray,
        point_b: np.ndarray,
    ) -> np.ndarray:
        return (
            (point_x - point_b[0]) * (point_a[1] - point_b[1])
            - (point_a[0] - point_b[0]) * (point_y - point_b[1])
        )

    sign_1 = signed_area(x_m, y_m, vertex_a, vertex_b)
    sign_2 = signed_area(x_m, y_m, vertex_b, vertex_c)
    sign_3 = signed_area(x_m, y_m, vertex_c, vertex_a)

    has_negative = (sign_1 < 0.0) | (sign_2 < 0.0) | (sign_3 < 0.0)
    has_positive = (sign_1 > 0.0) | (sign_2 > 0.0) | (sign_3 > 0.0)

    return ~(has_negative & has_positive)


def rectangle_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    center_x_m: float,
    center_y_m: float,
    width_m: float,
    height_m: float,
    angle_rad: float,
) -> np.ndarray:
    """Return points inside a rotated rectangle."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)

    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)

    dx = x - center_x_m
    dy = y - center_y_m

    xr = cosine * dx + sine * dy
    yr = -sine * dx + cosine * dy

    return (np.abs(xr) <= 0.5 * width_m) & (np.abs(yr) <= 0.5 * height_m)


def circle_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
) -> np.ndarray:
    """Return points inside a circle."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    return (x - center_x_m) ** 2 + (y - center_y_m) ** 2 <= radius_m ** 2


def triangle_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    angle_rad: float,
) -> np.ndarray:
    """Return points inside an equilateral triangle."""
    angles = np.array(
        [
            np.pi / 2.0,
            np.pi / 2.0 + 2.0 * np.pi / 3.0,
            np.pi / 2.0 + 4.0 * np.pi / 3.0,
        ]
    ) + angle_rad

    vertices_m = np.column_stack(
        [
            center_x_m + radius_m * np.cos(angles),
            center_y_m + radius_m * np.sin(angles),
        ]
    )
    return _points_in_triangle(x_m, y_m, vertices_m)


def rasterize_feature_primitive(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    feature_kind: int,
    center_x_m: float,
    center_y_m: float,
    p1_m: float,
    p2_m: float,
    angle_rad: float,
) -> np.ndarray:
    """Rasterize one feature primitive on arrays of x/y coordinates."""
    if feature_kind == FEATURE_NONE:
        return np.zeros_like(np.asarray(x_m, dtype=bool), dtype=bool)

    if feature_kind == FEATURE_CIRCLE:
        return circle_mask(
            x_m,
            y_m,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
            radius_m=p1_m,
        )

    if feature_kind == FEATURE_RECTANGLE:
        return rectangle_mask(
            x_m,
            y_m,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
            width_m=p1_m,
            height_m=p2_m,
            angle_rad=angle_rad,
        )

    raise ValueError(f"Unsupported feature kind: {feature_kind}")


def feature_union_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    genome: Genome,
    feature_index: int,
) -> np.ndarray:
    if feature_index < 0 or feature_index >= N_FEATURES:
        raise IndexError("feature_index is out of range")

    feature_kind = int(genome.feature_kind[feature_index])
    if feature_kind == FEATURE_NONE:
        return np.zeros_like(np.asarray(x_m, dtype=bool), dtype=bool)

    radius_m = float(genome.feature_radius_m[feature_index])
    theta_rad = float(genome.feature_theta_rad[feature_index])
    p1_m = float(genome.feature_p1_m[feature_index])
    p2_m = float(genome.feature_p2_m[feature_index])
    angle_rad = float(genome.feature_angle_rad[feature_index])

    feature_mask = np.zeros_like(np.asarray(x_m, dtype=bool), dtype=bool)

    for reflection in (-1.0, 1.0):
        for quarter_turn in range(4):
            theta = reflection * theta_rad + quarter_turn * (np.pi / 2.0)
            center_x_m = radius_m * np.cos(theta)
            center_y_m = radius_m * np.sin(theta)

            primitive = rasterize_feature_primitive(
                x_m,
                y_m,
                feature_kind=feature_kind,
                center_x_m=center_x_m,
                center_y_m=center_y_m,
                p1_m=p1_m,
                p2_m=p2_m,
                angle_rad=reflection * angle_rad + quarter_turn * (np.pi / 2.0),
            )
            feature_mask |= primitive

    return feature_mask


def apply_feature_to_mask(
    mask: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    genome: Genome,
    feature_index: int,
) -> None:
    operation = int(genome.feature_operation[feature_index])
    feature_mask = feature_union_mask(x_m, y_m, genome, feature_index)

    if operation == FEATURE_OP_ADD_RF:
        mask[feature_mask] = True
    elif operation == FEATURE_OP_REMOVE_RF:
        mask[feature_mask] = False
    else:
        raise ValueError(f"Unsupported feature operation: {operation}")
