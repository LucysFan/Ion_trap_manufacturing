from __future__ import annotations

from typing import Any

import numpy as np

from core.ga.candidates import Candidate
from core.geometry.rf_shapes import rectangle_mask, spline_values
from core.optimization.genome import (
    FEATURE_CIRCLE,
    FEATURE_NONE,
    FEATURE_RECTANGLE,
    Genome,
    TOPOLOGY_CENTRAL_RF_CROSS,
    TOPOLOGY_CENTRAL_RF_DISK,
    TOPOLOGY_CENTRAL_RF_RING,
    TOPOLOGY_GROUNDED_MOAT,
)


GeometryLike = Candidate | Genome


def _inner_control(geometry: GeometryLike) -> np.ndarray:
    if isinstance(geometry, Genome):
        return np.asarray(geometry.inner_control_m, dtype=float)
    return np.asarray(geometry.inner_m, dtype=float)


def _outer_control(geometry: GeometryLike) -> np.ndarray:
    if isinstance(geometry, Genome):
        return np.asarray(geometry.outer_control_m, dtype=float)
    return np.asarray(geometry.outer_m, dtype=float)


def _resolve_arm_length_m(
    cfg: Any | None = None,
    *,
    arm_length_m: float | None = None,
) -> float:
    if arm_length_m is not None:
        return float(arm_length_m)

    if cfg is not None and hasattr(cfg, "arm_length_m"):
        return float(cfg.arm_length_m)

    raise ValueError(
        "arm_length_m is required: pass cfg with arm_length_m "
        "or use arm_length_m=..."
    )


def apply_c4v_feature(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: GeometryLike,
    feature_index: int,
) -> None:
    kind = int(geometry.feature_kind[feature_index])
    if kind == FEATURE_NONE:
        return

    operation = int(geometry.feature_operation[feature_index])
    radius = float(geometry.feature_radius_m[feature_index])
    theta0 = float(geometry.feature_theta_rad[feature_index])
    p1 = float(geometry.feature_p1_m[feature_index])
    p2 = float(geometry.feature_p2_m[feature_index])
    angle0 = float(geometry.feature_angle_rad[feature_index])

    feature_mask = np.zeros_like(mask, dtype=bool)

    for reflection in (-1.0, 1.0):
        for quarter_turn in range(4):
            theta = reflection * theta0 + quarter_turn * np.pi / 2.0
            cx = radius * np.cos(theta)
            cy = radius * np.sin(theta)

            if kind == FEATURE_CIRCLE:
                primitive = (x - cx) ** 2 + (y - cy) ** 2 <= p1 ** 2
            elif kind == FEATURE_RECTANGLE:
                primitive = rectangle_mask(
                    x,
                    y,
                    cx,
                    cy,
                    p1,
                    p2,
                    reflection * angle0 + quarter_turn * np.pi / 2.0,
                )
            else:
                continue

            feature_mask |= primitive

    if operation > 0:
        mask[feature_mask] = True
    else:
        mask[feature_mask] = False


def rf_mask(
    geometry: GeometryLike,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Any | None = None,
    *,
    arm_length_m: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    arm_length_m = _resolve_arm_length_m(cfg, arm_length_m=arm_length_m)

    ax = np.abs(x)
    ay = np.abs(y)
    along_x = ax >= ay
    longitudinal = np.where(along_x, ax, ay)
    transverse = np.where(along_x, ay, ax)

    inner = spline_values(
        _inner_control(geometry),
        longitudinal / arm_length_m,
    )
    outer = spline_values(
        _outer_control(geometry),
        longitudinal / arm_length_m,
    )

    mask = (
        (longitudinal <= arm_length_m)
        & (transverse >= inner)
        & (transverse <= outer)
    )

    radial = np.hypot(x, y)
    size = float(geometry.topology_size_m)
    width = float(geometry.topology_width_m)
    topology = int(geometry.topology)

    if topology == TOPOLOGY_CENTRAL_RF_DISK:
        mask[radial <= size] = True
    elif topology == TOPOLOGY_CENTRAL_RF_RING:
        ring = np.abs(radial - size) <= width / 2.0
        mask[ring] = True
    elif topology == TOPOLOGY_CENTRAL_RF_CROSS:
        central_cross = (
            (radial <= size)
            & (
                (np.abs(x) <= width / 2.0)
                | (np.abs(y) <= width / 2.0)
            )
        )
        mask[central_cross] = True
    elif topology == TOPOLOGY_GROUNDED_MOAT:
        moat = np.abs(radial - size) <= width / 2.0
        mask[moat] = False

    for feature_index in range(len(geometry.feature_kind)):
        apply_c4v_feature(mask, x, y, geometry, feature_index)

    return mask.astype(np.float64, copy=False)


def build_masks(
    population: list[GeometryLike],
    x: np.ndarray,
    y: np.ndarray,
    cfg: Any | None = None,
    *,
    arm_length_m: float | None = None,
) -> np.ndarray:
    arm_length_m = _resolve_arm_length_m(cfg, arm_length_m=arm_length_m)
    return np.stack(
        [
            rf_mask(
                geometry,
                x,
                y,
                arm_length_m=arm_length_m,
            )
            for geometry in population
        ]
    )


__all__ = ["apply_c4v_feature", "build_masks", "rf_mask"]