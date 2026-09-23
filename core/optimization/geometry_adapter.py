from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.geometry.arm_boundary import (
    MODE_SPLINE,
    evaluate_inner_outer_boundaries,
    normalized_arm_coordinate,
)
from core.geometry.shape_features import apply_feature_to_mask
from core.optimization.genome import (
    Genome,
    N_FEATURES,
    TOPOLOGY_CENTRAL_RF_CROSS,
    TOPOLOGY_CENTRAL_RF_DISK,
    TOPOLOGY_CENTRAL_RF_RING,
    TOPOLOGY_GROUNDED_MOAT,
    TOPOLOGY_OPEN_CROSS,
)


@dataclass(frozen=True)
class GeometryAdapterConfig:
    arm_length_m: float = 900e-6
    boundary_mode: str = MODE_SPLINE


def genome_to_boundary_arrays(
    genome: Genome,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    config: GeometryAdapterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = config or GeometryAdapterConfig()

    s, longitudinal_m, transverse_m, along_x = normalized_arm_coordinate(
        x_m,
        y_m,
        arm_length_m=cfg.arm_length_m,
    )

    inner_boundary_m, outer_boundary_m = evaluate_inner_outer_boundaries(
        genome.inner_control_m,
        genome.outer_control_m,
        s,
        mode=cfg.boundary_mode,
        clamp=True,
    )

    return s, longitudinal_m, transverse_m, along_x, inner_boundary_m, outer_boundary_m


def build_simple_mask_from_genome(
    genome: Genome,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    config: GeometryAdapterConfig | None = None,
) -> np.ndarray:
    cfg = config or GeometryAdapterConfig()

    (
        _s,
        longitudinal_m,
        transverse_m,
        _along_x,
        inner_boundary_m,
        outer_boundary_m,
    ) = genome_to_boundary_arrays(
        genome,
        x_m,
        y_m,
        config=cfg,
    )

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)

    mask = (
        (longitudinal_m <= cfg.arm_length_m)
        & (transverse_m >= inner_boundary_m)
        & (transverse_m <= outer_boundary_m)
    )

    radial_m = np.hypot(x, y)
    topology = int(genome.topology)
    size_m = float(genome.topology_size_m)
    width_m = float(genome.topology_width_m)

    if topology == TOPOLOGY_OPEN_CROSS:
        pass
    elif topology == TOPOLOGY_CENTRAL_RF_DISK:
        mask[radial_m <= size_m] = True
    elif topology == TOPOLOGY_CENTRAL_RF_RING:
        ring = np.abs(radial_m - size_m) <= 0.5 * width_m
        mask[ring] = True
    elif topology == TOPOLOGY_CENTRAL_RF_CROSS:
        central_cross = (radial_m <= size_m) & (
            (np.abs(x) <= 0.5 * width_m) | (np.abs(y) <= 0.5 * width_m)
        )
        mask[central_cross] = True
    elif topology == TOPOLOGY_GROUNDED_MOAT:
        moat = np.abs(radial_m - size_m) <= 0.5 * width_m
        mask[moat] = False
    else:
        raise ValueError(f"Unsupported topology index: {topology}")

    mask = mask.astype(bool, copy=True)
    for feature_index in range(N_FEATURES):
        apply_feature_to_mask(mask, x, y, genome, feature_index)

    return mask.astype(np.float64)


def build_simple_masks_from_population(
    population: list[Genome],
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    config: GeometryAdapterConfig | None = None,
) -> np.ndarray:
    return np.stack(
        [
            build_simple_mask_from_genome(
                genome,
                x_m,
                y_m,
                config=config,
            )
            for genome in population
        ],
        axis=0,
    )


def genome_to_builder_kwargs(genome: Genome) -> dict[str, Any]:
    return {
        "inner_control_m": genome.inner_control_m.copy(),
        "outer_control_m": genome.outer_control_m.copy(),
        "topology": int(genome.topology),
        "topology_size_m": float(genome.topology_size_m),
        "topology_width_m": float(genome.topology_width_m),
        "feature_kind": genome.feature_kind.copy(),
        "feature_operation": genome.feature_operation.copy(),
        "feature_radius_m": genome.feature_radius_m.copy(),
        "feature_theta_rad": genome.feature_theta_rad.copy(),
        "feature_p1_m": genome.feature_p1_m.copy(),
        "feature_p2_m": genome.feature_p2_m.copy(),
        "feature_angle_rad": genome.feature_angle_rad.copy(),
    }