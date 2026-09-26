"""Common BEM adapter for ordinary and central-window-cross islands."""

from __future__ import annotations

from typing import TypeAlias

from core.geometry.central_window_cross import (
    CentralWindowCrossBEMModel,
    build_geometry_aware_quadtree_central_window_cross_bem,
)
from core.geometry.junction_templates import XJunctionParameters
from core.geometry.mask_builder import (
    XJunctionBEMModel,
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.geometry.mixed_island_genomes import (
    CentralWindowCrossGenome,
    OrdinaryIslandGenome,
)

MixedBEMModel: TypeAlias = XJunctionBEMModel | CentralWindowCrossBEMModel


def build_mixed_geometry_aware_bem(
    genome: OrdinaryIslandGenome | CentralWindowCrossGenome,
    *,
    ordinary_parameters: XJunctionParameters | None = None,
    central_half_extent_m: float = 180e-6,
    central_max_cell_m: float = 28e-6,
    boundary_max_cell_m: float = 10e-6,
    outer_max_cell_m: float = 180e-6,
    min_cell_m: float = 5e-6,
) -> MixedBEMModel:
    """Build the appropriate geometry-aware BEM model for one island genome.

    ``ordinary_parameters`` is deliberately explicit because the existing
    ordinary movable-knot genome and the older XJunctionParameters template
    are separate representations in the current codebase.
    """
    common_kwargs = {
        "central_half_extent_m": central_half_extent_m,
        "central_max_cell_m": central_max_cell_m,
        "boundary_max_cell_m": boundary_max_cell_m,
        "outer_max_cell_m": outer_max_cell_m,
        "min_cell_m": min_cell_m,
    }

    if isinstance(genome, CentralWindowCrossGenome):
        return build_geometry_aware_quadtree_central_window_cross_bem(
            genome.to_parameters(),
            **common_kwargs,
        )

    if isinstance(genome, OrdinaryIslandGenome):
        if ordinary_parameters is None:
            raise ValueError(
                "ordinary_parameters is required for an OrdinaryIslandGenome."
            )

        return build_geometry_aware_quadtree_x_junction_bem(
            ordinary_parameters,
            **common_kwargs,
        )

    raise TypeError(f"Unsupported genome type: {type(genome)!r}")


__all__ = [
    "MixedBEMModel",
    "build_mixed_geometry_aware_bem",
]