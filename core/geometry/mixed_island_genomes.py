"""Genome definitions and operators for mixed X-junction topology islands.

Two geometry families are intentionally kept distinct:

1. ``OrdinaryIslandGenome``
   Existing movable-knot / generic RF-mask search space.

2. ``CentralWindowCrossGenome``
   A constrained C4v RF cross surrounding a grounded central window.

They share a small common protocol:
- ``family``
- ``clip()``
- ``is_physically_valid()``
- ``to_parameters()``

This allows a mixed-island workflow to evaluate both families through a
common adapter while preventing meaningless crossover between topologies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.central_window_cross import (
    CentralWindowCrossParameters,
    VALID_WINDOW_KINDS,
    WINDOW_CIRCLE,
    WINDOW_ROUNDED_SQUARE,
    WINDOW_SQUARE,
)
from core.optimization.genome import Genome, random_genome

FAMILY_ORDINARY = "ordinary"
FAMILY_CENTRAL_WINDOW_CROSS = "central_window_cross"

WINDOW_KIND_TO_CODE = {
    WINDOW_SQUARE: 0,
    WINDOW_CIRCLE: 1,
    WINDOW_ROUNDED_SQUARE: 2,
}
CODE_TO_WINDOW_KIND = {
    code: name
    for name, code in WINDOW_KIND_TO_CODE.items()
}


from typing import Any


@dataclass
class OrdinaryIslandGenome:
    """Wrapper for an ordinary-family genome representation."""

    genome: Any

    @property
    def family(self) -> str:
        return FAMILY_ORDINARY

    def copy(self) -> "OrdinaryIslandGenome":
        return OrdinaryIslandGenome(genome=self.genome.copy())


@dataclass
class CentralWindowCrossGenome:
    """Constrained genome for a grounded-window RF cross.

    ``window_kind_code`` is intentionally discrete:
        0 -> square
        1 -> circle
        2 -> rounded_square

    The remaining genes control the scale and smooth local transition from
    the central window to the straight RF arms.
    """

    window_kind_code: int = 0
    window_half_size_m: float = 55e-6
    superellipse_exponent: float = 4.0

    rf_rail_width_m: float = 120e-6
    arm_length_m: float = 600e-6
    outer_extent_m: float = 900e-6

    transition_length_m: float = 80e-6
    inner_edge_shift_at_window_m: float = 0.0
    outer_edge_shift_at_window_m: float = 0.0
    taper_power: float = 2.0

    min_ground_gap_m: float = 2e-6
    min_rf_width_m: float = 12e-6

    WINDOW_HALF_SIZE_MIN_M: float = 15e-6
    WINDOW_HALF_SIZE_MAX_M: float = 180e-6

    RF_RAIL_WIDTH_MIN_M: float = 20e-6
    RF_RAIL_WIDTH_MAX_M: float = 320e-6

    ARM_LENGTH_MIN_M: float = 200e-6
    ARM_LENGTH_MAX_M: float = 1_500e-6

    OUTER_EXTENT_MIN_M: float = 300e-6
    OUTER_EXTENT_MAX_M: float = 2_000e-6

    TRANSITION_LENGTH_MIN_M: float = 0.0
    TRANSITION_LENGTH_MAX_M: float = 350e-6

    EDGE_SHIFT_MIN_M: float = -80e-6
    EDGE_SHIFT_MAX_M: float = 100e-6

    SUPERELLIPSE_EXPONENT_MIN: float = 2.0
    SUPERELLIPSE_EXPONENT_MAX: float = 12.0

    TAPER_POWER_MIN: float = 0.75
    TAPER_POWER_MAX: float = 6.0

    @property
    def family(self) -> str:
        return FAMILY_CENTRAL_WINDOW_CROSS

    @property
    def window_kind(self) -> str:
        code = int(
            np.clip(
                round(self.window_kind_code),
                min(CODE_TO_WINDOW_KIND),
                max(CODE_TO_WINDOW_KIND),
            )
        )
        return CODE_TO_WINDOW_KIND[code]

    def copy(self) -> "CentralWindowCrossGenome":
        return CentralWindowCrossGenome(
            window_kind_code=int(self.window_kind_code),
            window_half_size_m=float(self.window_half_size_m),
            superellipse_exponent=float(self.superellipse_exponent),
            rf_rail_width_m=float(self.rf_rail_width_m),
            arm_length_m=float(self.arm_length_m),
            outer_extent_m=float(self.outer_extent_m),
            transition_length_m=float(self.transition_length_m),
            inner_edge_shift_at_window_m=float(
                self.inner_edge_shift_at_window_m
            ),
            outer_edge_shift_at_window_m=float(
                self.outer_edge_shift_at_window_m
            ),
            taper_power=float(self.taper_power),
            min_ground_gap_m=float(self.min_ground_gap_m),
            min_rf_width_m=float(self.min_rf_width_m),
        )

    @classmethod
    def vector_length(cls) -> int:
        return 10

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.window_kind_code),
                self.window_half_size_m,
                self.superellipse_exponent,
                self.rf_rail_width_m,
                self.arm_length_m,
                self.outer_extent_m,
                self.transition_length_m,
                self.inner_edge_shift_at_window_m,
                self.outer_edge_shift_at_window_m,
                self.taper_power,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        *,
        min_ground_gap_m: float = 2e-6,
        min_rf_width_m: float = 12e-6,
    ) -> "CentralWindowCrossGenome":
        values = np.asarray(vector, dtype=np.float64).reshape(-1)

        if values.shape != (cls.vector_length(),):
            raise ValueError(
                f"Expected vector length {cls.vector_length()}, "
                f"got {values.size}."
            )

        return cls(
            window_kind_code=int(round(values[0])),
            window_half_size_m=float(values[1]),
            superellipse_exponent=float(values[2]),
            rf_rail_width_m=float(values[3]),
            arm_length_m=float(values[4]),
            outer_extent_m=float(values[5]),
            transition_length_m=float(values[6]),
            inner_edge_shift_at_window_m=float(values[7]),
            outer_edge_shift_at_window_m=float(values[8]),
            taper_power=float(values[9]),
            min_ground_gap_m=min_ground_gap_m,
            min_rf_width_m=min_rf_width_m,
        )

    def clip(self) -> "CentralWindowCrossGenome":
        clipped = self.copy()

        clipped.window_kind_code = int(
            np.clip(
                round(clipped.window_kind_code),
                min(CODE_TO_WINDOW_KIND),
                max(CODE_TO_WINDOW_KIND),
            )
        )
        clipped.window_half_size_m = float(
            np.clip(
                clipped.window_half_size_m,
                self.WINDOW_HALF_SIZE_MIN_M,
                self.WINDOW_HALF_SIZE_MAX_M,
            )
        )
        clipped.superellipse_exponent = float(
            np.clip(
                clipped.superellipse_exponent,
                self.SUPERELLIPSE_EXPONENT_MIN,
                self.SUPERELLIPSE_EXPONENT_MAX,
            )
        )
        clipped.rf_rail_width_m = float(
            np.clip(
                clipped.rf_rail_width_m,
                self.RF_RAIL_WIDTH_MIN_M,
                self.RF_RAIL_WIDTH_MAX_M,
            )
        )
        clipped.arm_length_m = float(
            np.clip(
                clipped.arm_length_m,
                self.ARM_LENGTH_MIN_M,
                self.ARM_LENGTH_MAX_M,
            )
        )
        clipped.outer_extent_m = float(
            np.clip(
                clipped.outer_extent_m,
                self.OUTER_EXTENT_MIN_M,
                self.OUTER_EXTENT_MAX_M,
            )
        )
        clipped.transition_length_m = float(
            np.clip(
                clipped.transition_length_m,
                self.TRANSITION_LENGTH_MIN_M,
                self.TRANSITION_LENGTH_MAX_M,
            )
        )
        clipped.inner_edge_shift_at_window_m = float(
            np.clip(
                clipped.inner_edge_shift_at_window_m,
                self.EDGE_SHIFT_MIN_M,
                self.EDGE_SHIFT_MAX_M,
            )
        )
        clipped.outer_edge_shift_at_window_m = float(
            np.clip(
                clipped.outer_edge_shift_at_window_m,
                self.EDGE_SHIFT_MIN_M,
                self.EDGE_SHIFT_MAX_M,
            )
        )
        clipped.taper_power = float(
            np.clip(
                clipped.taper_power,
                self.TAPER_POWER_MIN,
                self.TAPER_POWER_MAX,
            )
        )

        minimum_arm_m = (
            clipped.window_half_size_m
            + clipped.transition_length_m
            + 25e-6
        )
        clipped.arm_length_m = max(
            clipped.arm_length_m,
            minimum_arm_m,
        )
        clipped.outer_extent_m = max(
            clipped.outer_extent_m,
            clipped.arm_length_m + 25e-6,
        )

        return clipped

    def to_parameters(self) -> CentralWindowCrossParameters:
        """Convert the clipped genome to validated geometry parameters."""
        clipped = self.clip()

        if clipped.window_kind not in VALID_WINDOW_KINDS:
            raise RuntimeError("Unexpected validated window kind.")

        return CentralWindowCrossParameters(
            window_half_size_m=clipped.window_half_size_m,
            rf_rail_width_m=clipped.rf_rail_width_m,
            arm_length_m=clipped.arm_length_m,
            outer_extent_m=clipped.outer_extent_m,
            window_kind=clipped.window_kind,
            superellipse_exponent=clipped.superellipse_exponent,
            transition_length_m=clipped.transition_length_m,
            inner_edge_shift_at_window_m=(
                clipped.inner_edge_shift_at_window_m
            ),
            outer_edge_shift_at_window_m=(
                clipped.outer_edge_shift_at_window_m
            ),
            taper_power=clipped.taper_power,
            min_ground_gap_m=clipped.min_ground_gap_m,
            min_rf_width_m=clipped.min_rf_width_m,
        )

    def is_physically_valid(self) -> bool:
        try:
            self.to_parameters().validate()
        except ValueError:
            return False
        return True

    def summary(self) -> dict[str, object]:
        return {
            "family": self.family,
            "window_kind": self.window_kind,
            "window_half_size_m": float(self.window_half_size_m),
            "superellipse_exponent": float(
                self.superellipse_exponent
            ),
            "rf_rail_width_m": float(self.rf_rail_width_m),
            "arm_length_m": float(self.arm_length_m),
            "outer_extent_m": float(self.outer_extent_m),
            "transition_length_m": float(self.transition_length_m),
            "inner_edge_shift_at_window_m": float(
                self.inner_edge_shift_at_window_m
            ),
            "outer_edge_shift_at_window_m": float(
                self.outer_edge_shift_at_window_m
            ),
            "taper_power": float(self.taper_power),
        }


MixedIslandGenome = OrdinaryIslandGenome | CentralWindowCrossGenome


def random_central_window_cross_genome(
    rng: np.random.Generator,
    *,
    broad: bool = False,
    arm_length_m: float = 600e-6,
    outer_extent_m: float = 900e-6,
) -> CentralWindowCrossGenome:
    """Sample a physically valid special-family genome."""
    for _ in range(256):
        if broad:
            genome = CentralWindowCrossGenome(
                window_kind_code=int(rng.integers(0, 3)),
                window_half_size_m=float(rng.uniform(20e-6, 150e-6)),
                superellipse_exponent=float(rng.uniform(2.0, 10.0)),
                rf_rail_width_m=float(rng.uniform(35e-6, 260e-6)),
                arm_length_m=float(rng.uniform(350e-6, 1_100e-6)),
                outer_extent_m=float(rng.uniform(600e-6, 1_500e-6)),
                transition_length_m=float(rng.uniform(0.0, 250e-6)),
                inner_edge_shift_at_window_m=float(
                    rng.uniform(-30e-6, 60e-6)
                ),
                outer_edge_shift_at_window_m=float(
                    rng.uniform(-45e-6, 90e-6)
                ),
                taper_power=float(rng.uniform(0.9, 5.0)),
            )
        else:
            genome = CentralWindowCrossGenome(
                window_kind_code=int(rng.integers(0, 3)),
                window_half_size_m=float(rng.uniform(35e-6, 90e-6)),
                superellipse_exponent=float(rng.uniform(3.0, 7.0)),
                rf_rail_width_m=float(rng.uniform(80e-6, 180e-6)),
                arm_length_m=arm_length_m,
                outer_extent_m=outer_extent_m,
                transition_length_m=float(rng.uniform(25e-6, 140e-6)),
                inner_edge_shift_at_window_m=float(
                    rng.normal(0.0, 12e-6)
                ),
                outer_edge_shift_at_window_m=float(
                    rng.normal(0.0, 18e-6)
                ),
                taper_power=float(rng.uniform(1.2, 3.5)),
            )

        genome = genome.clip()
        if genome.is_physically_valid():
            return genome

    raise RuntimeError(
        "Could not generate a valid central-window-cross genome "
        "in 256 attempts; check sampling ranges and geometry constraints."
    )


def random_mixed_island_genome(
    rng: np.random.Generator,
    *,
    special_fraction: float = 0.30,
    broad: bool = False,
    arm_length_m: float = 600e-6,
    outer_extent_m: float = 900e-6,
) -> MixedIslandGenome:
    """Sample an ordinary or special genome according to the island mix."""
    if not 0.0 <= special_fraction <= 1.0:
        raise ValueError("special_fraction must be between zero and one.")

    if rng.random() < special_fraction:
        return random_central_window_cross_genome(
            rng,
            broad=broad,
            arm_length_m=arm_length_m,
            outer_extent_m=outer_extent_m,
        )

    return OrdinaryIslandGenome(
        genome=random_genome(rng, broad=broad),
    )


def crossover_central_window_cross(
    parent_a: CentralWindowCrossGenome,
    parent_b: CentralWindowCrossGenome,
    rng: np.random.Generator,
) -> CentralWindowCrossGenome:
    """Arithmetic crossover for continuous special-family genes."""
    alpha = rng.uniform(0.0, 1.0, size=CentralWindowCrossGenome.vector_length())

    vector_a = parent_a.to_vector()
    vector_b = parent_b.to_vector()
    child_vector = alpha * vector_a + (1.0 - alpha) * vector_b

    child_vector[0] = (
        vector_a[0] if rng.random() < 0.5 else vector_b[0]
    )

    return CentralWindowCrossGenome.from_vector(
        child_vector,
        min_ground_gap_m=parent_a.min_ground_gap_m,
        min_rf_width_m=parent_a.min_rf_width_m,
    ).clip()


def mutate_central_window_cross(
    genome: CentralWindowCrossGenome,
    rng: np.random.Generator,
    *,
    mutation_probability: float = 0.25,
    scale: float = 1.0,
) -> CentralWindowCrossGenome:
    """Gaussian mutation with physically meaningful per-gene scales."""
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must be between zero and one.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    child = genome.copy()

    if rng.random() < mutation_probability:
        child.window_kind_code = int(rng.integers(0, 3))
    if rng.random() < mutation_probability:
        child.window_half_size_m += rng.normal(0.0, 14e-6 * scale)
    if rng.random() < mutation_probability:
        child.superellipse_exponent += rng.normal(0.0, 1.0 * scale)
    if rng.random() < mutation_probability:
        child.rf_rail_width_m += rng.normal(0.0, 22e-6 * scale)
    if rng.random() < mutation_probability:
        child.transition_length_m += rng.normal(0.0, 25e-6 * scale)
    if rng.random() < mutation_probability:
        child.inner_edge_shift_at_window_m += rng.normal(
            0.0,
            10e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.outer_edge_shift_at_window_m += rng.normal(
            0.0,
            14e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.taper_power += rng.normal(0.0, 0.45 * scale)
    child = child.clip()

    minimum_inner_shift_m = (
        child.min_ground_gap_m
        + 1e-9
        - child.window_half_size_m
    )
    child.inner_edge_shift_at_window_m = max(
        child.inner_edge_shift_at_window_m,
        minimum_inner_shift_m,
    )

    minimum_outer_shift_m = (
        child.inner_edge_shift_at_window_m
        + child.min_rf_width_m
        - child.rf_rail_width_m
    )
    child.outer_edge_shift_at_window_m = max(
        child.outer_edge_shift_at_window_m,
        minimum_outer_shift_m,
    )

    child = child.clip()

    if child.is_physically_valid():
        return child

    return genome.copy()


def crossover_same_family(
    parent_a: MixedIslandGenome,
    parent_b: MixedIslandGenome,
    rng: np.random.Generator,
) -> MixedIslandGenome:
    """Crossover only within a topology family.

    Cross-family recombination is deliberately prohibited. Mixed islands
    exchange information through selection/migration rather than by blending
    incompatible geometry encodings.
    """
    if parent_a.family != parent_b.family:
        raise ValueError(
            "Cross-family crossover is prohibited; "
            "select parents from the same topology family."
        )

    if isinstance(parent_a, CentralWindowCrossGenome):
        if not isinstance(parent_b, CentralWindowCrossGenome):
            raise TypeError("Special-family parent types do not match.")
        return crossover_central_window_cross(parent_a, parent_b, rng)

    if not isinstance(parent_a, OrdinaryIslandGenome):
        raise TypeError("Unsupported ordinary-family genome type.")
    if not isinstance(parent_b, OrdinaryIslandGenome):
        raise TypeError("Ordinary-family parent types do not match.")

    vector_a = parent_a.genome.to_vector()
    vector_b = parent_b.genome.to_vector()
    alpha = rng.uniform(0.0, 1.0, size=vector_a.size)

    return OrdinaryIslandGenome(
        genome=Genome.from_vector(
            alpha * vector_a + (1.0 - alpha) * vector_b
        ).clip()
    )


__all__ = [
    "CentralWindowCrossGenome",
    "CODE_TO_WINDOW_KIND",
    "FAMILY_CENTRAL_WINDOW_CROSS",
    "FAMILY_ORDINARY",
    "MixedIslandGenome",
    "OrdinaryIslandGenome",
    "WINDOW_KIND_TO_CODE",
    "crossover_central_window_cross",
    "crossover_same_family",
    "mutate_central_window_cross",
    "random_central_window_cross_genome",
    "random_mixed_island_genome",
]