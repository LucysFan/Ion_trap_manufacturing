"""Step 10: mixed-topology island GA for planar RF X-junction layouts.

The population is split into two families:

- ordinary:
  Conventional connected X-junctions based on the existing
  ``XJunctionParameters`` template. This is the classical family.

- central_window_cross:
  Special C4v RF cross around a grounded central square, circular, or
  rounded-square window.

The families are NOT isolated:

1. Routine ring migration passes elites through all islands.
2. When an elite enters an island of the other family, it is translated into
   the target family through physically meaningful geometry parameters.
3. A stagnant island enters a burst mode:
   - mutation strength sharply increases;
   - weak candidates are replaced;
   - translated elites from the other family are injected;
   - broad random immigrants are injected.

This keeps the baseline connected-cross branch alive while letting special
central-window solutions influence it and vice versa.

Run from repository root:

    python workflows/10_mixed_topology_island_ga.py
"""

from __future__ import annotations

import argparse

from collections import Counter
from functools import partial

from matplotlib import animation

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.targets import (
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
)
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.path_metrics import compute_path_metrics
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.validation import validate_rf_transport_path
from core.geometry.central_window_cross import (
    CentralWindowCrossParameters,
    build_geometry_aware_quadtree_central_window_cross_bem,
)
from core.geometry.junction_templates import (
    XJunctionParameters,
    make_house_style_x_junction,
)
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.geometry.mixed_island_genomes import (
    CentralWindowCrossGenome,
    OrdinaryIslandGenome,
    crossover_central_window_cross,
    mutate_central_window_cross,
    random_central_window_cross_genome,
)


Family = Literal["ordinary", "central_window_cross"]
AnyGenome = OrdinaryIslandGenome | CentralWindowCrossGenome


# =============================================================================
# Search budget
# =============================================================================

SEED = 20260926

# Debug / first physically complete run.
#
# Production target after validation:
#     N_ORDINARY_ISLANDS = 21
#     N_SPECIAL_ISLANDS = 9
#     POPULATION_PER_ISLAND = 15 or 20
#
# Do not increase production budget before introducing an L0/L1/L2 evaluator
# cascade. This workflow does a full independent BEM solve per candidate.
N_ORDINARY_ISLANDS = 5
N_SPECIAL_ISLANDS = 3
POPULATION_PER_ISLAND = 8
GENERATIONS = 12
MAX_BEM_EVALUATIONS = 768

ELITE_COUNT = 2
TOURNAMENT_SIZE = 3

# Routine migration through a ring containing all islands.
MIGRATION_INTERVAL = 3
MIGRANTS_PER_ISLAND = 1

# Burst / escape from stagnation.
STAGNATION_GENERATIONS = 4
BURST_GENERATIONS = 2
BURST_MUTATION_MULTIPLIER = 3.0
BURST_MUTATION_PROBABILITY = 0.85
BURST_REPLACE_FRACTION = 0.50
BURST_RANDOM_FRACTION = 0.25

# Relative improvement required to reset local stagnation.
RELATIVE_IMPROVEMENT_TOLERANCE = 0.01

# Annealed baseline mutation amplitude.
INITIAL_MUTATION_SCALE = 1.00
FINAL_MUTATION_SCALE = 0.30

# Physical route used for every candidate.
X_VALUES_M = np.linspace(-350e-6, 350e-6, 41)

# Geometry-aware adaptive BEM mesh.
CENTRAL_HALF_EXTENT_M = 180e-6
CENTRAL_MAX_CELL_M = 30e-6
BOUNDARY_MAX_CELL_M = 10e-6
OUTER_MAX_CELL_M = 180e-6
MIN_CELL_M = 5e-6

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "10_mixed_topology_island_ga"
)


# =============================================================================
# Data model
# =============================================================================

@dataclass
class CandidateEvaluation:
    """Physical BEM evaluation and selection data for one candidate."""

    island_id: int
    generation: int
    family: Family
    genome: AnyGenome

    valid: bool
    score: float

    height_peak_m: float = math.inf
    height_rms_m: float = math.inf
    lateral_peak_m: float = math.inf
    barrier_ev: float = math.inf

    n_panels: int = 0
    trace_valid: bool = False
    path_valid: bool = False
    reason: str = ""

    def to_row(self) -> dict[str, object]:
        """Convert to a CSV-compatible flat dictionary."""
        row: dict[str, object] = {
            "generation": self.generation,
            "island_id": self.island_id,
            "family": self.family,
            "valid": self.valid,
            "score": self.score,
            "trace_valid": self.trace_valid,
            "path_valid": self.path_valid,
            "height_peak_m": self.height_peak_m,
            "height_rms_m": self.height_rms_m,
            "lateral_peak_m": self.lateral_peak_m,
            "barrier_ev": self.barrier_ev,
            "n_panels": self.n_panels,
            "reason": self.reason,
        }

        if isinstance(self.genome, CentralWindowCrossGenome):
            row.update(
                {
                    "window_kind": self.genome.window_kind,
                    "window_half_size_m": (
                        self.genome.window_half_size_m
                    ),
                    "superellipse_exponent": (
                        self.genome.superellipse_exponent
                    ),
                    "rf_rail_width_m": (
                        self.genome.rf_rail_width_m
                    ),
                    "arm_length_m": self.genome.arm_length_m,
                    "outer_extent_m": self.genome.outer_extent_m,
                    "transition_length_m": (
                        self.genome.transition_length_m
                    ),
                    "inner_edge_shift_at_window_m": (
                        self.genome.inner_edge_shift_at_window_m
                    ),
                    "outer_edge_shift_at_window_m": (
                        self.genome.outer_edge_shift_at_window_m
                    ),
                    "taper_power": self.genome.taper_power,
                }
            )
        else:
            ordinary = unwrap_ordinary(self.genome)
            row.update(
                {
                    "central_island_kind": ordinary.island_kind,
                    "central_island_radius_m": (
                        ordinary.central_island_radius_m
                    ),
                    "rf_start_radius_m": (
                        ordinary.rf_start_radius_m
                    ),
                    "ordinary_taper_length_m": (
                        ordinary.taper_length_m
                    ),
                    "ordinary_inner_shift_m": (
                        ordinary.inner_shift_m
                    ),
                    "ordinary_outer_shift_m": (
                        ordinary.outer_shift_m
                    ),
                    "ordinary_taper_power": (
                        ordinary.taper_power
                    ),
                    "inner_offset_60_m": (
                        ordinary.inner_offset_60_m
                    ),
                    "inner_offset_90_m": (
                        ordinary.inner_offset_90_m
                    ),
                    "inner_offset_120_m": (
                        ordinary.inner_offset_120_m
                    ),
                }
            )

        return row


@dataclass
class Island:
    """A local population with independent stagnation/burst state."""

    island_id: int
    family: Family
    population: list[AnyGenome] = field(default_factory=list)

    best_score_seen: float = math.inf
    stagnant_generations: int = 0
    burst_generations_remaining: int = 0
    burst_count: int = 0

    def in_burst(self) -> bool:
        """Return whether this island currently has burst exploration active."""
        return self.burst_generations_remaining > 0


# =============================================================================
# Classical ordinary family
# =============================================================================

@dataclass
class OrdinaryTemplateGenome:
    """Compact conventional connected-X-junction genome.

    This family modifies the validated classical template rather than building
    disconnected artificial RF elements. It has one central grounded opening,
    C4v symmetry, local taper control, and one connected inner contour.

    The endpoints of the inner contour are always zero, so only the interior
    control values at 60, 90 and 120 um are evolved.
    """

    central_island_radius_m: float = 35e-6
    central_island_kind_code: int = 0

    rf_start_radius_m: float = 30e-6
    taper_length_m: float = 150e-6
    inner_shift_m: float = -25e-6
    outer_shift_m: float = 20e-6
    taper_power: float = 2.0

    inner_offset_60_m: float = 0.0
    inner_offset_90_m: float = 0.0
    inner_offset_120_m: float = 0.0

    @property
    def island_kind(self) -> str:
        """Return the valid central grounded island kind."""
        return (
            "square"
            if int(self.central_island_kind_code) == 0
            else "circle"
        )

    def copy(self) -> "OrdinaryTemplateGenome":
        """Return a detached copy."""
        return OrdinaryTemplateGenome(
            central_island_radius_m=self.central_island_radius_m,
            central_island_kind_code=self.central_island_kind_code,
            rf_start_radius_m=self.rf_start_radius_m,
            taper_length_m=self.taper_length_m,
            inner_shift_m=self.inner_shift_m,
            outer_shift_m=self.outer_shift_m,
            taper_power=self.taper_power,
            inner_offset_60_m=self.inner_offset_60_m,
            inner_offset_90_m=self.inner_offset_90_m,
            inner_offset_120_m=self.inner_offset_120_m,
        )

    def clip(self) -> "OrdinaryTemplateGenome":
        """Clip genes and repair basic coupled coordinate constraints."""
        child = self.copy()

        child.central_island_kind_code = int(
            np.clip(round(child.central_island_kind_code), 0, 1)
        )
        child.central_island_radius_m = float(
            np.clip(child.central_island_radius_m, 0.0, 80e-6)
        )
        child.rf_start_radius_m = float(
            np.clip(child.rf_start_radius_m, 30e-6, 120e-6)
        )
        child.taper_length_m = float(
            np.clip(child.taper_length_m, 125e-6, 220e-6)
        )
        child.inner_shift_m = float(
            np.clip(child.inner_shift_m, -25e-6, 20e-6)
        )
        child.outer_shift_m = float(
            np.clip(child.outer_shift_m, -15e-6, 65e-6)
        )
        child.taper_power = float(
            np.clip(child.taper_power, 1.0, 4.0)
        )
        child.inner_offset_60_m = float(
            np.clip(child.inner_offset_60_m, -20e-6, 15e-6)
        )
        child.inner_offset_90_m = float(
            np.clip(child.inner_offset_90_m, -20e-6, 15e-6)
        )
        child.inner_offset_120_m = float(
            np.clip(child.inner_offset_120_m, -20e-6, 15e-6)
        )

        child.rf_start_radius_m = max(
            child.rf_start_radius_m,
            child.central_island_radius_m,
        )
        child.taper_length_m = max(
            child.taper_length_m,
            125e-6,
        )

        return child

    def to_parameters(self) -> XJunctionParameters:
        """Translate this ordinary genome into the existing geometry template."""
        child = self.clip()

        parameters = make_house_style_x_junction(
            ion_height_m=TARGET_ION_HEIGHT_M,
            arm_length_m=600e-6,
            outer_extent_m=900e-6,
            central_island_radius_m=child.central_island_radius_m,
            central_island_kind=child.island_kind,
            taper_length_m=child.taper_length_m,
            inner_edge_shift_at_centre_m=child.inner_shift_m,
            outer_edge_shift_at_centre_m=child.outer_shift_m,
            taper_power=child.taper_power,
            rf_start_radius_override_m=child.rf_start_radius_m,
            inner_contour_knots_m=(
                child.rf_start_radius_m,
                60e-6,
                90e-6,
                120e-6,
                child.taper_length_m,
            ),
            inner_contour_offsets_m=(
                0.0,
                child.inner_offset_60_m,
                child.inner_offset_90_m,
                child.inner_offset_120_m,
                0.0,
            ),
        )

        parameters.validate()
        return parameters

    def is_physically_valid(self) -> bool:
        """Check parameter and manufacturing constraints."""
        try:
            parameters = self.to_parameters()
            report = check_x_junction_manufacturability(parameters)
        except ValueError:
            return False

        return bool(report.valid)

    def summary(self) -> dict[str, object]:
        """Return JSON-safe genome information."""
        return {
            "family": "ordinary",
            "central_island_kind": self.island_kind,
            "central_island_radius_m": self.central_island_radius_m,
            "rf_start_radius_m": self.rf_start_radius_m,
            "taper_length_m": self.taper_length_m,
            "inner_shift_m": self.inner_shift_m,
            "outer_shift_m": self.outer_shift_m,
            "taper_power": self.taper_power,
            "inner_offset_60_m": self.inner_offset_60_m,
            "inner_offset_90_m": self.inner_offset_90_m,
            "inner_offset_120_m": self.inner_offset_120_m,
        }


def random_ordinary_template_genome(
    rng: np.random.Generator,
    *,
    broad: bool = False,
) -> OrdinaryTemplateGenome:
    """Generate one physically valid conventional connected-cross genome."""
    for _ in range(512):
        if broad:
            genome = OrdinaryTemplateGenome(
                central_island_radius_m=float(
                    rng.uniform(0.0, 75e-6)
                ),
                central_island_kind_code=int(rng.integers(0, 2)),
                rf_start_radius_m=float(
                    rng.uniform(30e-6, 115e-6)
                ),
                taper_length_m=float(
                    rng.uniform(125e-6, 220e-6)
                ),
                inner_shift_m=float(
                    rng.uniform(-25e-6, 20e-6)
                ),
                outer_shift_m=float(
                    rng.uniform(-15e-6, 60e-6)
                ),
                taper_power=float(
                    rng.uniform(1.0, 4.0)
                ),
                inner_offset_60_m=float(
                    rng.uniform(-20e-6, 15e-6)
                ),
                inner_offset_90_m=float(
                    rng.uniform(-20e-6, 15e-6)
                ),
                inner_offset_120_m=float(
                    rng.uniform(-20e-6, 15e-6)
                ),
            )
        else:
            genome = OrdinaryTemplateGenome(
                central_island_radius_m=float(
                    rng.uniform(0.0, 55e-6)
                ),
                central_island_kind_code=int(rng.integers(0, 2)),
                rf_start_radius_m=float(
                    rng.uniform(30e-6, 75e-6)
                ),
                taper_length_m=float(
                    rng.uniform(135e-6, 185e-6)
                ),
                inner_shift_m=float(
                    rng.uniform(-22e-6, 5e-6)
                ),
                outer_shift_m=float(
                    rng.uniform(5e-6, 45e-6)
                ),
                taper_power=float(
                    rng.uniform(1.25, 3.0)
                ),
                inner_offset_60_m=float(
                    rng.uniform(-15e-6, 5e-6)
                ),
                inner_offset_90_m=float(
                    rng.uniform(-18e-6, 5e-6)
                ),
                inner_offset_120_m=float(
                    rng.uniform(-15e-6, 5e-6)
                ),
            )

        genome = genome.clip()

        if genome.is_physically_valid():
            return genome

    raise RuntimeError(
        "Unable to sample a valid ordinary template genome "
        "after 512 attempts."
    )


def crossover_ordinary_template(
    parent_a: OrdinaryTemplateGenome,
    parent_b: OrdinaryTemplateGenome,
    rng: np.random.Generator,
) -> OrdinaryTemplateGenome:
    """Arithmetic crossover for continuous conventional geometry genes."""
    alpha = rng.uniform(0.0, 1.0)

    return OrdinaryTemplateGenome(
        central_island_radius_m=(
            alpha * parent_a.central_island_radius_m
            + (1.0 - alpha) * parent_b.central_island_radius_m
        ),
        central_island_kind_code=(
            parent_a.central_island_kind_code
            if rng.random() < 0.5
            else parent_b.central_island_kind_code
        ),
        rf_start_radius_m=(
            alpha * parent_a.rf_start_radius_m
            + (1.0 - alpha) * parent_b.rf_start_radius_m
        ),
        taper_length_m=(
            alpha * parent_a.taper_length_m
            + (1.0 - alpha) * parent_b.taper_length_m
        ),
        inner_shift_m=(
            alpha * parent_a.inner_shift_m
            + (1.0 - alpha) * parent_b.inner_shift_m
        ),
        outer_shift_m=(
            alpha * parent_a.outer_shift_m
            + (1.0 - alpha) * parent_b.outer_shift_m
        ),
        taper_power=(
            alpha * parent_a.taper_power
            + (1.0 - alpha) * parent_b.taper_power
        ),
        inner_offset_60_m=(
            alpha * parent_a.inner_offset_60_m
            + (1.0 - alpha) * parent_b.inner_offset_60_m
        ),
        inner_offset_90_m=(
            alpha * parent_a.inner_offset_90_m
            + (1.0 - alpha) * parent_b.inner_offset_90_m
        ),
        inner_offset_120_m=(
            alpha * parent_a.inner_offset_120_m
            + (1.0 - alpha) * parent_b.inner_offset_120_m
        ),
    ).clip()


def mutate_ordinary_template(
    genome: OrdinaryTemplateGenome,
    rng: np.random.Generator,
    *,
    mutation_probability: float,
    scale: float,
) -> OrdinaryTemplateGenome:
    """Mutate a conventional genome while never returning invalid geometry."""
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError(
            "mutation_probability must be between zero and one."
        )
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    child = genome.copy()

    if rng.random() < mutation_probability:
        child.central_island_kind_code = int(rng.integers(0, 2))
    if rng.random() < mutation_probability:
        child.central_island_radius_m += rng.normal(
            0.0,
            12e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.rf_start_radius_m += rng.normal(
            0.0,
            12e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.taper_length_m += rng.normal(
            0.0,
            18e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.inner_shift_m += rng.normal(
            0.0,
            7e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.outer_shift_m += rng.normal(
            0.0,
            10e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.taper_power += rng.normal(
            0.0,
            0.35 * scale,
        )
    if rng.random() < mutation_probability:
        child.inner_offset_60_m += rng.normal(
            0.0,
            6e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.inner_offset_90_m += rng.normal(
            0.0,
            6e-6 * scale,
        )
    if rng.random() < mutation_probability:
        child.inner_offset_120_m += rng.normal(
            0.0,
            6e-6 * scale,
        )

    child = child.clip()
    return child if child.is_physically_valid() else genome.copy()


# =============================================================================
# Representation helpers
# =============================================================================

def unwrap_ordinary(
    genome: AnyGenome,
) -> OrdinaryTemplateGenome:
    """Extract the workflow-10 ordinary template representation."""
    if not isinstance(genome, OrdinaryIslandGenome):
        raise TypeError("Expected OrdinaryIslandGenome.")

    payload = genome.genome

    if not isinstance(payload, OrdinaryTemplateGenome):
        raise TypeError(
            "Workflow 10 requires "
            "OrdinaryIslandGenome.genome to contain "
            "OrdinaryTemplateGenome."
        )

    return payload


def make_ordinary_wrapper(
    genome: OrdinaryTemplateGenome,
) -> OrdinaryIslandGenome:
    """Wrap a classical template genome for the mixed-island API."""
    return OrdinaryIslandGenome(genome=genome)  # type: ignore[arg-type]


def clone_genome(
    genome: AnyGenome,
) -> AnyGenome:
    """Return a detached copy of either family."""
    if isinstance(genome, CentralWindowCrossGenome):
        return genome.copy()

    return make_ordinary_wrapper(
        unwrap_ordinary(genome).copy()
    )


def random_genome_for_family(
    family: Family,
    rng: np.random.Generator,
    *,
    broad: bool,
) -> AnyGenome:
    """Return a fresh valid genome in the requested family."""
    if family == "central_window_cross":
        return random_central_window_cross_genome(
            rng,
            broad=broad,
            arm_length_m=600e-6,
            outer_extent_m=900e-6,
        )

    return make_ordinary_wrapper(
        random_ordinary_template_genome(
            rng,
            broad=broad,
        )
    )


# =============================================================================
# Cross-family genome translation
# =============================================================================

def translate_ordinary_to_special(
    source: OrdinaryIslandGenome,
    rng: np.random.Generator,
) -> CentralWindowCrossGenome:
    """Translate conventional connected-cross information to special genes.

    Geometry correspondence:

    - ordinary grounded island / RF start -> special central-window size;
    - ordinary local inner/outer shifts -> special window-transition shifts;
    - ordinary taper length/power -> special transition length/power;
    - classical square/circle -> special square/circle.

    The special topology remains physically constrained and validated after
    mapping. A small perturbation of the superellipse exponent avoids exact
    duplicate seeds during cross-family migration.
    """
    ordinary = unwrap_ordinary(source).clip()

    window_half_size_m = max(
        ordinary.central_island_radius_m,
        ordinary.rf_start_radius_m,
        25e-6,
    )

    rf_rail_width_m = (
        1.99 * TARGET_ION_HEIGHT_M
        + ordinary.outer_shift_m
        - ordinary.inner_shift_m
    )

    special = CentralWindowCrossGenome(
        window_kind_code=int(ordinary.central_island_kind_code),
        window_half_size_m=window_half_size_m,
        superellipse_exponent=float(
            np.clip(
                4.0 + rng.normal(0.0, 0.35),
                2.0,
                8.0,
            )
        ),
        rf_rail_width_m=rf_rail_width_m,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        transition_length_m=max(
            ordinary.taper_length_m,
            30e-6,
        ),
        inner_edge_shift_at_window_m=ordinary.inner_shift_m,
        outer_edge_shift_at_window_m=ordinary.outer_shift_m,
        taper_power=ordinary.taper_power,
    ).clip()

    if special.is_physically_valid():
        return special

    fallback = CentralWindowCrossGenome(
        window_kind_code=int(ordinary.central_island_kind_code),
        window_half_size_m=max(
            35e-6,
            min(ordinary.rf_start_radius_m, 85e-6),
        ),
        superellipse_exponent=4.0,
        rf_rail_width_m=140e-6,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        transition_length_m=120e-6,
        inner_edge_shift_at_window_m=-5e-6,
        outer_edge_shift_at_window_m=20e-6,
        taper_power=2.0,
    ).clip()

    if not fallback.is_physically_valid():
        raise RuntimeError(
            "Central-window fallback translation became invalid."
        )

    return fallback


def translate_special_to_ordinary(
    source: CentralWindowCrossGenome,
    rng: np.random.Generator,
) -> OrdinaryIslandGenome:
    """Translate central-window-cross information to ordinary genes.

    Rounded-square windows map to ordinary square islands because the
    classical template only has square and circular central openings.
    """
    special = source.clip()

    kind_code = (
        1
        if special.window_kind == "circle"
        else 0
    )

    ordinary = OrdinaryTemplateGenome(
        central_island_radius_m=special.window_half_size_m,
        central_island_kind_code=kind_code,
        rf_start_radius_m=special.window_half_size_m,
        taper_length_m=max(
            125e-6,
            min(special.transition_length_m, 220e-6),
        ),
        inner_shift_m=special.inner_edge_shift_at_window_m,
        outer_shift_m=special.outer_edge_shift_at_window_m,
        taper_power=special.taper_power,
        inner_offset_60_m=float(rng.normal(0.0, 3e-6)),
        inner_offset_90_m=float(rng.normal(0.0, 4e-6)),
        inner_offset_120_m=float(rng.normal(0.0, 3e-6)),
    ).clip()

    if ordinary.is_physically_valid():
        return make_ordinary_wrapper(ordinary)

    fallback = OrdinaryTemplateGenome(
        central_island_radius_m=min(
            special.window_half_size_m,
            55e-6,
        ),
        central_island_kind_code=kind_code,
        rf_start_radius_m=max(
            special.window_half_size_m,
            30e-6,
        ),
        taper_length_m=150e-6,
        inner_shift_m=-10e-6,
        outer_shift_m=20e-6,
        taper_power=2.0,
        inner_offset_60_m=0.0,
        inner_offset_90_m=0.0,
        inner_offset_120_m=0.0,
    ).clip()

    if not fallback.is_physically_valid():
        raise RuntimeError(
            "Ordinary fallback translation became invalid."
        )

    return make_ordinary_wrapper(fallback)


def translate_to_family(
    genome: AnyGenome,
    *,
    destination_family: Family,
    rng: np.random.Generator,
) -> AnyGenome:
    """Copy or physically translate a candidate to another family."""
    if destination_family == "ordinary":
        if isinstance(genome, OrdinaryIslandGenome):
            return clone_genome(genome)

        if isinstance(genome, CentralWindowCrossGenome):
            return translate_special_to_ordinary(genome, rng)

    if destination_family == "central_window_cross":
        if isinstance(genome, CentralWindowCrossGenome):
            return genome.copy()

        if isinstance(genome, OrdinaryIslandGenome):
            return translate_ordinary_to_special(genome, rng)

    raise TypeError(
        f"Cannot translate {type(genome)!r} "
        f"to {destination_family!r}."
    )


# =============================================================================
# BEM build and physical evaluation
# =============================================================================

def candidate_geometry_parameters(
    genome: AnyGenome,
) -> XJunctionParameters | CentralWindowCrossParameters:
    """Convert a genome to its corresponding physical geometry object."""
    if isinstance(genome, CentralWindowCrossGenome):
        return genome.to_parameters()

    return unwrap_ordinary(genome).to_parameters()


def build_candidate_bem(
    parameters: XJunctionParameters | CentralWindowCrossParameters,
):
    """Build a geometry-aware BEM model for either topology family."""
    mesh_kwargs = {
        "central_half_extent_m": CENTRAL_HALF_EXTENT_M,
        "central_max_cell_m": CENTRAL_MAX_CELL_M,
        "boundary_max_cell_m": BOUNDARY_MAX_CELL_M,
        "outer_max_cell_m": OUTER_MAX_CELL_M,
        "min_cell_m": MIN_CELL_M,
    }

    if isinstance(parameters, CentralWindowCrossParameters):
        return build_geometry_aware_quadtree_central_window_cross_bem(
            parameters,
            **mesh_kwargs,
        )

    return build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        **mesh_kwargs,
    )


def score_metrics(
    *,
    trace_valid: bool,
    path_valid: bool,
    height_peak_m: float,
    height_rms_m: float,
    lateral_peak_m: float,
    barrier_ev: float,
) -> float:
    """Return a dimensionless scalar selection score; lower is better."""
    values = np.asarray(
        [
            height_peak_m,
            height_rms_m,
            lateral_peak_m,
            barrier_ev,
        ],
        dtype=float,
    )

    if (
        not trace_valid
        or not path_valid
        or not np.all(np.isfinite(values))
    ):
        return 1e12

    return float(
        1.00 * (height_peak_m / 3e-6) ** 2
        + 0.35 * (height_rms_m / 1.5e-6) ** 2
        + 0.75 * (lateral_peak_m / 3e-6) ** 2
        + 0.65 * (max(barrier_ev, 0.0) / 0.100) ** 2
    )


def evaluate_candidate(
    *,
    island_id: int,
    generation: int,
    family: Family,
    genome: AnyGenome,
    show_bem_progress: bool = False,
) -> CandidateEvaluation:
    """Build, solve, trace and score one candidate.

    Invalid geometry, mesh failures, BEM failures, trace failures and physical
    validation failures are recorded as high-score candidates. The GA remains
    alive even if an individual candidate is unusable.
    """
    evaluation = CandidateEvaluation(
        island_id=island_id,
        generation=generation,
        family=family,
        genome=genome,
        valid=False,
        score=1e12,
    )

    try:
        parameters = candidate_geometry_parameters(genome)

        if isinstance(parameters, XJunctionParameters):
            manufacturability = check_x_junction_manufacturability(
                parameters
            )

            if not manufacturability.valid:
                evaluation.reason = "; ".join(
                    manufacturability.messages
                )
                return evaluation

        model = build_candidate_bem(parameters)
        evaluation.n_panels = int(model.n_panels)

        model.bem.assemble(show_progress=show_bem_progress)
        model.bem.solve(show_progress=False)

        trace = trace_rf_transverse_minimum(
            model.bem,
            X_VALUES_M,
            initial_y_m=0.0,
            initial_z_m=TARGET_ION_HEIGHT_M,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )
        metrics = compute_path_metrics(trace)

        energies_ev = pseudopotential_profile_ev(
            model.bem,
            trace,
            rf_voltage_peak_v=100.0,
            rf_angular_frequency_rad_s=(
                RF_ANGULAR_FREQUENCY_RAD_S
            ),
        )
        barrier = compute_barrier_metrics(
            energies_ev,
            trace,
        )
        validation = validate_rf_transport_path(metrics)

        evaluation.trace_valid = bool(trace.valid)
        evaluation.path_valid = bool(validation.valid)
        evaluation.height_peak_m = float(
            metrics.height_peak_deviation_m
        )
        evaluation.height_rms_m = float(
            metrics.height_rms_deviation_m
        )
        evaluation.lateral_peak_m = float(
            metrics.maximum_lateral_offset_m
        )
        evaluation.barrier_ev = float(
            barrier.barrier_height_ev
        )

        evaluation.score = score_metrics(
            trace_valid=evaluation.trace_valid,
            path_valid=evaluation.path_valid,
            height_peak_m=evaluation.height_peak_m,
            height_rms_m=evaluation.height_rms_m,
            lateral_peak_m=evaluation.lateral_peak_m,
            barrier_ev=evaluation.barrier_ev,
        )

        evaluation.valid = (
            np.isfinite(evaluation.score)
            and evaluation.score < 1e12
        )
        evaluation.reason = "; ".join(validation.messages)

    except Exception as error:
        evaluation.reason = (
            f"{type(error).__name__}: {error}"
        )

    return evaluation


def candidate_label(
    evaluation: CandidateEvaluation,
) -> str:
    """Return stable directory/file label for one evaluated candidate."""
    return (
        f"gen_{evaluation.generation:03d}"
        f"_island_{evaluation.island_id:02d}"
        f"_{evaluation.family}"
        f"_score_{evaluation.score:.4g}"
    ).replace(".", "p")


def save_layout_only(
    model: Any,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Save electrode layout even when RF tracing fails.

    Works for both regular-grid and quadtree BEM models. Every rectangular
    BEM panel is drawn as RF (orange) or grounded/DC plane (dark blue).
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    panels_m = np.asarray(model.bem.panels_m, dtype=float)
    voltages_v = np.asarray(
        model.bem.electrode_voltages_v,
        dtype=float,
    )

    patches = [
        Rectangle(
            (panel[0] * 1e6, panel[1] * 1e6),
            (panel[2] - panel[0]) * 1e6,
            (panel[3] - panel[1]) * 1e6,
        )
        for panel in panels_m
    ]

    colours = np.where(
        voltages_v > 0.5,
        1.0,
        0.0,
    )

    figure, axis = plt.subplots(figsize=(8.5, 8.0))

    collection = PatchCollection(
        patches,
        array=colours,
        cmap="coolwarm",
        edgecolor="none",
        alpha=0.90,
    )
    axis.add_collection(collection)

    extent_um = float(
        max(
            np.max(np.abs(panels_m[:, :2])),
            np.max(np.abs(panels_m[:, 2:])),
        )
        * 1e6
    )

    axis.set_xlim(-extent_um, extent_um)
    axis.set_ylim(-extent_um, extent_um)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)

    colourbar = figure.colorbar(
        collection,
        ax=axis,
        shrink=0.80,
        ticks=[0.0, 1.0],
    )
    colourbar.ax.set_yticklabels(["Ground / DC", "RF"])

    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_layout_and_trace(
    model: Any,
    trace: Any,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot the RF/ground panel layout and the traced RF minimum.

    Unlike the missing helper from ``junction_evaluator``, this function is
    local to workflow 10 and supports the quadtree panel models used by both
    ordinary and special families.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    panels_m = np.asarray(model.bem.panels_m, dtype=float)
    voltages_v = np.asarray(
        model.bem.electrode_voltages_v,
        dtype=float,
    )

    patches = [
        Rectangle(
            (panel[0] * 1e6, panel[1] * 1e6),
            (panel[2] - panel[0]) * 1e6,
            (panel[3] - panel[1]) * 1e6,
        )
        for panel in panels_m
    ]

    colours = np.where(
        voltages_v > 0.5,
        1.0,
        0.0,
    )

    figure, axis = plt.subplots(figsize=(8.8, 8.2))

    collection = PatchCollection(
        patches,
        array=colours,
        cmap="coolwarm",
        edgecolor="none",
        alpha=0.88,
    )
    axis.add_collection(collection)

    trace_x_m = np.asarray(trace.x_m, dtype=float)
    trace_y_m = np.asarray(trace.y_m, dtype=float)
    trace_z_m = np.asarray(trace.z_m, dtype=float)

    axis.plot(
        trace_x_m * 1e6,
        trace_y_m * 1e6,
        color="black",
        linewidth=1.7,
        marker="o",
        markersize=2.8,
        label="RF minimum projection",
        zorder=4,
    )

    valid = bool(getattr(trace, "valid", False))

    if valid:
        marker_colour = "lime"
        trace_label = "trace valid"
    else:
        marker_colour = "red"
        trace_label = "trace invalid"

    axis.scatter(
        trace_x_m[0] * 1e6,
        trace_y_m[0] * 1e6,
        s=55,
        color=marker_colour,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label=trace_label,
    )

    extent_um = float(
        max(
            np.max(np.abs(panels_m[:, :2])),
            np.max(np.abs(panels_m[:, 2:])),
        )
        * 1e6
    )

    axis.set_xlim(-extent_um, extent_um)
    axis.set_ylim(-extent_um, extent_um)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(
        f"{title}\n"
        f"trace valid={valid}; "
        f"z range={np.min(trace_z_m) * 1e6:.2f}..."
        f"{np.max(trace_z_m) * 1e6:.2f} um"
    )
    axis.grid(alpha=0.20)
    axis.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_path_diagnostics(
    trace: Any,
    metrics: Any,
    barrier: Any,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot height, lateral displacement and RF pseudopotential diagnostics.

    The plot is saved even for invalid paths. It makes failed smoke tests
    useful: clamped height, lateral excursion and broken trace become visible.
    """
    x_m = np.asarray(trace.x_m, dtype=float)
    y_m = np.asarray(trace.y_m, dtype=float)
    z_m = np.asarray(trace.z_m, dtype=float)

    x_um = x_m * 1e6
    y_um = y_m * 1e6
    z_um = z_m * 1e6

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 9.4),
        sharex=True,
    )

    axes[0].plot(
        x_um,
        z_um,
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
    )
    axes[0].axhline(
        TARGET_ION_HEIGHT_M * 1e6,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label="Target height",
    )
    axes[0].set_ylabel("z [um]")
    axes[0].grid(alpha=0.30)
    axes[0].legend(loc="best")

    axes[1].plot(
        x_um,
        y_um,
        color="tab:green",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
    )
    axes[1].axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set_ylabel("y [um]")
    axes[1].grid(alpha=0.30)

    barrier_height_ev = float(
        getattr(barrier, "barrier_height_ev", np.nan)
    )
    height_peak_m = float(
        getattr(metrics, "height_peak_deviation_m", np.nan)
    )
    lateral_peak_m = float(
        getattr(metrics, "maximum_lateral_offset_m", np.nan)
    )

    axes[2].axis("off")

    diagnostic_text = "\n".join(
        [
            f"trace valid: {bool(getattr(trace, 'valid', False))}",
            (
                "peak |dz|: "
                f"{height_peak_m * 1e6:.4f} um"
            ),
            (
                "max |y|: "
                f"{lateral_peak_m * 1e6:.4f} um"
            ),
            (
                "RF barrier: "
                f"{barrier_height_ev * 1e3:.5f} meV"
            ),
        ]
    )

    axes[2].text(
        0.03,
        0.85,
        diagnostic_text,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "whitesmoke",
            "edgecolor": "gray",
        },
    )

    axes[2].set_xlabel("x [um]")

    figure.suptitle(title)
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96),
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    

def candidate_report_payload(
    evaluation: CandidateEvaluation,
) -> dict[str, object]:
    """Create JSON-safe report data regardless of candidate validity."""
    if isinstance(
        evaluation.genome,
        CentralWindowCrossGenome,
    ):
        genome_summary: dict[str, object] = (
            evaluation.genome.summary()
        )
    else:
        genome_summary = unwrap_ordinary(
            evaluation.genome
        ).summary()

    return {
        "generation": evaluation.generation,
        "island_id": evaluation.island_id,
        "family": evaluation.family,
        "valid": evaluation.valid,
        "score": evaluation.score,
        "trace_valid": evaluation.trace_valid,
        "path_valid": evaluation.path_valid,
        "reason": evaluation.reason,
        "metrics": {
            "height_peak_m": evaluation.height_peak_m,
            "height_rms_m": evaluation.height_rms_m,
            "lateral_peak_m": evaluation.lateral_peak_m,
            "barrier_ev": evaluation.barrier_ev,
            "n_panels": evaluation.n_panels,
        },
        "genome": genome_summary,
    }


def write_candidate_report_json(
    evaluation: CandidateEvaluation,
    *,
    output_path: Path,
) -> None:
    """Write candidate metadata before its expensive diagnostic rerun."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            candidate_report_payload(evaluation),
            file,
            indent=2,
            default=float,
        )


def rerun_candidate_diagnostics(
    evaluation: CandidateEvaluation,
    *,
    output_directory: Path,
    show_bem_progress: bool,
) -> None:
    """Create diagnostic artifacts for valid or invalid candidates.

    The original GA evaluation only retains scalar metrics. This rerun
    recreates model, solve and trace so the report includes layout and
    physical path diagnostics.

    It intentionally writes a report even when:
    - geometry conversion fails;
    - manufacturability rejects the candidate;
    - BEM assembly/solve fails;
    - RF trace is invalid;
    - path validation rejects the transport path.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_candidate_report_json(
        evaluation,
        output_path=output_directory / "evaluation.json",
    )

    title = (
        f"generation={evaluation.generation}, "
        f"island={evaluation.island_id}, "
        f"family={evaluation.family}, "
        f"score={evaluation.score:.5g}"
    )

    try:
        parameters = candidate_geometry_parameters(
            evaluation.genome
        )

        if isinstance(parameters, XJunctionParameters):
            manufacturability = check_x_junction_manufacturability(
                parameters
            )

            manufacturability_payload = {
                "valid": bool(manufacturability.valid),
                "messages": list(manufacturability.messages),
            }

            with (
                output_directory
                / "manufacturability.json"
            ).open("w", encoding="utf-8") as file:
                json.dump(
                    manufacturability_payload,
                    file,
                    indent=2,
                )

        model = build_candidate_bem(parameters)

        save_layout_only(
            model,
            output_path=output_directory / "layout_only.png",
            title=title,
        )

        model.bem.assemble(
            show_progress=show_bem_progress,
        )
        model.bem.solve(show_progress=False)

        trace = trace_rf_transverse_minimum(
            model.bem,
            X_VALUES_M,
            initial_y_m=0.0,
            initial_z_m=TARGET_ION_HEIGHT_M,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )

        metrics = compute_path_metrics(trace)

        energies_ev = pseudopotential_profile_ev(
            model.bem,
            trace,
            rf_voltage_peak_v=100.0,
            rf_angular_frequency_rad_s=(
                RF_ANGULAR_FREQUENCY_RAD_S
            ),
        )
        barrier = compute_barrier_metrics(
            energies_ev,
            trace,
        )
        validation = validate_rf_transport_path(metrics)

        rerun_payload = {
            "trace_valid": bool(trace.valid),
            "path_valid": bool(validation.valid),
            "validation_messages": list(validation.messages),
            "height_peak_m": float(
                metrics.height_peak_deviation_m
            ),
            "height_rms_m": float(
                metrics.height_rms_deviation_m
            ),
            "lateral_peak_m": float(
                metrics.maximum_lateral_offset_m
            ),
            "barrier_ev": float(
                barrier.barrier_height_ev
            ),
            "n_panels": int(model.n_panels),
        }

        with (
            output_directory
            / "rerun_metrics.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                rerun_payload,
                file,
                indent=2,
                default=float,
            )

        # These two plots are useful even when trace/path validity is False:
        # they show where the trace travelled and which metric invalidated it.
        plot_layout_and_trace(
            model,
            trace,
            output_path=(
                output_directory
                / "layout_and_trace.png"
            ),
            title=title,
        )

        plot_path_diagnostics(
            trace,
            metrics,
            barrier,
            output_path=(
                output_directory
                / "path_diagnostics.png"
            ),
            title=title,
        )

        if not trace.valid or not validation.valid:
            with (
                output_directory
                / "failure.txt"
            ).open("w", encoding="utf-8") as file:
                file.write(
                    "The candidate was rerun, but did not pass "
                    "the trace/path validation.\n\n"
                )
                file.write(
                    "trace_valid = "
                    f"{bool(trace.valid)}\n"
                )
                file.write(
                    "path_valid = "
                    f"{bool(validation.valid)}\n\n"
                )
                file.write("Validation messages:\n")
                for message in validation.messages:
                    file.write(f"- {message}\n")

    except Exception as error:
        with (
            output_directory
            / "failure.txt"
        ).open("w", encoding="utf-8") as file:
            file.write(
                "Diagnostic rerun failed before a complete "
                "path report could be created.\n\n"
            )
            file.write(
                f"{type(error).__name__}: {error}\n"
            )


def choose_report_candidates(
    evaluations: list[CandidateEvaluation],
    *,
    top_per_family: int,
) -> list[CandidateEvaluation]:
    """Choose report candidates even if no candidate is valid.

    Priority is:
    1. Valid candidates by score.
    2. Invalid candidates by score.
    3. Candidates with lowest finite score if all are invalid.

    Therefore each family receives diagnostic output whenever it was
    evaluated, including a run with zero valid transport solutions.
    """
    selected: list[CandidateEvaluation] = []

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        family_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.family == family
        ]

        ranked = sorted(
            family_evaluations,
            key=lambda item: (
                not item.valid,
                item.score,
                item.generation,
                item.island_id,
            ),
        )

        selected.extend(ranked[:top_per_family])

    return selected


def write_failure_summary(
    evaluations: list[CandidateEvaluation],
    *,
    output_path: Path,
) -> None:
    """Write a readable failure report, including zero-valid runs."""
    counts_by_family: dict[str, Counter[str]] = {
        "ordinary": Counter(),
        "central_window_cross": Counter(),
    }

    for evaluation in evaluations:
        reason = (
            evaluation.reason.strip()
            if evaluation.reason.strip()
            else "no_reason_recorded"
        )
        counts_by_family[evaluation.family][reason] += 1

    lines = [
        "# Step 10 mixed-island GA: validity and failure summary",
        "",
        f"Total physical evaluations: {len(evaluations)}",
        "",
    ]

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        family_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.family == family
        ]
        valid_count = sum(
            evaluation.valid
            for evaluation in family_evaluations
        )

        lines.extend(
            [
                f"## {family}",
                "",
                f"- Evaluated: {len(family_evaluations)}",
                f"- Valid: {valid_count}",
                f"- Invalid: {len(family_evaluations) - valid_count}",
                "",
                "### Reasons",
                "",
            ]
        )

        if not counts_by_family[family]:
            lines.append("- No candidates were evaluated.")
        else:
            for reason, count in (
                counts_by_family[family]
                .most_common()
            ):
                lines.append(f"- {count}: {reason}")

        lines.append("")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_detailed_reports(
    evaluations: list[CandidateEvaluation],
    *,
    output_directory: Path,
    top_per_family: int,
    show_bem_progress: bool,
) -> None:
    """Generate detailed candidate reports whether or not GA found validity."""
    reports_directory = output_directory / "candidate_reports"
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_failure_summary(
        evaluations,
        output_path=output_directory / "failure_summary.md",
    )

    selected = choose_report_candidates(
        evaluations,
        top_per_family=top_per_family,
    )

    if not selected:
        with (
            output_directory
            / "failure_summary.md"
        ).open("a", encoding="utf-8") as file:
            file.write(
                "\nNo physical candidates were evaluated, "
                "so no candidate diagnostic folders exist.\n"
            )
        return

    for rank, evaluation in enumerate(selected, start=1):
        candidate_directory = (
            reports_directory
            / f"{rank:02d}_{candidate_label(evaluation)}"
        )

        print(
            "Detailed report rerun: "
            f"{candidate_directory.name}"
        )

        rerun_candidate_diagnostics(
            evaluation,
            output_directory=candidate_directory,
            show_bem_progress=show_bem_progress,
        )


def animation_frame_rows(
    evaluations: list[CandidateEvaluation],
    generation: int,
) -> list[CandidateEvaluation]:
    """Return all candidates evaluated in one generation."""
    return [
        evaluation
        for evaluation in evaluations
        if evaluation.generation == generation
    ]


def safe_plot_value(
    value: float,
    *,
    fallback: float,
) -> float:
    """Replace NaN/inf by a finite plotting fallback."""
    return float(value) if np.isfinite(value) else fallback


def create_evolution_animation(
    evaluations: list[CandidateEvaluation],
    *,
    output_path: Path,
) -> None:
    """Create a GIF animation for valid and invalid GA populations.

    Panels:
    - left: peak height deviation versus RF barrier;
    - right: scalar selection score versus island index.

    Ordinary candidates use blue; special candidates use orange.
    Valid candidates use circles; invalid candidates use x markers.
    This gives useful diagnostics even when valid-count is zero.
    """
    if not evaluations:
        return

    generations = sorted(
        {
            evaluation.generation
            for evaluation in evaluations
        }
    )

    if not generations:
        return

    finite_heights_um = [
        evaluation.height_peak_m * 1e6
        for evaluation in evaluations
        if np.isfinite(evaluation.height_peak_m)
    ]
    finite_barriers_mev = [
        evaluation.barrier_ev * 1e3
        for evaluation in evaluations
        if np.isfinite(evaluation.barrier_ev)
    ]
    finite_scores = [
        evaluation.score
        for evaluation in evaluations
        if np.isfinite(evaluation.score)
        and evaluation.score < 1e12
    ]

    height_fallback_um = (
        max(finite_heights_um) * 1.10
        if finite_heights_um
        else 50.0
    )
    barrier_fallback_mev = (
        max(finite_barriers_mev) * 1.10
        if finite_barriers_mev
        else 250.0
    )
    score_fallback = (
        max(finite_scores) * 10.0
        if finite_scores
        else 1e12
    )

    height_limit_um = max(
        10.0,
        height_fallback_um * 1.15,
    )
    barrier_limit_mev = max(
        10.0,
        barrier_fallback_mev * 1.15,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.7),
    )
    metric_axis, score_axis = axes

    family_colours = {
        "ordinary": "tab:blue",
        "central_window_cross": "tab:orange",
    }

    def draw_frame(frame_index: int) -> None:
        generation = generations[frame_index]
        rows = animation_frame_rows(
            evaluations,
            generation,
        )

        metric_axis.clear()
        score_axis.clear()

        for family in (
            "ordinary",
            "central_window_cross",
        ):
            family_rows = [
                row
                for row in rows
                if row.family == family
            ]

            valid_rows = [
                row
                for row in family_rows
                if row.valid
            ]
            invalid_rows = [
                row
                for row in family_rows
                if not row.valid
            ]

            colour = family_colours[family]

            if valid_rows:
                metric_axis.scatter(
                    [
                        safe_plot_value(
                            row.height_peak_m * 1e6,
                            fallback=height_fallback_um,
                        )
                        for row in valid_rows
                    ],
                    [
                        safe_plot_value(
                            row.barrier_ev * 1e3,
                            fallback=barrier_fallback_mev,
                        )
                        for row in valid_rows
                    ],
                    s=70,
                    marker="o",
                    color=colour,
                    alpha=0.85,
                    label=f"{family}: valid",
                )

            if invalid_rows:
                metric_axis.scatter(
                    [
                        safe_plot_value(
                            row.height_peak_m * 1e6,
                            fallback=height_fallback_um,
                        )
                        for row in invalid_rows
                    ],
                    [
                        safe_plot_value(
                            row.barrier_ev * 1e3,
                            fallback=barrier_fallback_mev,
                        )
                        for row in invalid_rows
                    ],
                    s=78,
                    marker="x",
                    color=colour,
                    alpha=0.80,
                    label=f"{family}: invalid",
                )

            if valid_rows:
                score_axis.scatter(
                    [
                        row.island_id
                        for row in valid_rows
                    ],
                    [
                        safe_plot_value(
                            row.score,
                            fallback=score_fallback,
                        )
                        for row in valid_rows
                    ],
                    s=70,
                    marker="o",
                    color=colour,
                    alpha=0.82,
                    label=f"{family}: valid",
                )

            if invalid_rows:
                score_axis.scatter(
                    [
                        row.island_id
                        for row in invalid_rows
                    ],
                    [
                        safe_plot_value(
                            row.score,
                            fallback=score_fallback,
                        )
                        for row in invalid_rows
                    ],
                    s=78,
                    marker="x",
                    color=colour,
                    alpha=0.82,
                    label=f"{family}: invalid",
                )

        metric_axis.axvline(
            3.0,
            color="tab:red",
            linestyle=":",
            linewidth=1.2,
            label="height target 3 um",
        )
        metric_axis.set_xlim(
            -0.5,
            height_limit_um,
        )
        metric_axis.set_ylim(
            -0.5,
            barrier_limit_mev,
        )
        metric_axis.set_xlabel("Peak height deviation [um]")
        metric_axis.set_ylabel("RF barrier [meV]")
        metric_axis.set_title(
            "Physical metrics: circle = valid, x = invalid"
        )
        metric_axis.grid(alpha=0.28)
        metric_axis.legend(
            loc="best",
            fontsize=8,
        )

        score_axis.set_yscale("log")
        score_axis.set_xlabel("Island ID")
        score_axis.set_ylabel("Selection score, lower is better")
        score_axis.set_title(
            "Island population score distribution"
        )
        score_axis.grid(alpha=0.28)
        score_axis.legend(
            loc="best",
            fontsize=8,
        )

        valid_count = sum(
            row.valid
            for row in rows
        )

        figure.suptitle(
            "Step 10 mixed-topology island GA | "
            f"generation {generation} | "
            f"valid {valid_count}/{len(rows)}"
        )
        figure.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.93),
        )

    movie = animation.FuncAnimation(
        figure,
        draw_frame,
        frames=len(generations),
        interval=900,
        repeat=True,
    )

    try:
        writer = animation.PillowWriter(fps=1)
        movie.save(
            output_path,
            writer=writer,
            dpi=130,
        )
    except Exception as error:
        fallback_path = output_path.with_suffix(".png")
        draw_frame(len(generations) - 1)
        figure.savefig(
            fallback_path,
            dpi=180,
            bbox_inches="tight",
        )
        print(
            "GIF animation could not be saved; "
            f"saved final animation frame instead: {fallback_path}. "
            f"Reason: {type(error).__name__}: {error}"
        )
    finally:
        plt.close(figure)

# =============================================================================
# Selection, crossover and mutation
# =============================================================================

def tournament_select(
    evaluations: list[CandidateEvaluation],
    rng: np.random.Generator,
) -> AnyGenome:
    """Select a parent through tournament selection."""
    if not evaluations:
        raise ValueError("Cannot select from an empty island.")

    count = min(TOURNAMENT_SIZE, len(evaluations))

    indices = rng.choice(
        len(evaluations),
        size=count,
        replace=False,
    )

    winner = min(
        (evaluations[int(index)] for index in indices),
        key=lambda item: item.score,
    )

    return clone_genome(winner.genome)


def make_child(
    *,
    family: Family,
    parent_a: AnyGenome,
    parent_b: AnyGenome,
    rng: np.random.Generator,
    mutation_scale: float,
    mutation_probability: float,
) -> AnyGenome:
    """Create a same-family child after parent selection."""
    if family == "central_window_cross":
        if not isinstance(parent_a, CentralWindowCrossGenome):
            raise TypeError("Special parent A has incorrect type.")
        if not isinstance(parent_b, CentralWindowCrossGenome):
            raise TypeError("Special parent B has incorrect type.")

        child = crossover_central_window_cross(
            parent_a,
            parent_b,
            rng,
        )

        return mutate_central_window_cross(
            child,
            rng,
            mutation_probability=mutation_probability,
            scale=mutation_scale,
        )

    ordinary_a = unwrap_ordinary(parent_a)
    ordinary_b = unwrap_ordinary(parent_b)

    child = crossover_ordinary_template(
        ordinary_a,
        ordinary_b,
        rng,
    )
    child = mutate_ordinary_template(
        child,
        rng,
        mutation_probability=mutation_probability,
        scale=mutation_scale,
    )

    return make_ordinary_wrapper(child)


def next_population(
    *,
    island: Island,
    evaluations: list[CandidateEvaluation],
    rng: np.random.Generator,
    mutation_scale: float,
) -> list[AnyGenome]:
    """Build one next-generation population with local elitism."""
    ranked = sorted(
        evaluations,
        key=lambda item: item.score,
    )

    if not ranked:
        raise ValueError(
            f"Island {island.island_id} has no evaluations."
        )

    next_genomes: list[AnyGenome] = [
        clone_genome(item.genome)
        for item in ranked[:ELITE_COUNT]
    ]

    effective_mutation_scale = mutation_scale
    effective_mutation_probability = 0.45

    if island.in_burst():
        effective_mutation_scale *= BURST_MUTATION_MULTIPLIER
        effective_mutation_probability = (
            BURST_MUTATION_PROBABILITY
        )

    while len(next_genomes) < POPULATION_PER_ISLAND:
        parent_a = tournament_select(ranked, rng)
        parent_b = tournament_select(ranked, rng)

        child = make_child(
            family=island.family,
            parent_a=parent_a,
            parent_b=parent_b,
            rng=rng,
            mutation_scale=effective_mutation_scale,
            mutation_probability=effective_mutation_probability,
        )

        next_genomes.append(child)

    return next_genomes


# =============================================================================
# Stagnation and burst logic
# =============================================================================

def update_island_stagnation(
    island: Island,
    evaluations: list[CandidateEvaluation],
) -> bool:
    """Update island progress and return True if a new burst starts."""
    if not evaluations:
        island.stagnant_generations += 1
        return False

    best_score = min(item.score for item in evaluations)

    if not np.isfinite(best_score):
        island.stagnant_generations += 1

    elif not np.isfinite(island.best_score_seen):
        island.best_score_seen = best_score
        island.stagnant_generations = 0

    else:
        improvement_threshold = island.best_score_seen * (
            1.0 - RELATIVE_IMPROVEMENT_TOLERANCE
        )

        if best_score < improvement_threshold:
            island.best_score_seen = best_score
            island.stagnant_generations = 0
        else:
            island.stagnant_generations += 1

    if (
        island.stagnant_generations >= STAGNATION_GENERATIONS
        and island.burst_generations_remaining == 0
    ):
        island.burst_generations_remaining = BURST_GENERATIONS
        island.stagnant_generations = 0
        island.burst_count += 1
        return True

    return False


def decrement_burst_counters(
    islands: list[Island],
) -> None:
    """Decrease the duration of all active bursts."""
    for island in islands:
        if island.burst_generations_remaining > 0:
            island.burst_generations_remaining -= 1


def burst_reseed_island(
    *,
    island: Island,
    all_generation_results: dict[int, list[CandidateEvaluation]],
    rng: np.random.Generator,
) -> None:
    """Inject translated opposite-family elites and random immigrants.

    The tail of the already-created next generation is replaced. Local elites
    are retained at the beginning of the population by ``next_population``.

    Composition:
        - 75% translated opposite-family seeds, strongly mutated;
        - 25% fresh broad random genomes.
    """
    replacement_count = max(
        1,
        int(
            math.ceil(
                BURST_REPLACE_FRACTION
                * POPULATION_PER_ISLAND
            )
        ),
    )

    random_count = max(
        1,
        int(
            round(
                BURST_RANDOM_FRACTION
                * replacement_count
            )
        ),
    )

    translated_count = replacement_count - random_count

    opposite_family: Family = (
        "ordinary"
        if island.family == "central_window_cross"
        else "central_window_cross"
    )

    opposite_candidates = sorted(
        [
            evaluation
            for results in all_generation_results.values()
            for evaluation in results
            if evaluation.family == opposite_family
        ],
        key=lambda item: item.score,
    )

    replacements: list[AnyGenome] = []

    for index in range(translated_count):
        if opposite_candidates:
            source = opposite_candidates[
                index % len(opposite_candidates)
            ].genome

            translated = translate_to_family(
                source,
                destination_family=island.family,
                rng=rng,
            )

            if island.family == "central_window_cross":
                if not isinstance(
                    translated,
                    CentralWindowCrossGenome,
                ):
                    raise TypeError(
                        "Translated special genome has wrong type."
                    )

                translated = mutate_central_window_cross(
                    translated,
                    rng,
                    mutation_probability=(
                        BURST_MUTATION_PROBABILITY
                    ),
                    scale=BURST_MUTATION_MULTIPLIER,
                )

            else:
                ordinary = unwrap_ordinary(translated)

                translated = make_ordinary_wrapper(
                    mutate_ordinary_template(
                        ordinary,
                        rng,
                        mutation_probability=(
                            BURST_MUTATION_PROBABILITY
                        ),
                        scale=BURST_MUTATION_MULTIPLIER,
                    )
                )

            replacements.append(translated)

        else:
            replacements.append(
                random_genome_for_family(
                    island.family,
                    rng,
                    broad=True,
                )
            )

    for _ in range(random_count):
        replacements.append(
            random_genome_for_family(
                island.family,
                rng,
                broad=True,
            )
        )

    replacements = replacements[:replacement_count]

    island.population[-replacement_count:] = replacements


# =============================================================================
# Cross-family migration
# =============================================================================

def migrate_across_all_islands(
    islands: list[Island],
    island_results: dict[int, list[CandidateEvaluation]],
    rng: np.random.Generator,
) -> None:
    """Ring-migrate elites through every island, including across families.

    Example:

        ordinary_0 -> ordinary_1 -> ... -> special_0 -> special_1
        -> ... -> ordinary_0

    At every ordinary/special boundary the migrant is translated into the
    geometry encoding of the destination family. Therefore the two branches
    truly exchange physically meaningful seeds.
    """
    if len(islands) < 2:
        return

    outgoing: list[list[AnyGenome]] = []

    for island in islands:
        ranked = sorted(
            island_results[island.island_id],
            key=lambda item: item.score,
        )

        outgoing.append(
            [
                clone_genome(item.genome)
                for item in ranked[:MIGRANTS_PER_ISLAND]
            ]
        )

    for destination_index, destination_island in enumerate(islands):
        source_index = (destination_index - 1) % len(islands)
        migrants = outgoing[source_index]

        translated_migrants = [
            translate_to_family(
                genome,
                destination_family=destination_island.family,
                rng=rng,
            )
            for genome in migrants
        ]

        destination_island.population[
            -MIGRANTS_PER_ISLAND:
        ] = translated_migrants


# =============================================================================
# Output
# =============================================================================

def write_csv(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Write all evaluation rows with union of family-specific columns."""
    if not rows:
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_progress(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot the best family score at each generation."""
    if not rows:
        return

    figure, axis = plt.subplots(
        figsize=(10.0, 5.8),
    )

    styles = {
        "ordinary": {
            "color": "tab:blue",
            "label": "ordinary connected-cross",
        },
        "central_window_cross": {
            "color": "tab:orange",
            "label": "special central-window cross",
        },
    }

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        family_rows = [
            row
            for row in rows
            if row["family"] == family
        ]

        generations = sorted(
            {
                int(row["generation"])
                for row in family_rows
            }
        )

        if not generations:
            continue

        best_scores = [
            min(
                float(row["score"])
                for row in family_rows
                if int(row["generation"]) == generation
            )
            for generation in generations
        ]

        axis.plot(
            generations,
            best_scores,
            marker="o",
            linewidth=1.8,
            **styles[family],
        )

    axis.set_yscale("log")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best scalar score, lower is better")
    axis.set_title(
        "Step 10: ordinary/special mixed island-GA progress"
    )
    axis.grid(alpha=0.30)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_best_genomes(
    evaluations: list[CandidateEvaluation],
    *,
    output_path: Path,
) -> None:
    """Save best valid solution of each family to JSON."""
    payload: dict[str, Any] = {}

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        candidates = [
            evaluation
            for evaluation in evaluations
            if evaluation.family == family
            and evaluation.valid
        ]

        if not candidates:
            payload[family] = {
                "status": "no_valid_candidate",
            }
            continue

        best = min(
            candidates,
            key=lambda item: item.score,
        )

        if isinstance(
            best.genome,
            CentralWindowCrossGenome,
        ):
            genome_summary = best.genome.summary()
        else:
            genome_summary = unwrap_ordinary(
                best.genome
            ).summary()

        payload[family] = {
            "status": "ok",
            "generation": best.generation,
            "island_id": best.island_id,
            "score": best.score,
            "metrics": {
                "height_peak_m": best.height_peak_m,
                "height_rms_m": best.height_rms_m,
                "lateral_peak_m": best.lateral_peak_m,
                "barrier_ev": best.barrier_ev,
                "n_panels": best.n_panels,
            },
            "genome": genome_summary,
        }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=float,
        )


def print_generation_summary(
    *,
    generation: int,
    islands: list[Island],
    evaluations: list[CandidateEvaluation],
    evaluation_count: int,
) -> None:
    """Print compact physical results after a completed generation."""
    print()
    print(
        f"Generation {generation + 1}/{GENERATIONS}; "
        f"evaluations {evaluation_count}/{MAX_BEM_EVALUATIONS}"
    )
    print("-" * 126)

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        results = [
            item
            for item in evaluations
            if item.family == family
        ]
        valid = [
            item
            for item in results
            if item.valid
        ]

        family_islands = [
            island
            for island in islands
            if island.family == family
        ]
        active_bursts = sum(
            island.in_burst()
            for island in family_islands
        )

        if not results:
            continue

        best = min(
            results,
            key=lambda item: item.score,
        )

        if valid:
            print(
                f"{family:>25}: "
                f"valid={len(valid):2d}/{len(results):2d}; "
                f"best={best.score:10.5g}; "
                f"dz={best.height_peak_m * 1e6:8.3f} um; "
                f"|y|={best.lateral_peak_m * 1e6:8.3f} um; "
                f"barrier={best.barrier_ev * 1e3:9.4f} meV; "
                f"burst islands={active_bursts}"
            )
        else:
            print(
                f"{family:>25}: "
                f"valid= 0/{len(results):2d}; "
                f"best failure={best.reason}; "
                f"burst islands={active_bursts}"
            )


# =============================================================================
# Initialization and main loop
# =============================================================================

def initialise_islands(
    rng: np.random.Generator,
) -> list[Island]:
    """Create the configured 70/30 mixed topology island population."""
    islands: list[Island] = []
    island_id = 0

    for _ in range(N_ORDINARY_ISLANDS):
        population = [
            random_genome_for_family(
                "ordinary",
                rng,
                broad=False,
            )
            for _ in range(POPULATION_PER_ISLAND)
        ]

        islands.append(
            Island(
                island_id=island_id,
                family="ordinary",
                population=population,
            )
        )
        island_id += 1

    for _ in range(N_SPECIAL_ISLANDS):
        population = [
            random_genome_for_family(
                "central_window_cross",
                rng,
                broad=False,
            )
            for _ in range(POPULATION_PER_ISLAND)
        ]

        islands.append(
            Island(
                island_id=island_id,
                family="central_window_cross",
                population=population,
            )
        )
        island_id += 1

    return islands


def mutation_scale_for_generation(
    generation: int,
) -> float:
    """Linearly reduce baseline mutation through the run."""
    if GENERATIONS <= 1:
        return FINAL_MUTATION_SCALE

    fraction = generation / (GENERATIONS - 1)

    return (
        INITIAL_MUTATION_SCALE
        + fraction
        * (
            FINAL_MUTATION_SCALE
            - INITIAL_MUTATION_SCALE
        )
    )

def parse_arguments() -> argparse.Namespace:
    """Parse runtime budget without editing source constants."""
    parser = argparse.ArgumentParser(
        description=(
            "Mixed-topology island GA for planar RF X-junctions."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a minimal PowerShell smoke test: "
            "1 ordinary island, 1 special island, "
            "2 candidates/island, 2 generations."
        ),
    )
    parser.add_argument(
        "--ordinary-islands",
        type=int,
        default=N_ORDINARY_ISLANDS,
        help="Number of conventional connected-cross islands.",
    )
    parser.add_argument(
        "--special-islands",
        type=int,
        default=N_SPECIAL_ISLANDS,
        help="Number of central-window-cross islands.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=POPULATION_PER_ISLAND,
        help="Population size per island.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GENERATIONS,
        help="Maximum number of GA generations.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=MAX_BEM_EVALUATIONS,
        help="Hard cap on real physical BEM evaluations.",
    )
    parser.add_argument(
        "--show-bem-progress",
        action="store_true",
        help="Show BEM matrix assembly progress for each candidate.",
    )

    parser.add_argument(
        "--report-top",
        type=int,
        default=5,
        help=(
            "Number of best candidates per family to rerun "
            "with detailed layout and physical diagnostics."
        ),
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Do not create the evolution animation.",
    )

    arguments = parser.parse_args()

    if arguments.smoke:
        arguments.ordinary_islands = 1
        arguments.special_islands = 1
        arguments.population = 2
        arguments.generations = 2
        arguments.max_evaluations = 8

    for name in (
        "ordinary_islands",
        "special_islands",
        "population",
        "generations",
        "max_evaluations",
    ):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    return arguments

def main(arguments: argparse.Namespace) -> None:
    """Run mixed-family island GA with cross-family migration and bursts."""
    
    global N_ORDINARY_ISLANDS
    global N_SPECIAL_ISLANDS
    global POPULATION_PER_ISLAND
    global GENERATIONS
    global MAX_BEM_EVALUATIONS

    N_ORDINARY_ISLANDS = arguments.ordinary_islands
    N_SPECIAL_ISLANDS = arguments.special_islands
    POPULATION_PER_ISLAND = arguments.population
    GENERATIONS = arguments.generations
    MAX_BEM_EVALUATIONS = arguments.max_evaluations
    
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(SEED)
    islands = initialise_islands(rng)

    planned_evaluations = (
        len(islands)
        * POPULATION_PER_ISLAND
        * GENERATIONS
    )

    print("Step 10: mixed-topology island genetic algorithm")
    print("=" * 126)
    print(
        f"Ordinary islands: {N_ORDINARY_ISLANDS}; "
        f"special islands: {N_SPECIAL_ISLANDS}; "
        f"total islands: {len(islands)}."
    )
    print(
        f"Population per island: {POPULATION_PER_ISLAND}; "
        f"generations: {GENERATIONS}; "
        f"planned BEM evaluations: {planned_evaluations}; "
        f"hard cap: {MAX_BEM_EVALUATIONS}."
    )
    print(
        f"Routine all-island migration every "
        f"{MIGRATION_INTERVAL} generations; "
        f"burst after {STAGNATION_GENERATIONS} stagnant generations."
    )

    all_evaluations: list[CandidateEvaluation] = []
    csv_rows: list[dict[str, object]] = []

    evaluation_count = 0
    stopped_early = False

    for generation in range(GENERATIONS):
        generation_results: dict[
            int,
            list[CandidateEvaluation],
        ] = {}

        for island in islands:
            island_results: list[
                CandidateEvaluation
            ] = []

            for genome in island.population:
                if evaluation_count >= MAX_BEM_EVALUATIONS:
                    stopped_early = True
                    break

                result = evaluate_candidate(
                    island_id=island.island_id,
                    generation=generation,
                    family=island.family,
                    genome=genome,
                    show_bem_progress=arguments.show_bem_progress,
                )

                island_results.append(result)
                all_evaluations.append(result)
                csv_rows.append(result.to_row())

                evaluation_count += 1

            generation_results[
                island.island_id
            ] = island_results

            if stopped_early:
                break

        if stopped_early:
            print()
            print(
                "Stopped at MAX_BEM_EVALUATIONS. "
                "Saving partial results."
            )
            break

        flattened_results = [
            result
            for island_result in generation_results.values()
            for result in island_result
        ]

        print_generation_summary(
            generation=generation,
            islands=islands,
            evaluations=flattened_results,
            evaluation_count=evaluation_count,
        )

        if generation + 1 >= GENERATIONS:
            break

        started_bursts: list[int] = []

        for island in islands:
            if update_island_stagnation(
                island,
                generation_results[island.island_id],
            ):
                started_bursts.append(island.island_id)

        if started_bursts:
            print(
                "Burst started on island(s): "
                + ", ".join(
                    str(island_id)
                    for island_id in started_bursts
                )
            )

        baseline_mutation_scale = (
            mutation_scale_for_generation(generation)
        )

        # Normal reproduction. An island already in burst receives a higher
        # mutation scale and higher mutation probability in next_population().
        for island in islands:
            island.population = next_population(
                island=island,
                evaluations=generation_results[
                    island.island_id
                ],
                rng=rng,
                mutation_scale=baseline_mutation_scale,
            )

        # Routine inter-island migration. The ring includes ordinary and
        # special islands, so migrants cross the family boundary and are
        # translated at that boundary.
        if (generation + 1) % MIGRATION_INTERVAL == 0:
            migrate_across_all_islands(
                islands,
                generation_results,
                rng,
            )
            print(
                "Routine cross-family ring migration applied."
            )

        # Burst reseeding happens after ordinary reproduction/migration.
        # It overwrites weak tail candidates but leaves elites intact.
        for island in islands:
            if island.in_burst():
                burst_reseed_island(
                    island=island,
                    all_generation_results=generation_results,
                    rng=rng,
                )

        decrement_burst_counters(islands)

    write_csv(
        csv_rows,
        output_path=(
            OUTPUT_DIRECTORY
            / "mixed_island_evaluations.csv"
        ),
    )

    plot_progress(
        csv_rows,
        output_path=(
            OUTPUT_DIRECTORY
            / "mixed_island_progress.png"
        ),
    )

    save_best_genomes(
        all_evaluations,
        output_path=(
            OUTPUT_DIRECTORY
            / "best_genomes.json"
        ),
    )

    save_detailed_reports(
        all_evaluations,
        output_directory=OUTPUT_DIRECTORY,
        top_per_family=arguments.report_top,
        show_bem_progress=arguments.show_bem_progress,
    )

    if not arguments.no_animation:
        create_evolution_animation(
            all_evaluations,
            output_path=(
                OUTPUT_DIRECTORY
                / "mixed_island_evolution.gif"
            ),
        )
    valid_evaluations = [
        evaluation
        for evaluation in all_evaluations
        if evaluation.valid
    ]

    print()
    print("=" * 126)
    print(
        f"Finished after {evaluation_count} physical BEM evaluations. "
        f"Valid candidates: {len(valid_evaluations)}."
    )
    print(f"Results directory: {OUTPUT_DIRECTORY}")

    if not valid_evaluations:
        print(
            "No valid transport candidate was found. "
            "Read mixed_island_evaluations.csv, then tighten initial ranges "
            "or relax only the constraint that rejects candidates."
        )
        return

    for family in (
        "ordinary",
        "central_window_cross",
    ):
        family_valid = [
            evaluation
            for evaluation in valid_evaluations
            if evaluation.family == family
        ]

        if not family_valid:
            print(f"{family}: no valid candidate.")
            continue

        best = min(
            family_valid,
            key=lambda item: item.score,
        )

        print(
            f"{family}: best score={best.score:.6g}; "
            f"peak dz={best.height_peak_m * 1e6:.4f} um; "
            f"max |y|={best.lateral_peak_m * 1e6:.4f} um; "
            f"barrier={best.barrier_ev * 1e3:.5f} meV; "
            f"generation={best.generation}; "
            f"island={best.island_id}."
        )


if __name__ == "__main__":
    main(parse_arguments())