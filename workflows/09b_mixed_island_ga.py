"""Workflow 09b: 70/30 mixed-island movable-knot C4v connected-cross GA.

This is an improved, directly compatible variant of workflow 09.

Geometry:
    - All candidates retain one connected conventional C4v RF cross.
    - Every arm is generated from the same movable-knot inner/outer contour.
    - No disconnected RF islands are introduced.
    - 70% ordinary islands search around conventional connected-cross seeds.
    - 30% special-prior islands start from and repeatedly inject local
      C4v-symmetric RF boundary protrusions/indentations.

Special-prior does NOT mean a new incompatible topology. It means deliberate
search in the same physically connected representation, using localized knot
features on the inner and/or outer boundaries. This makes ordinary <-> special
migration and crossover physically meaningful without a genome translator.

Outputs:
    history.csv
    rejection_summary.csv
    island_elites/elites_gen_XXXX.csv
    population_gen_XXXX.csv
    surface_animations/island_XX_top5_surfaces.gif
    fine/candidate_XXX/layout.png
    fine/candidate_XXX/pseudopotential_surface.png
    fine/candidate_XXX/profiles.png
    top5_lithography_candidates.json
    top5_lithography_candidates.csv

Smoke test from PowerShell:
    python workflows/09b_mixed_island_ga.py --smoke --fast

Production-style first run:
    python workflows/09b_mixed_island_ga.py `
        --ordinary-islands 21 `
        --special-islands 9 `
        --population 20 `
        --offspring 8 `
        --generations 100 `
        --backend auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config.targets import (
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
)
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.rf_null_trace import (
    trace_rf_transverse_minimum,
)
from core.ga.fixed_bem import (
    FixedMeshBEM,
    available_backend,
)
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.geometry.movable_knot_xjunction import (
    MovableKnotContour,
    build_movable_knot_parameters,
    fixed_knots_to_logits,
    softmax_gap_knots,
)


# =============================================================================
# Genome layout: same proven representation as workflow 09
# =============================================================================

N_INTERVALS = 8
N_KNOTS = N_INTERVALS + 1

MIN_GAP_M = 3.0e-6
LOCK_MIN_M = 100e-6
LOCK_MAX_M = 180e-6

FIXED_CORE_KNOTS_M = (
    np.array(
        [0, 7, 15, 27, 43, 65, 92, 122, 150],
        dtype=float,
    )
    * 1e-6
)

# ---------------------------------------------------------------------------
# Constrained special family from workflows 07c--07g.
#
# It is still the same connected conventional C4v cross. Only connected
# inner/outer RF boundaries around s=(60,90,120) um vary.
# ---------------------------------------------------------------------------

SPECIAL_START_M = 30e-6
SPECIAL_INNER_SHIFT_M = -25e-6
SPECIAL_OUTER_SHIFT_M = 20e-6
SPECIAL_TAPER_LENGTH_M = 150e-6
SPECIAL_TAPER_POWER = 2.0

SPECIAL_LOCK_M = 150e-6
SPECIAL_BULGE_M = 0.0
SPECIAL_BULGE_CENTER_M = 70e-6
SPECIAL_BULGE_WIDTH_M = 25e-6

# Full movable-knot genome contains seven interior controls. In special mode,
# the middle three controls carry the connected 07-family profile:
#
#     d_in(60), d_in(90), d_in(120)
#
# The remaining controls are zero.
SPECIAL_INNER_INDICES = np.array([2, 3, 4], dtype=int)

# Tested / intentionally conservative hard limits.
SPECIAL_INNER_MIN_M = -18e-6
SPECIAL_INNER_MAX_M = +4e-6
SPECIAL_OUTER_MIN_M = -10e-6
SPECIAL_OUTER_MAX_M = +10e-6

GAPS = slice(0, N_INTERVALS)
INNER = slice(
    GAPS.stop,
    GAPS.stop + N_KNOTS - 2,
)
OUTER = slice(
    INNER.stop,
    INNER.stop + N_KNOTS - 2,
)

LOCK = OUTER.stop
CENTER_IN = LOCK + 1
CENTER_OUT = LOCK + 2
START = LOCK + 3
LENGTH = LOCK + 4
POWER = LOCK + 5
BULGE = LOCK + 6
BULGE_CENTER = LOCK + 7
BULGE_WIDTH = LOCK + 8
N_GENES = LOCK + 9

LOW = np.array(
    [-4.0] * N_INTERVALS
    + [-28e-6] * (N_KNOTS - 2)
    + [-28e-6] * (N_KNOTS - 2)
    + [
        LOCK_MIN_M,
        -50e-6,
        -8e-6,
        8e-6,
        60e-6,
        0.70,
        -30e-6,
        10e-6,
        8e-6,
    ],
    dtype=float,
)

HIGH = np.array(
    [4.0] * N_INTERVALS
    + [28e-6] * (N_KNOTS - 2)
    + [28e-6] * (N_KNOTS - 2)
    + [
        LOCK_MAX_M,
        -3e-6,
        65e-6,
        55e-6,
        310e-6,
        5.25,
        30e-6,
        140e-6,
        80e-6,
    ],
    dtype=float,
)


Family = Literal["ordinary", "special"]


# =============================================================================
# Configuration and population types
# =============================================================================

@dataclass(frozen=True)
class Settings:
    backend: str

    ordinary_islands: int
    special_islands: int

    population: int
    offspring: int
    generations: int

    migration_interval: int
    migrants: int
    retain_per_island: int

    seed: int
    coarse_points: int
    fine_points: int
    fine_top: int
    max_panels: int
    fast: bool

    animation_stride: int
    surface_points: int

    stall_generations: int
    burst_generations: int
    improvement_fraction: float
    burst_mutation_multiplier: float

    output_dir: Path

    rf_peak_v: float = 100.0
    target_z_m: float = TARGET_ION_HEIGHT_M
    dz_limit_m: float = 3e-6
    barrier_limit_ev: float = 1e-3
    lateral_limit_m: float = 5e-6
    rail_min_width_m: float = 18e-6

    @property
    def islands(self) -> int:
        return self.ordinary_islands + self.special_islands


@dataclass(frozen=True)
class IslandStyle:
    name: str
    weights: tuple[float, float]
    explore: float
    final: float
    emphasis: str
    family: str


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    family: Family
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result["valid"])

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf])

        return np.array(
            [
                self.result["dz_peak_m"] / 3e-6,
                self.result["barrier_ev"] / 1e-3,
            ],
            dtype=float,
        )

    def copy(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(),
            result=dict(self.result),
            island=self.island,
            generation=self.generation,
            origin=self.origin,
            family=self.family,
            rank=self.rank,
            crowding=self.crowding,
        )


@dataclass
class Island:
    style: IslandStyle
    rng: np.random.Generator
    population: list[Individual]


# =============================================================================
# Fixed BEM field wrapper
# =============================================================================

class BEMField:
    """Single-candidate field adapter required by RF-null tracing."""

    def __init__(
        self,
        bem: FixedMeshBEM,
        charge: Any,
    ) -> None:
        self.bem = bem
        self.charge = charge

    def electric_field(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> tuple[float, float, float]:
        values = self.bem.field_batch(
            np.array([x_m], dtype=float),
            np.array([y_m], dtype=float),
            np.array([z_m], dtype=float),
            self.charge,
        )
        field = self.bem.asnumpy(values)[0, 0]
        return (
            float(field[0]),
            float(field[1]),
            float(field[2]),
        )


# =============================================================================
# Genome construction, decoding and local C4v features
# =============================================================================

def repair(genome: np.ndarray) -> np.ndarray:
    """Clip one genome to the established workflow-09 bounds."""
    values = np.asarray(genome, dtype=float).copy()

    if values.shape != (N_GENES,):
        raise ValueError(
            f"Expected genome shape {(N_GENES,)}, "
            f"got {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Genome contains non-finite values.")

    return np.clip(values, LOW, HIGH)


def decode(
    genome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode gap logits and inner/outer movable contour offsets."""
    values = repair(genome)

    lock_m = float(values[LOCK])

    knots_m = softmax_gap_knots(
        values[GAPS],
        lock_m=lock_m,
        min_gap_m=MIN_GAP_M,
    )

    inner_offsets_m = np.r_[
        0.0,
        values[INNER],
        0.0,
    ]

    outer_offsets_m = np.r_[
        0.0,
        values[OUTER],
        0.0,
    ]

    return (
        knots_m,
        inner_offsets_m,
        outer_offsets_m,
    )


def baseline() -> np.ndarray:
    """Return the validated conventional 09 baseline cross."""
    genome = np.zeros(N_GENES, dtype=float)

    genome[LOCK] = FIXED_CORE_KNOTS_M[-1]
    genome[GAPS] = fixed_knots_to_logits(
        FIXED_CORE_KNOTS_M,
        min_gap_m=MIN_GAP_M,
    )

    genome[CENTER_IN:] = [
        -25e-6,
        20e-6,
        30e-6,
        150e-6,
        2.0,
        0.0,
        70e-6,
        25e-6,
    ]

    return repair(genome)


def conventional_cross() -> np.ndarray:
    """Exact connected conventional straight-cross seed."""
    return baseline()


def gaussian_knot_feature(
    *,
    knots_m: np.ndarray,
    centre_m: float,
    width_m: float,
    amplitude_m: float,
) -> np.ndarray:
    """Generate a localized feature sampled at interior movable knots.

    The endpoint values remain zero, preserving the required attachment of
    the movable contour to the ordinary straight transport cross. Since the
    underlying geometry is C4v, this one-arm feature appears on all four
    arms with the correct symmetry.
    """
    if width_m <= 0.0:
        raise ValueError("width_m must be positive.")

    normalized = (
        (np.asarray(knots_m, dtype=float) - centre_m)
        / width_m
    )

    values = amplitude_m * np.exp(
        -0.5 * normalized**2
    )

    values[0] = 0.0
    values[-1] = 0.0

    return values


def special_feature_seed(
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    """Create a random, but constrained, 07c--07g special seed.

    It never constructs a new RF topology. The result is a conventional
    connected C4v cross with a local connected inner/outer contour change.
    """
    child = baseline()

    # Start from exact validated 07 geometry.
    child[LOCK] = SPECIAL_LOCK_M
    child[GAPS] = fixed_knots_to_logits(
        FIXED_CORE_KNOTS_M,
        min_gap_m=MIN_GAP_M,
    )

    child[CENTER_IN] = SPECIAL_INNER_SHIFT_M
    child[CENTER_OUT] = SPECIAL_OUTER_SHIFT_M
    child[START] = SPECIAL_START_M
    child[LENGTH] = SPECIAL_TAPER_LENGTH_M
    child[POWER] = SPECIAL_TAPER_POWER

    child[BULGE] = SPECIAL_BULGE_M
    child[BULGE_CENTER] = SPECIAL_BULGE_CENTER_M
    child[BULGE_WIDTH] = SPECIAL_BULGE_WIDTH_M

    mode = int(rng.integers(0, 5))

    if mode == 0:
        # Random triangular RF protrusion toward transport channel.
        d90_um = float(rng.uniform(-16.0, -11.0))
        d60_um = float(
            rng.uniform(
                -0.75 * abs(d90_um),
                -0.40 * abs(d90_um),
            )
        )
        d120_um = float(
            rng.uniform(
                -0.75 * abs(d90_um),
                -0.40 * abs(d90_um),
            )
        )
        a_out_um = 0.0
        label = "special_random_triangle"

    elif mode == 1:
        # Random broad connected inner profile.
        level_um = float(rng.uniform(-13.0, -8.0))
        d60_um = float(level_um + rng.normal(0.0, 1.2))
        d90_um = float(level_um + rng.normal(0.0, 1.2))
        d120_um = float(level_um + rng.normal(0.0, 1.2))
        a_out_um = 0.0
        label = "special_random_broad_inner"

    elif mode == 2:
        # Coupled connected inner/outer profile, broad outer amplitude.
        d60_um = float(rng.uniform(-15.0, -9.0))
        d90_um = float(rng.uniform(-14.0, -9.0))
        d120_um = float(rng.uniform(-12.0, -7.0))
        a_out_um = float(rng.uniform(-10.0, +10.0))
        label = "special_random_coupled"

    elif mode == 3:
        # Local outer inward notch around a valid inner profile.
        d60_um = float(rng.uniform(-14.0, -8.0))
        d90_um = float(rng.uniform(-12.0, -8.0))
        d120_um = float(rng.uniform(-11.0, -6.0))
        a_out_um = float(rng.uniform(-10.0, -3.0))
        label = "special_random_outer_notch"

    else:
        # Local outer outward bulge around a valid inner profile.
        d60_um = float(rng.uniform(-14.0, -8.0))
        d90_um = float(rng.uniform(-15.0, -9.0))
        d120_um = float(rng.uniform(-11.0, -6.0))
        a_out_um = float(rng.uniform(+3.0, +10.0))
        label = "special_random_outer_bulge"

    child[INNER] = 0.0
    child[INNER][SPECIAL_INNER_INDICES] = (
        np.clip(
            [d60_um, d90_um, d120_um],
            SPECIAL_INNER_MIN_M * 1e6,
            SPECIAL_INNER_MAX_M * 1e6,
        )
        * 1e-6
    )

    child[OUTER] = 0.0
    child[OUTER][SPECIAL_INNER_INDICES] = (
        np.clip(
            a_out_um,
            SPECIAL_OUTER_MIN_M * 1e6,
            SPECIAL_OUTER_MAX_M * 1e6,
        )
        * 1e-6
    )

    return label, special_07_repair(child)
def ordinary_seeds() -> list[tuple[str, np.ndarray]]:
    """Return established 09-style ordinary connected-cross seeds."""
    base = baseline()

    inner = base.copy()
    inner[INNER] = (
        np.array(
            [-4, -8, -12, -12, -8, -4, -2],
            dtype=float,
        )
        * 1e-6
    )

    outer = base.copy()
    outer[OUTER] = (
        np.array(
            [2, 5, 9, 11, 8, 4, 2],
            dtype=float,
        )
        * 1e-6
    )

    coupled = base.copy()
    coupled[INNER] = (
        np.array(
            [-3, -7, -11, -13, -9, -5, -2],
            dtype=float,
        )
        * 1e-6
    )
    coupled[OUTER] = (
        np.array(
            [2, 5, 10, 12, 8, 4, 1],
            dtype=float,
        )
        * 1e-6
    )

    packed = base.copy()
    packed[GAPS] = np.array(
        [1.4, 1.0, 0.6, 0.1, -0.3, -0.7, -1.0, -1.2],
        dtype=float,
    )
    packed[LOCK] = 130e-6

    return [
        ("conventional_exact", conventional_cross()),
        ("ordinary_inner", repair(inner)),
        ("ordinary_outer", repair(outer)),
        ("ordinary_coupled", repair(coupled)),
        ("ordinary_packed_x", repair(packed)),
    ]


def parameters(genome: np.ndarray):
    """Build standard project X-junction parameters from a genome."""
    values = repair(genome)

    knots_m, inner_offsets_m, outer_offsets_m = decode(values)

    contour = MovableKnotContour(
        knots_m=knots_m,
        inner_offsets_m=inner_offsets_m,
        outer_offsets_m=outer_offsets_m,
        lock_m=float(values[LOCK]),
        min_gap_m=MIN_GAP_M,
    )

    return build_movable_knot_parameters(
        contour=contour,
        inner_edge_shift_at_centre_m=float(
            values[CENTER_IN]
        ),
        outer_edge_shift_at_centre_m=float(
            values[CENTER_OUT]
        ),
        rf_start_radius_override_m=float(
            values[START]
        ),
        taper_length_m=float(values[LENGTH]),
        taper_power=float(values[POWER]),
        outer_bulge_amplitude_m=float(
            values[BULGE]
        ),
        outer_bulge_center_m=float(
            values[BULGE_CENTER]
        ),
        outer_bulge_sigma_m=float(
            values[BULGE_WIDTH]
        ),
    )


# =============================================================================
# BEM, tracing and physical evaluation
# =============================================================================

def rejected(
    reason: str,
    stage: str,
) -> dict[str, Any]:
    """Return a stable invalid evaluation record."""
    return {
        "valid": False,
        "verified": False,
        "reason": reason,
        "stage": stage,
        "dz_peak_m": np.inf,
        "dz_rms_m": np.inf,
        "barrier_ev": np.inf,
        "lateral_peak_m": np.inf,
        "barrier_reference_ev": np.nan,
        "panels": 0,
        "points": 0,
        "converged": 0,
    }


def mesh(
    parameters_object: Any,
    *,
    fast: bool,
):
    """Build the established geometry-aware quadtree mesh."""
    if fast:
        return build_geometry_aware_quadtree_x_junction_bem(
            parameters_object,
            central_half_extent_m=145e-6,
            central_max_cell_m=46e-6,
            boundary_max_cell_m=17e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=8e-6,
        )

    return build_geometry_aware_quadtree_x_junction_bem(
        parameters_object,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )


def evaluate(
    genome: np.ndarray,
    cfg: Settings,
    *,
    stage: str = "coarse",
    retain: bool = False,
) -> tuple[dict[str, Any], tuple[Any, Any, np.ndarray, BEMField] | None]:
    """Evaluate one candidate using the same physical workflow as 09."""
    values = repair(genome)

    try:
        parameter_object = parameters(values)

        report = check_x_junction_manufacturability(
            parameter_object
        )

        if not report.valid:
            return (
                rejected(
                    "geometry: "
                    + "; ".join(
                        map(str, report.messages)
                    ),
                    stage,
                ),
                None,
            )

        longitudinal_m = np.linspace(
            0.0,
            parameter_object.arm_length_m,
            2001,
        )

        inner_m, outer_m = (
            parameter_object.rail_boundaries_m(
                longitudinal_m
            )
        )

        if np.min(inner_m) <= 0.5e-6:
            return rejected("inner clearance", stage), None

        if (
            np.min(outer_m - inner_m)
            < cfg.rail_min_width_m
        ):
            return rejected("rail width", stage), None

        model = mesh(
            parameter_object,
            fast=cfg.fast and stage == "coarse",
        )

        if model.n_panels > cfg.max_panels:
            return rejected("panel limit", stage), None

        bem = FixedMeshBEM(
            model.bem.panels_m,
            backend=cfg.backend,
        )

        bem.assemble(block_rows=64)
        bem.factorize()
        bem.release_matrix()

        charge = bem.solve_masks(
            model.bem.electrode_voltages_v
        )
        field = BEMField(bem, charge)

        points = (
            cfg.fine_points
            if stage == "fine"
            else cfg.coarse_points
        )

        trace = trace_rf_transverse_minimum(
            field,
            np.linspace(
                -350e-6,
                350e-6,
                points,
            ),
            initial_y_m=0.0,
            initial_z_m=cfg.target_z_m,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )

        converged = np.asarray(
            trace.converged,
            dtype=bool,
        )
        z_m = np.asarray(trace.z_m, dtype=float)
        y_m = np.asarray(trace.y_m, dtype=float)

        if (
            not trace.valid
            or converged.shape != (points,)
            or not np.all(converged)
        ):
            return rejected("incomplete trace", stage), None

        if (
            not np.all(np.isfinite(z_m))
            or not np.all(np.isfinite(y_m))
        ):
            return rejected("nonfinite trace", stage), None

        pseudo_ev = np.asarray(
            pseudopotential_profile_ev(
                field,
                trace,
                rf_voltage_peak_v=cfg.rf_peak_v,
                rf_angular_frequency_rad_s=(
                    RF_ANGULAR_FREQUENCY_RAD_S
                ),
            ),
            dtype=float,
        )

        barrier = compute_barrier_metrics(
            pseudo_ev,
            trace,
        )

        if (
            not barrier.valid
            or pseudo_ev.shape != (points,)
            or not np.all(np.isfinite(pseudo_ev))
        ):
            return rejected(
                "invalid pseudopotential",
                stage,
            ), None

        dz_m = z_m - cfg.target_z_m

        peak_m = float(np.max(np.abs(dz_m)))
        rms_m = float(np.sqrt(np.mean(dz_m**2)))
        lateral_m = float(np.max(np.abs(y_m)))
        barrier_ev = float(barrier.barrier_height_ev)

        if not np.all(
            np.isfinite(
                [
                    peak_m,
                    rms_m,
                    lateral_m,
                    barrier_ev,
                ]
            )
        ):
            return rejected("nonfinite metrics", stage), None

        result = {
            "valid": True,
            "verified": bool(
                stage == "fine"
                and peak_m <= cfg.dz_limit_m
                and barrier_ev <= cfg.barrier_limit_ev
                and lateral_m <= cfg.lateral_limit_m
            ),
            "reason": "ok",
            "stage": stage,
            "backend": bem.backend_name,
            "dz_peak_m": peak_m,
            "dz_rms_m": rms_m,
            "barrier_ev": barrier_ev,
            "lateral_peak_m": lateral_m,
            "barrier_reference_ev": float(
                barrier.reference_energy_ev
            ),
            "panels": int(model.n_panels),
            "points": int(points),
            "converged": int(np.sum(converged)),
        }

        data = (
            (model, trace, pseudo_ev, field)
            if retain
            else None
        )

        return result, data

    except Exception as error:
        if (
            cfg.backend == "cuda"
            and "out of memory"
            in str(error).lower()
        ):
            raise RuntimeError(
                "CUDA OOM: reduce --max-panels "
                "or use --fast."
            ) from error

        return (
            rejected(
                f"geometry/BEM error "
                f"{type(error).__name__}: "
                f"{str(error).replace(chr(10), ' ').strip()}",
                stage,
            ),
            None,
        )


# =============================================================================
# NSGA-II ranking and operators
# =============================================================================

def dominates(
    left: Individual,
    right: Individual,
) -> bool:
    """Return true when left Pareto-dominates right."""
    if left.valid != right.valid:
        return left.valid

    return bool(
        left.valid
        and np.all(left.objectives <= right.objectives)
        and np.any(left.objectives < right.objectives)
    )


def rank_and_crowding(
    population: list[Individual],
) -> list[list[int]]:
    """Assign NSGA-II non-dominated ranks and crowding distances."""
    count = len(population)

    defeated: list[list[int]] = [
        [] for _ in population
    ]
    losses = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]

    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left = population[left_index]
            right = population[right_index]

            if dominates(left, right):
                defeated[left_index].append(right_index)
                losses[right_index] += 1
            elif dominates(right, left):
                defeated[right_index].append(left_index)
                losses[left_index] += 1

    fronts[0] = [
        index
        for index in range(count)
        if losses[index] == 0
    ]

    while fronts[-1]:
        next_front: list[int] = []

        for winner in fronts[-1]:
            for loser in defeated[winner]:
                losses[loser] -= 1

                if losses[loser] == 0:
                    next_front.append(loser)

        fronts.append(next_front)

    fronts.pop()

    for rank, front in enumerate(fronts):
        for index in front:
            population[index].rank = rank
            population[index].crowding = 0.0

        valid = [
            index
            for index in front
            if population[index].valid
        ]

        if len(valid) <= 2:
            for index in valid:
                population[index].crowding = np.inf
            continue

        for axis in range(2):
            order = sorted(
                valid,
                key=lambda index: population[
                    index
                ].objectives[axis],
            )

            low = population[order[0]].objectives[axis]
            high = population[order[-1]].objectives[axis]

            if high <= low:
                continue

            population[order[0]].crowding = np.inf
            population[order[-1]].crowding = np.inf

            for position in range(1, len(order) - 1):
                index = order[position]

                if np.isfinite(
                    population[index].crowding
                ):
                    before = population[
                        order[position - 1]
                    ].objectives[axis]
                    after = population[
                        order[position + 1]
                    ].objectives[axis]

                    population[index].crowding += (
                        (after - before)
                        / (high - low)
                    )

    return fronts


def survivors(
    pool: list[Individual],
    size: int,
) -> list[Individual]:
    """Select an NSGA-II survivor population."""
    result: list[Individual] = []

    for front in rank_and_crowding(pool):
        ordered = sorted(
            front,
            key=lambda index: pool[index].crowding,
            reverse=True,
        )

        available = size - len(result)

        result.extend(
            pool[index].copy()
            for index in ordered[:available]
        )

        if len(result) == size:
            break

    return result


def style_score(
    individual: Individual,
    style: IslandStyle,
) -> float:
    """Scalar preference used only for tournament tie-breaking."""
    if not individual.valid:
        return np.inf

    if style.emphasis == "joint":
        return max(
            individual.result["dz_peak_m"] / 10e-6,
            individual.result["barrier_ev"] / 10e-3,
        )

    return float(
        np.dot(
            np.asarray(style.weights),
            individual.objectives,
        )
    )


def tournament(
    population: list[Individual],
    style: IslandStyle,
    rng: np.random.Generator,
) -> Individual:
    """Select a parent using rank, crowding then local style score."""
    first, second = (
        population[int(index)]
        for index in rng.integers(
            0,
            len(population),
            2,
        )
    )

    if first.valid != second.valid:
        return first if first.valid else second

    if first.rank != second.rank:
        return (
            first
            if first.rank < second.rank
            else second
        )

    if first.crowding != second.crowding:
        return (
            first
            if first.crowding > second.crowding
            else second
        )

    return (
        first
        if style_score(first, style)
        <= style_score(second, style)
        else second
    )


def anneal(
    style: IslandStyle,
    fraction: float,
) -> float:
    """Return exploration scale with late cosine cooling."""
    if fraction <= 0.75:
        return style.explore

    normalized = (fraction - 0.75) / 0.25

    return (
        style.final
        + 0.5
        * (style.explore - style.final)
        * (1.0 + math.cos(math.pi * normalized))
    )


def sigmas(
    style: IslandStyle,
    fraction: float,
) -> np.ndarray:
    """Return family/style-specific mutation amplitudes."""
    scale = anneal(style, fraction)
    values = np.zeros(N_GENES, dtype=float)

    values[GAPS] = 0.34 * scale
    values[INNER] = 8e-6 * scale
    values[OUTER] = 8e-6 * scale
    values[LOCK] = 22e-6 * scale
    values[CENTER_IN:CENTER_OUT + 1] = 9e-6 * scale
    values[START] = 10e-6 * scale
    values[LENGTH] = 24e-6 * scale
    values[POWER] = 0.35 * scale
    values[BULGE] = 10e-6 * scale
    values[BULGE_CENTER] = 18e-6 * scale
    values[BULGE_WIDTH] = 12e-6 * scale

    if style.emphasis == "inner":
        values[INNER] *= 1.8
    elif style.emphasis == "outer":
        values[OUTER] *= 1.8
    elif style.emphasis == "x":
        values[GAPS] *= 2.3
        values[LOCK] *= 1.7
    elif style.emphasis == "inner_x":
        values[INNER] *= 1.7
        values[GAPS] *= 1.5
    elif style.emphasis == "outer_x":
        values[OUTER] *= 1.7
        values[GAPS] *= 1.5
    elif style.emphasis == "central":
        values[CENTER_IN:START + 1] *= 2.0
    elif style.emphasis == "wide":
        values *= 1.35
    elif style.emphasis == "local":
        values *= 0.55
    elif style.emphasis == "feature":
        values[INNER] *= 2.4
        values[OUTER] *= 2.4
        values[GAPS] *= 1.5
        values[LOCK] *= 1.4
    elif style.emphasis == "feature_inner":
        values[INNER] *= 3.0
        values[OUTER] *= 1.4
        values[GAPS] *= 1.7
    elif style.emphasis == "feature_outer":
        values[INNER] *= 1.4
        values[OUTER] *= 3.0
        values[GAPS] *= 1.7
    elif style.emphasis == "feature_coupled":
        values[INNER] *= 2.5
        values[OUTER] *= 2.5
        values[CENTER_IN:CENTER_OUT + 1] *= 1.8

    return values


def mutate(
    parent: np.ndarray,
    style: IslandStyle,
    rng: np.random.Generator,
    fraction: float,
    burst: bool = False,
    burst_mutation_multiplier: float = 1.0,
) -> np.ndarray:
    """Mutate a genome with focused ordinary/special boundary exploration."""
    child = np.asarray(parent, dtype=float).copy()
    
    if style.family == "special":
        return mutate_special_07_family(
            child,
            style,
            rng,
            fraction,
            burst=burst,
            burst_mutation_multiplier=burst_mutation_multiplier,
        )

    sigma = sigmas(style, fraction)

    if burst:
        sigma *= burst_mutation_multiplier

    chance = np.full(N_GENES, 0.55, dtype=float)

    if style.emphasis in (
        "x",
        "inner_x",
        "outer_x",
        "feature",
        "feature_inner",
        "feature_outer",
        "feature_coupled",
    ):
        chance[GAPS] = 0.88

    if style.emphasis in (
        "inner",
        "inner_x",
        "feature_inner",
    ):
        chance[INNER] = 0.88

    if style.emphasis in (
        "outer",
        "outer_x",
        "feature_outer",
    ):
        chance[OUTER] = 0.88

    if style.family == "special":
        chance[INNER] = np.maximum(
            chance[INNER],
            0.84,
        )
        chance[OUTER] = np.maximum(
            chance[OUTER],
            0.84,
        )

    if burst:
        chance = np.maximum(chance, 0.82)
        chance[GAPS] = 0.92
        chance[INNER] = 0.88
        chance[OUTER] = 0.88
        chance[LOCK] = 0.92
        chance[CENTER_IN:CENTER_OUT + 1] = 0.90
        chance[START:LENGTH + 1] = 0.85

    active = rng.random(N_GENES) < chance

    child[active] += rng.normal(
        0.0,
        sigma[active],
    )

    return repair(child)


def mutate_special_07_family(
    genome: np.ndarray,
    style: IslandStyle,
    rng: np.random.Generator,
    fraction: float,
    *,
    burst: bool,
    burst_mutation_multiplier: float,
) -> np.ndarray:
    """Mutate only d60, d90, d120 and broad a_out.

    This function intentionally cannot mutate:
    - movable knot locations;
    - lock;
    - centre shifts;
    - RF start;
    - taper;
    - taper power;
    - Gaussian outer bulge.

    That is the essential restriction that keeps special islands in the
    connected 07c--07g geometry family.
    """
    child = special_07_repair(genome)

    scale = anneal(style, fraction)

    inner_sigma_m = 2.0e-6 * scale
    outer_sigma_m = 1.5e-6 * scale
    mutation_probability = 0.58

    if burst:
        inner_sigma_m *= burst_mutation_multiplier
        outer_sigma_m *= burst_mutation_multiplier
        mutation_probability = 0.90

    # Avoid chained NumPy indexing: boolean indexing otherwise returns copy.
    inner_middle = child[INNER][
        SPECIAL_INNER_INDICES
    ].copy()

    active_inner = (
        rng.random(len(SPECIAL_INNER_INDICES))
        < mutation_probability
    )

    if np.any(active_inner):
        inner_middle[active_inner] += rng.normal(
            0.0,
            inner_sigma_m,
            size=int(np.sum(active_inner)),
        )

    inner_middle = np.clip(
        inner_middle,
        SPECIAL_INNER_MIN_M,
        SPECIAL_INNER_MAX_M,
    )

    child[INNER][SPECIAL_INNER_INDICES] = inner_middle

    outer_common_m = float(
        np.mean(
            child[OUTER][SPECIAL_INNER_INDICES]
        )
    )

    if rng.random() < 0.80 * mutation_probability:
        outer_common_m += float(
            rng.normal(0.0, outer_sigma_m)
        )

    outer_common_m = float(
        np.clip(
            outer_common_m,
            SPECIAL_OUTER_MIN_M,
            SPECIAL_OUTER_MAX_M,
        )
    )

    child[OUTER] = 0.0
    child[OUTER][SPECIAL_INNER_INDICES] = outer_common_m

    return special_07_repair(child)
def special_07_repair(genome: np.ndarray) -> np.ndarray:
    """Project any inherited genome onto the constrained 07c--07g family.

    This is critical after:
    - ordinary -> special migration;
    - ordinary/special crossover;
    - burst mixing;
    - random restart.

    It prevents a special island from inheriting free movable knots, arbitrary
    taper/global genes, or a Gaussian outer bulge that could visually destroy
    the conventional connected RF cross.

    Free physical controls after projection:
        d_in(60), d_in(90), d_in(120)
        a_out shared at 60, 90, 120 um
    """
    source = repair(genome)
    child = baseline()

    # Fixed validated 06c / 07-family baseline.
    child[LOCK] = SPECIAL_LOCK_M
    child[GAPS] = fixed_knots_to_logits(
        FIXED_CORE_KNOTS_M,
        min_gap_m=MIN_GAP_M,
    )

    child[CENTER_IN] = SPECIAL_INNER_SHIFT_M
    child[CENTER_OUT] = SPECIAL_OUTER_SHIFT_M
    child[START] = SPECIAL_START_M
    child[LENGTH] = SPECIAL_TAPER_LENGTH_M
    child[POWER] = SPECIAL_TAPER_POWER

    # No unconstrained Gaussian outer bulge in special family.
    child[BULGE] = SPECIAL_BULGE_M
    child[BULGE_CENTER] = SPECIAL_BULGE_CENTER_M
    child[BULGE_WIDTH] = SPECIAL_BULGE_WIDTH_M

    # Only the central three inner controls survive.
    child[INNER] = 0.0

    inner_middle = np.asarray(
        source[INNER][SPECIAL_INNER_INDICES],
        dtype=float,
    )

    child[INNER][SPECIAL_INNER_INDICES] = np.clip(
        inner_middle,
        SPECIAL_INNER_MIN_M,
        SPECIAL_INNER_MAX_M,
    )

    # Outer contour is a broad connected profile: the SAME amplitude is used
    # at all active special positions, as in workflow 07g.
    outer_common_m = float(
        np.clip(
            np.mean(source[OUTER][SPECIAL_INNER_INDICES]),
            SPECIAL_OUTER_MIN_M,
            SPECIAL_OUTER_MAX_M,
        )
    )

    child[OUTER] = 0.0
    child[OUTER][SPECIAL_INNER_INDICES] = outer_common_m

    return repair(child)
def block_crossover(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cross genome blocks while retaining meaningful contour structure."""
    child = np.empty(N_GENES, dtype=float)

    for block in (
        GAPS,
        INNER,
        OUTER,
        slice(LOCK, N_GENES),
    ):
        alpha = float(rng.uniform(-0.10, 1.10))

        child[block] = (
            alpha * left[block]
            + (1.0 - alpha) * right[block]
        )

    return repair(child)


# =============================================================================
# Island styles and persistence
# =============================================================================

def make_styles(
    ordinary_islands: int,
    special_islands: int,
) -> list[IslandStyle]:
    ordinary_templates = (
        IslandStyle(
            "ordinary_height",
            (7.0, 1.0),
            1.00,
            0.10,
            "inner",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_barrier",
            (1.0, 7.0),
            1.00,
            0.10,
            "outer",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_balanced",
            (2.0, 2.0),
            0.90,
            0.09,
            "both",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_movable_x",
            (2.0, 2.0),
            1.05,
            0.10,
            "x",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_central",
            (3.0, 2.0),
            1.05,
            0.10,
            "central",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_inner_x",
            (4.0, 2.0),
            1.10,
            0.11,
            "inner_x",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_outer_x",
            (2.0, 4.0),
            1.10,
            0.11,
            "outer_x",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_wide",
            (2.0, 2.0),
            1.25,
            0.14,
            "wide",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_joint",
            (2.0, 2.0),
            0.95,
            0.08,
            "joint",
            "ordinary",
        ),
        IslandStyle(
            "ordinary_local",
            (2.0, 2.0),
            0.42,
            0.03,
            "local",
            "ordinary",
        ),
    )

    special_templates = (
        IslandStyle(
            "special_height",
            (6.0, 1.0),
            1.20,
            0.12,
            "special_inner",
            "special",
        ),
        IslandStyle(
            "special_barrier",
            (1.0, 6.0),
            1.20,
            0.12,
            "special_outer",
            "special",
        ),
        IslandStyle(
            "special_coupled",
            (3.0, 3.0),
            1.25,
            0.13,
            "special_coupled",
            "special",
        ),
        IslandStyle(
            "special_local",
            (2.0, 2.0),
            0.80,
            0.06,
            "special_local",
            "special",
        ),
        IslandStyle(
            "special_diverse",
            (2.0, 2.0),
            1.30,
            0.14,
            "special_diverse",
            "special",
        ),
    )

    ordinary = [
        ordinary_templates[index % len(ordinary_templates)]
        for index in range(ordinary_islands)
    ]

    special = [
        special_templates[index % len(special_templates)]
        for index in range(special_islands)
    ]

    return ordinary + special

def serialize(
    individual: Individual,
) -> dict[str, Any]:
    """Convert an individual to a JSON/CSV-safe record."""
    knots_m, inner_m, outer_m = decode(
        individual.genome
    )

    return {
        "island": individual.island,
        "generation": individual.generation,
        "origin": individual.origin,
        "family": individual.family,
        "rank": individual.rank,
        "crowding": individual.crowding,
        "genome": individual.genome.tolist(),
        "movable_knots_m": knots_m.tolist(),
        "inner_offsets_m": inner_m.tolist(),
        "outer_offsets_m": outer_m.tolist(),
        "lock_m": float(individual.genome[LOCK]),
        **individual.result,
    }


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write heterogeneous rows to CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted(
        set().union(
            *(row.keys() for row in rows)
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            value,
                            allow_nan=True,
                        )
                        if isinstance(
                            value,
                            (list, dict),
                        )
                        else value
                    )
                    for key, value in row.items()
                }
            )


def elite_rows(
    islands: list[Island],
    cfg: Settings,
) -> list[dict[str, Any]]:
    """Return retained elites from each island."""
    rows: list[dict[str, Any]] = []

    for island in islands:
        rank_and_crowding(island.population)

        best = sorted(
            island.population,
            key=lambda item: (
                item.rank,
                -item.crowding,
                style_score(item, island.style),
            ),
        )[:cfg.retain_per_island]

        rows.extend(serialize(item) for item in best)

    return rows


# =============================================================================
# Geometry, potential maps, reports and animation
# =============================================================================

def draw_geometry(
    axis: Any,
    row: dict[str, Any],
    label: str,
) -> None:
    """Draw readable connected RF contours rather than BEM panel rectangles."""
    raw_genome = row["genome"]

    if isinstance(raw_genome, str):
        raw_genome = json.loads(raw_genome)

    genome = repair(
        np.asarray(raw_genome, dtype=float)
    )

    parameter_object = parameters(genome)

    longitudinal_m = np.linspace(
        0.0,
        260e-6,
        900,
    )
    inner_m, outer_m = (
        parameter_object.rail_boundaries_m(
            longitudinal_m
        )
    )

    for theta in (
        0.0,
        np.pi / 2.0,
        np.pi,
        3.0 * np.pi / 2.0,
    ):
        cosine = np.cos(theta)
        sine = np.sin(theta)

        x_inner = (
            longitudinal_m * cosine
            - inner_m * sine
        )
        y_inner = (
            longitudinal_m * sine
            + inner_m * cosine
        )
        x_outer = (
            longitudinal_m * cosine
            - outer_m * sine
        )
        y_outer = (
            longitudinal_m * sine
            + outer_m * cosine
        )

        axis.fill(
            np.r_[x_inner, x_outer[::-1]] * 1e6,
            np.r_[y_inner, y_outer[::-1]] * 1e6,
            color="#d84b4b",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.92,
        )

    knots_m, _, _ = decode(genome)
    knot_inner_m, knot_outer_m = (
        parameter_object.rail_boundaries_m(
            knots_m
        )
    )

    axis.plot(
        knots_m * 1e6,
        knot_inner_m * 1e6,
        "ko",
        markersize=2.4,
    )
    axis.plot(
        knots_m * 1e6,
        knot_outer_m * 1e6,
        "wo",
        markeredgecolor="black",
        markersize=2.4,
    )

    axis.axhline(
        0.0,
        color="0.70",
        linewidth=0.35,
    )
    axis.axvline(
        0.0,
        color="0.70",
        linewidth=0.35,
    )

    family = str(row.get("family", "ordinary"))

    axis.set(
        xlim=(-260.0, 260.0),
        ylim=(-260.0, 260.0),
        aspect="equal",
    )
    axis.set_xticks([])
    axis.set_yticks([])

    dz_um = float(row["dz_peak_m"]) * 1e6
    barrier_mev = float(row["barrier_ev"]) * 1e3
    lock_um = float(row["lock_m"]) * 1e6

    axis.set_title(
        f"{label} [{family}]\n"
        f"dz={dz_um:.2f} um, "
        f"U={barrier_mev:.2f} meV\n"
        f"lock={lock_um:.0f} um",
        fontsize=8,
    )


def pseudopotential_surface_ev(
    field: BEMField,
    *,
    z_m: float,
    extent_m: float,
    points: int,
    rf_peak_v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute U_ps(x,y,z) over a square plane above the electrodes."""
    coordinates_m = np.linspace(
        -extent_m,
        extent_m,
        points,
    )
    x_grid_m, y_grid_m = np.meshgrid(
        coordinates_m,
        coordinates_m,
        indexing="xy",
    )

    x_flat_m = x_grid_m.reshape(-1)
    y_flat_m = y_grid_m.reshape(-1)
    z_flat_m = np.full_like(
        x_flat_m,
        z_m,
    )

    values = field.bem.field_batch(
        x_flat_m,
        y_flat_m,
        z_flat_m,
        field.charge,
        candidate_chunk=1,
    )

    electric_field = field.bem.asnumpy(values)[0]

    field_squared = np.sum(
        electric_field**2,
        axis=1,
    )

    # Ca-40 charge/mass constants used in existing project evaluator.
    elementary_charge_c = 1.602176634e-19
    atomic_mass_unit_kg = 1.66053906660e-27
    ca40_mass_kg = 39.96259098 * atomic_mass_unit_kg

    pseudo_ev = (
        elementary_charge_c
        * rf_peak_v**2
        * field_squared
        / (
            4.0
            * ca40_mass_kg
            * RF_ANGULAR_FREQUENCY_RAD_S**2
        )
    )

    return (
        x_grid_m,
        y_grid_m,
        pseudo_ev.reshape(x_grid_m.shape),
    )


def fine_plots(
    data: tuple[Any, Any, np.ndarray, BEMField],
    result: dict[str, Any],
    directory: Path,
    title: str,
    cfg: Settings,
) -> None:
    """Save normal electrode layout, U_ps surface and path profiles."""
    model, trace, pseudo_ev, field = data

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels_m = np.asarray(
        model.bem.panels_m,
        dtype=float,
    )
    rf_mask = (
        np.asarray(
            model.bem.electrode_voltages_v,
            dtype=float,
        )
        > 0.5
    )

    rectangles = [
        Rectangle(
            (
                panel[0] * 1e6,
                panel[2] * 1e6,
            ),
            (panel[1] - panel[0]) * 1e6,
            (panel[3] - panel[2]) * 1e6,
        )
        for panel in panels_m
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 7.0),
    )

    collection = PatchCollection(
        rectangles,
        array=rf_mask.astype(float),
        cmap="RdYlBu_r",
        edgecolor="none",
    )
    axis.add_collection(collection)

    axis.plot(
        np.asarray(trace.x_m) * 1e6,
        np.asarray(trace.y_m) * 1e6,
        "k-",
        linewidth=1.25,
        label="RF null trace",
    )

    axis.set(
        xlim=(-230.0, 230.0),
        ylim=(-230.0, 230.0),
        aspect="equal",
        title=title,
        xlabel="x [um]",
        ylabel="y [um]",
    )
    axis.legend(loc="upper right")
    figure.colorbar(
        collection,
        ax=axis,
        label="RF electrode mask",
    )
    figure.tight_layout()
    figure.savefig(
        directory / "layout.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    x_grid_m, y_grid_m, surface_ev = (
        pseudopotential_surface_ev(
            field,
            z_m=cfg.target_z_m,
            extent_m=230e-6,
            points=cfg.surface_points,
            rf_peak_v=cfg.rf_peak_v,
        )
    )

    surface_mev = surface_ev * 1e3
    finite = surface_mev[np.isfinite(surface_mev)]

    if finite.size:
        upper = float(
            np.percentile(finite, 97.0)
        )
        upper = max(upper, 1e-6)
    else:
        upper = 1.0

    figure, axis = plt.subplots(
        figsize=(8.0, 7.0),
    )

    image = axis.contourf(
        x_grid_m * 1e6,
        y_grid_m * 1e6,
        np.clip(surface_mev, 0.0, upper),
        levels=60,
        cmap="magma",
    )

    axis.contour(
        x_grid_m * 1e6,
        y_grid_m * 1e6,
        np.clip(surface_mev, 0.0, upper),
        levels=12,
        colors="white",
        linewidths=0.35,
        alpha=0.45,
    )

    axis.plot(
        np.asarray(trace.x_m) * 1e6,
        np.asarray(trace.y_m) * 1e6,
        color="cyan",
        linewidth=1.2,
        label="RF null trace",
    )

    axis.set(
        xlim=(-230.0, 230.0),
        ylim=(-230.0, 230.0),
        aspect="equal",
        xlabel="x [um]",
        ylabel="y [um]",
        title=(
            f"{title}\n"
            f"Pseudopotential at "
            f"z={cfg.target_z_m * 1e6:.1f} um"
        ),
    )
    axis.legend(loc="upper right")

    figure.colorbar(
        image,
        ax=axis,
        label="RF pseudopotential [meV]",
    )
    figure.tight_layout()
    figure.savefig(
        directory / "pseudopotential_surface.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    x_um = np.asarray(trace.x_m) * 1e6

    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(9.5, 9.0),
    )

    axes[0].plot(
        x_um,
        (
            np.asarray(trace.z_m)
            - TARGET_ION_HEIGHT_M
        )
        * 1e6,
        color="tab:blue",
    )
    axes[0].axhline(
        3.0,
        color="red",
        linestyle="--",
    )
    axes[0].axhline(
        -3.0,
        color="red",
        linestyle="--",
    )
    axes[0].set_ylabel("z - target [um]")

    axes[1].plot(
        x_um,
        (
            pseudo_ev
            - result["barrier_reference_ev"]
        )
        * 1e3,
        color="tab:purple",
    )
    axes[1].axhline(
        1.0,
        color="red",
        linestyle="--",
    )
    axes[1].set_ylabel("RF pseudo - ref [meV]")

    axes[2].plot(
        x_um,
        np.asarray(trace.y_m) * 1e6,
        color="tab:green",
    )
    axes[2].set(
        xlabel="x [um]",
        ylabel="y [um]",
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96),
    )
    figure.savefig(
        directory / "profiles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def animate_surfaces(
    cfg: Settings,
) -> None:
    """Create topology-aware GIFs from retained elite geometry frames.

    Geometry animation stays inexpensive during GA. Full U_ps maps are
    generated for fine candidates only, avoiding a large post-run BEM cost
    for every elite in every generation.
    """
    try:
        import pandas as pd
    except ImportError:
        print("GIF skipped: pandas is required.")
        return

    files = sorted(
        (
            cfg.output_dir / "island_elites"
        ).glob("elites_gen_*.csv")
    )[::max(1, cfg.animation_stride)]

    if not files:
        return

    frames = [
        (
            int(
                file.stem.rsplit("_", 1)[-1]
            ),
            pd.read_csv(file),
        )
        for file in files
    ]

    target = cfg.output_dir / "surface_animations"
    target.mkdir(exist_ok=True)

    for island_id in range(cfg.islands):
        choices = []

        for generation, table in frames:
            subset = table[
                (table["island"] == island_id)
                & (table["valid"] == True)
            ].sort_values(
                [
                    "rank",
                    "crowding",
                    "dz_peak_m",
                    "barrier_ev",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    True,
                ],
            ).head(5)

            choices.append(
                (generation, subset)
            )

        if not any(
            not table.empty
            for _, table in choices
        ):
            continue

        figure, axes = plt.subplots(
            1,
            5,
            figsize=(21.0, 4.8),
            constrained_layout=True,
        )

        def update(frame_index: int):
            generation, table = choices[frame_index]

            for axis in axes:
                axis.clear()

            rows = table.to_dict("records")

            for position, axis in enumerate(axes):
                if position < len(rows):
                    try:
                        draw_geometry(
                            axis,
                            rows[position],
                            f"Top {position + 1}",
                        )
                    except Exception as error:
                        axis.text(
                            0.5,
                            0.5,
                            f"draw error\n"
                            f"{type(error).__name__}",
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                        )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "no valid elite",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
                    axis.set_axis_off()

            family = (
                rows[0]["family"]
                if rows
                else "unknown"
            )

            figure.suptitle(
                f"Island {island_id:02d} "
                f"[{family}] top-5 "
                f"connected RF contours | "
                f"generation {generation}",
                fontsize=13,
            )

            return axes

        animation = FuncAnimation(
            figure,
            update,
            frames=len(choices),
            interval=650,
            blit=False,
            repeat=True,
        )

        try:
            animation.save(
                target
                / f"island_{island_id:02d}_top5_surfaces.gif",
                writer=PillowWriter(fps=2),
                dpi=115,
            )
        except Exception as error:
            print(
                f"GIF skipped for island {island_id}: "
                f"{type(error).__name__}: {error}"
            )

        plt.close(figure)


# =============================================================================
# CLI
# =============================================================================

def parser() -> argparse.ArgumentParser:
    """Create command line parser."""
    command = argparse.ArgumentParser(
        description=__doc__,
    )

    command.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    command.add_argument(
        "--ordinary-islands",
        type=int,
        default=21,
    )
    command.add_argument(
        "--special-islands",
        type=int,
        default=9,
    )
    command.add_argument(
        "--population",
        type=int,
        default=20,
    )
    command.add_argument(
        "--offspring",
        type=int,
        default=8,
    )
    command.add_argument(
        "--generations",
        type=int,
        default=100,
    )

    command.add_argument(
        "--migration-interval",
        type=int,
        default=3,
    )
    command.add_argument(
        "--migrants",
        type=int,
        default=3,
    )
    command.add_argument(
        "--retain-per-island",
        type=int,
        default=5,
    )

    command.add_argument(
        "--coarse-points",
        type=int,
        default=61,
    )
    command.add_argument(
        "--fine-points",
        type=int,
        default=181,
    )
    command.add_argument(
        "--fine-top",
        type=int,
        default=25,
    )
    command.add_argument(
        "--max-panels",
        type=int,
        default=7000,
    )

    command.add_argument(
        "--fast",
        action="store_true",
    )
    command.add_argument(
        "--animation-stride",
        type=int,
        default=2,
    )
    command.add_argument(
        "--surface-points",
        type=int,
        default=101,
    )

    command.add_argument(
        "--stall-generations",
        type=int,
        default=5,
    )
    command.add_argument(
        "--burst-generations",
        type=int,
        default=3,
    )
    command.add_argument(
        "--improvement-fraction",
        type=float,
        default=0.01,
    )
    command.add_argument(
        "--burst-mutation-multiplier",
        type=float,
        default=3.0,
    )

    command.add_argument(
        "--seed",
        type=int,
        default=20260926,
    )
    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/09b_mixed_island_ga"
        ),
    )

    command.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Small PowerShell smoke run: "
            "2 ordinary + 1 special islands, "
            "population 8, offspring 2, generations 2."
        ),
    )

    return command


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """Run the 70/30 mixed ordinary/special island GA."""
    arguments = parser().parse_args()

    if arguments.smoke:
        arguments.ordinary_islands = 2
        arguments.special_islands = 1
        arguments.population = 8
        arguments.offspring = 2
        arguments.generations = 2
        arguments.migration_interval = 1
        arguments.migrants = 1
        arguments.retain_per_island = 3
        arguments.coarse_points = 31
        arguments.fine_points = 61
        arguments.fine_top = 5
        arguments.max_panels = 5000
        arguments.fast = True
        arguments.animation_stride = 1
        arguments.surface_points = 61
        arguments.stall_generations = 1
        arguments.burst_generations = 1

    backend = available_backend(arguments.backend)

    if not backend.available:
        raise RuntimeError(backend.reason)

    if arguments.ordinary_islands < 1:
        raise ValueError(
            "--ordinary-islands must be at least one."
        )

    if arguments.special_islands < 1:
        raise ValueError(
            "--special-islands must be at least one."
        )

    if arguments.population < 8:
        raise ValueError(
            "--population must be at least eight."
        )

    if arguments.offspring < 1:
        raise ValueError(
            "--offspring must be positive."
        )

    if arguments.generations < 1:
        raise ValueError(
            "--generations must be positive."
        )

    cfg = Settings(
        backend=backend.selected,
        ordinary_islands=arguments.ordinary_islands,
        special_islands=arguments.special_islands,
        population=arguments.population,
        offspring=arguments.offspring,
        generations=arguments.generations,
        migration_interval=arguments.migration_interval,
        migrants=arguments.migrants,
        retain_per_island=arguments.retain_per_island,
        seed=arguments.seed,
        coarse_points=arguments.coarse_points,
        fine_points=arguments.fine_points,
        fine_top=arguments.fine_top,
        max_panels=arguments.max_panels,
        fast=arguments.fast,
        animation_stride=arguments.animation_stride,
        surface_points=arguments.surface_points,
        stall_generations=arguments.stall_generations,
        burst_generations=arguments.burst_generations,
        improvement_fraction=arguments.improvement_fraction,
        burst_mutation_multiplier=(
            arguments.burst_mutation_multiplier
        ),
        output_dir=arguments.output_dir,
    )

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = make_styles(
        cfg.ordinary_islands,
        cfg.special_islands,
    )

    (
        cfg.output_dir / "settings.json"
    ).write_text(
        json.dumps(
            {
                "settings": asdict(cfg),
                "backend": backend.as_dict(),
                "genome_genes": N_GENES,
                "n_knots": N_KNOTS,
                "min_gap_m": MIN_GAP_M,
                "physical_objectives": [
                    "peak_rf_null_height",
                    "rf_pseudopotential_barrier",
                ],
                "ordinary_islands": cfg.ordinary_islands,
                "special_islands": cfg.special_islands,
                "special_definition": (
                    "Connected C4v RF cross with local "
                    "movable-knot inner/outer protrusions "
                    "and indentations."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "Workflow 09b mixed-island connected C4v GA | "
        f"backend={backend.selected} | "
        f"device={backend.device_name}"
    )
    print(
        f"Islands: {cfg.islands} total = "
        f"{cfg.ordinary_islands} ordinary + "
        f"{cfg.special_islands} special-prior."
    )
    print(
        f"Population/island={cfg.population}; "
        f"offspring/island={cfg.offspring}; "
        f"generations={cfg.generations}."
    )
    print(
        "All candidates use the same connected "
        "movable-knot RF-cross representation. "
        "Special islands are localized C4v "
        "protrusion/indentation priors."
    )

    master_rng = np.random.default_rng(cfg.seed)

    ordinary_seed_list = ordinary_seeds()

    cache: dict[
        tuple[float, ...],
        dict[str, Any],
    ] = {}

    best_height: tuple[np.ndarray, dict[str, Any]] | None = None
    best_barrier: tuple[np.ndarray, dict[str, Any]] | None = None
    best_joint: tuple[np.ndarray, dict[str, Any]] | None = None

    def joint(result: dict[str, Any]) -> float:
        """Physical compromise score for progress tracking."""
        return max(
            result["dz_peak_m"] / 10e-6,
            result["barrier_ev"] / 10e-3,
        )

    def coarse(
        genome: np.ndarray,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        """Cached coarse evaluator and live best-candidate tracker."""
        nonlocal best_height
        nonlocal best_barrier
        nonlocal best_joint

        values = repair(genome)
        key = tuple(np.round(values, 12))

        if key not in cache:
            cache[key], _ = evaluate(
                values,
                cfg,
                stage="coarse",
                retain=False,
            )

        result = dict(cache[key])

        if result["valid"]:
            if (
                best_height is None
                or result["dz_peak_m"]
                < best_height[1]["dz_peak_m"]
            ):
                best_height = (values.copy(), result.copy())

            if (
                best_barrier is None
                or result["barrier_ev"]
                < best_barrier[1]["barrier_ev"]
            ):
                best_barrier = (values.copy(), result.copy())

            if (
                best_joint is None
                or joint(result) < joint(best_joint[1])
            ):
                best_joint = (values.copy(), result.copy())

            payload = {
                "best_height": (
                    None
                    if best_height is None
                    else {
                        "genome": best_height[0].tolist(),
                        "result": best_height[1],
                    }
                ),
                "best_barrier": (
                    None
                    if best_barrier is None
                    else {
                        "genome": best_barrier[0].tolist(),
                        "result": best_barrier[1],
                    }
                ),
                "best_joint": (
                    None
                    if best_joint is None
                    else {
                        "genome": best_joint[0].tolist(),
                        "result": best_joint[1],
                    }
                ),
            }

            (
                cfg.output_dir / "best_live_coarse.json"
            ).write_text(
                json.dumps(
                    payload,
                    indent=2,
                    default=float,
                ),
                encoding="utf-8",
            )

        if progress is not None:
            if best_height is None:
                progress.set_postfix_str(
                    "valid=0",
                    refresh=False,
                )
            else:
                progress.set_postfix(
                    {
                        "dz": (
                            f"{best_height[1]['dz_peak_m'] * 1e6:.2f}um"
                        ),
                        "U": (
                            f"{best_barrier[1]['barrier_ev'] * 1e3:.2f}meV"
                        ),
                        "joint": (
                            f"{best_joint[1]['dz_peak_m'] * 1e6:.2f}um/"
                            f"{best_joint[1]['barrier_ev'] * 1e3:.2f}meV"
                        ),
                    },
                    refresh=False,
                )

        return result

    islands: list[Island] = []

    with tqdm(
        total=cfg.islands * cfg.population,
        desc="Initial BEM",
        unit="candidate",
    ) as progress:
        for island_id, style in enumerate(styles):
            island_rng = np.random.default_rng(
                int(
                    master_rng.integers(
                        0,
                        2**32 - 1,
                    )
                )
            )

            population: list[Individual] = []

            if style.family == "ordinary":
                if island_id == 0:
                    genome = conventional_cross()
                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(
                                genome,
                                progress,
                            ),
                            island=island_id,
                            generation=0,
                            origin="ordinary_exact_conventional_cross",
                            family=style.family,
                        )
                    )
                    progress.update()

                ordering = np.roll(
                    np.arange(
                        len(ordinary_seed_list)
                    ),
                    island_id % len(ordinary_seed_list),
                )

                for index in ordering:
                    if len(population) >= cfg.population:
                        break

                    name, seed = ordinary_seed_list[
                        int(index)
                    ]

                    perturbation = island_rng.normal(
                        0.0,
                        (HIGH - LOW)
                        * 0.012
                        * style.explore,
                    )

                    genome = repair(
                        seed + perturbation
                    )

                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(
                                genome,
                                progress,
                            ),
                            island=island_id,
                            generation=0,
                            origin=(
                                f"ordinary_seed_{name}"
                            ),
                            family=style.family,
                        )
                    )
                    progress.update()

            else:
                while (
                    len(population) < cfg.population
                ):
                    name, seed = special_feature_seed(
                        island_rng
                    )

                    perturbation = island_rng.normal(
                        0.0,
                        (HIGH - LOW)
                        * 0.010
                        * style.explore,
                    )

                    genome = repair(seed + perturbation)

                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(
                                genome,
                                progress,
                            ),
                            island=island_id,
                            generation=0,
                            origin=name,
                            family=style.family,
                        )
                    )
                    progress.update()

            while len(population) < cfg.population:
                if style.family == "special":
                    name, seed = special_feature_seed(
                        island_rng
                    )
                else:
                    name, seed = ordinary_seed_list[
                        int(
                            island_rng.integers(
                                len(ordinary_seed_list)
                            )
                        )
                    ]

                genome = repair(
                    seed
                    + island_rng.normal(
                        0.0,
                        (HIGH - LOW)
                        * 0.035
                        * style.explore,
                    )
                )

                population.append(
                    Individual(
                        genome=genome,
                        result=coarse(
                            genome,
                            progress,
                        ),
                        island=island_id,
                        generation=0,
                        origin=f"initial_{name}",
                        family=style.family,
                    )
                )
                progress.update()

            rank_and_crowding(population)

            islands.append(
                Island(
                    style=style,
                    rng=island_rng,
                    population=population,
                )
            )

    valid_initial = [
        item
        for island in islands
        for item in island.population
        if item.valid
    ]

    if not valid_initial:
        reasons = Counter(
            item.result["reason"]
            for island in islands
            for item in island.population
        )
        raise RuntimeError(
            "No valid initial geometry. "
            f"Reasons: {reasons.most_common(20)!r}"
        )

    archive: list[Individual] = []
    history: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    best_joint_seen = np.inf
    stall_count = 0
    burst_remaining = 0
    search_mode = "NORMAL"

    def update_archive(
        individuals: list[Individual],
    ) -> None:
        """Merge valid candidates into a bounded Pareto archive."""
        nonlocal archive

        pool: dict[
            tuple[float, ...],
            Individual,
        ] = {
            tuple(
                np.round(item.genome, 12)
            ): item.copy()
            for item in archive + individuals
            if item.valid
        }

        archive = list(pool.values())

        if not archive:
            return

        rank_and_crowding(archive)

        archive.sort(
            key=lambda item: (
                item.rank,
                -item.crowding,
            )
        )
        archive = archive[:1000]

    def current_best_joint_score() -> float:
        """Return current best valid compromise score."""
        candidates = [
            item
            for island in islands
            for item in island.population
            if item.valid
        ]

        if not candidates:
            return np.inf

        return min(
            joint(item.result)
            for item in candidates
        )

    def summary(
        generation: int,
    ) -> dict[str, Any]:
        """Return generation summary and rejection statistics."""
        all_individuals = [
            item
            for island in islands
            for item in island.population
        ]

        valid = [
            item
            for item in all_individuals
            if item.valid
        ]

        rejected = Counter(
            item.result["reason"]
            for item in all_individuals
            if not item.valid
        )

        for reason, count in rejected.items():
            rejection_rows.append(
                {
                    "generation": generation,
                    "reason": reason,
                    "count": count,
                }
            )

        if not valid:
            return {
                "generation": generation,
                "mode": search_mode,
                "stall_count": stall_count,
                "burst_remaining": burst_remaining,
                "joint_score": np.inf,
                "valid": 0,
                "invalid": len(all_individuals),
                "archive": len(archive),
                "best_dz_um": np.inf,
                "best_U_mev": np.inf,
                "unique_evaluations": len(cache),
                "rejects": "; ".join(
                    f"{key}:{value}"
                    for key, value
                    in rejected.most_common(4)
                ),
            }

        best_dz = min(
            valid,
            key=lambda item: item.result["dz_peak_m"],
        )
        best_u = min(
            valid,
            key=lambda item: item.result["barrier_ev"],
        )
        best_compromise = min(
            valid,
            key=lambda item: joint(item.result),
        )

        return {
            "generation": generation,
            "mode": search_mode,
            "stall_count": stall_count,
            "burst_remaining": burst_remaining,
            "joint_score": current_best_joint_score(),
            "valid": len(valid),
            "invalid": len(all_individuals) - len(valid),
            "archive": len(archive),
            "best_dz_um": (
                best_dz.result["dz_peak_m"] * 1e6
            ),
            "U_at_best_dz_mev": (
                best_dz.result["barrier_ev"] * 1e3
            ),
            "best_U_mev": (
                best_u.result["barrier_ev"] * 1e3
            ),
            "dz_at_best_U_um": (
                best_u.result["dz_peak_m"] * 1e6
            ),
            "best_joint_dz_um": (
                best_compromise.result["dz_peak_m"]
                * 1e6
            ),
            "best_joint_U_mev": (
                best_compromise.result["barrier_ev"]
                * 1e3
            ),
            "unique_evaluations": len(cache),
            "rejects": "; ".join(
                f"{key}:{value}"
                for key, value
                in rejected.most_common(4)
            ),
        }

    update_archive(
        [
            item
            for island in islands
            for item in island.population
        ]
    )

    history.append(summary(0))

    save_csv(
        elite_rows(islands, cfg),
        cfg.output_dir
        / "island_elites"
        / "elites_gen_0000.csv",
    )

    print(
        "G000",
        history[-1],
        flush=True,
    )

    for generation in range(
        1,
        cfg.generations + 1,
    ):
        burst_active = burst_remaining > 0
        search_mode = (
            "BURST"
            if burst_active
            else "NORMAL"
        )

        next_islands: list[Island] = []

        with tqdm(
            total=cfg.islands * cfg.offspring,
            desc=(
                f"Generation {generation} "
                f"[{search_mode}]"
            ),
            unit="candidate",
        ) as progress:
            for island_id, island in enumerate(islands):
                rank_and_crowding(island.population)

                elite = sorted(
                    island.population,
                    key=lambda item: (
                        item.rank,
                        -item.crowding,
                        style_score(
                            item,
                            island.style,
                        ),
                    ),
                )[:max(2, min(5, len(island.population)))]

                children: list[Individual] = []

                while len(children) < cfg.offspring:
                    random_value = island.rng.random()

                    if not burst_active:
                        if random_value < 0.60:
                            parent = (
                                elite[0]
                                if len(elite) == 1
                                or island.rng.random() < 0.80
                                else elite[1]
                            )

                            genome = parent.genome.copy()

                            origin = (
                                "top1_direct_mutation"
                                if parent is elite[0]
                                else "top2_direct_mutation"
                            )

                        elif random_value < 0.88:
                            left = tournament(
                                island.population,
                                island.style,
                                island.rng,
                            )
                            right = tournament(
                                island.population,
                                island.style,
                                island.rng,
                            )

                            genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )
                            origin = "pareto_block_crossover"

                        elif random_value < 0.96:
                            left = elite[
                                int(
                                    island.rng.integers(
                                        len(elite)
                                    )
                                )
                            ]
                            right = island.population[
                                int(
                                    island.rng.integers(
                                        len(island.population)
                                    )
                                )
                            ]

                            genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )
                            origin = (
                                "elite_random_crossover"
                            )

                        else:
                            if (
                                island.style.family == "special"
                            ):
                                seed_name, genome = (
                                    special_feature_seed(
                                        island.rng
                                    )
                                )
                            else:
                                seed_name, genome = (
                                    ordinary_seed_list[
                                        int(
                                            island.rng.integers(
                                                len(
                                                    ordinary_seed_list
                                                )
                                            )
                                        )
                                    ]
                                )

                            origin = (
                                "normal_family_restart_"
                                + seed_name
                            )

                    else:
                        if random_value < 0.18:
                            parent = elite[
                                int(
                                    island.rng.integers(
                                        min(
                                            2,
                                            len(elite),
                                        )
                                    )
                                )
                            ]

                            genome = parent.genome.copy()
                            origin = "burst_elite_mutation"

                        elif random_value < 0.46:
                            left = tournament(
                                island.population,
                                island.style,
                                island.rng,
                            )
                            right = tournament(
                                island.population,
                                island.style,
                                island.rng,
                            )

                            genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )
                            origin = (
                                "burst_pareto_crossover"
                            )

                        elif random_value < 0.72:
                            # Cross with a live candidate from another island.
                            donor_island = islands[
                                int(
                                    island.rng.integers(
                                        len(islands)
                                    )
                                )
                            ]
                            donor = donor_island.population[
                                int(
                                    island.rng.integers(
                                        len(
                                            donor_island.population
                                        )
                                    )
                                )
                            ]
                            parent = elite[
                                int(
                                    island.rng.integers(
                                        len(elite)
                                    )
                                )
                            ]

                            genome = block_crossover(
                                parent.genome,
                                donor.genome,
                                island.rng,
                            )
                            origin = (
                                "burst_cross_island_mixing"
                            )

                        else:
                            # During burst, deliberately alternate ordinary
                            # and special feature seeds. This is the intended
                            # ordinary <-> special diversity injection.
                            if island.rng.random() < 0.50:
                                seed_name, genome = (
                                    special_feature_seed(
                                        island.rng
                                    )
                                )
                            else:
                                seed_name, genome = (
                                    ordinary_seed_list[
                                        int(
                                            island.rng.integers(
                                                len(
                                                    ordinary_seed_list
                                                )
                                            )
                                        )
                                    ]
                                )

                            genome = repair(
                                genome
                                + island.rng.normal(
                                    0.0,
                                    (HIGH - LOW) * 0.10,
                                )
                            )
                            origin = (
                                "burst_mixed_restart_"
                                + seed_name
                            )
                    # Special islands must remain in the restricted,
                    # connected 07c--07g contour family even if the genome
                    # came from ordinary crossover, migration, or burst.
                    if island.style.family == "special":
                        genome = special_07_repair(genome)

                    genome = mutate(
                        genome,
                        island.style,
                        island.rng,
                        generation / cfg.generations,
                        burst=burst_active,
                        burst_mutation_multiplier=(
                            cfg.burst_mutation_multiplier
                        ),
                    )

                    # Hard safety projection after mutation, immediately
                    # before geometry/BEM evaluation.
                    if island.style.family == "special":
                        genome = special_07_repair(genome)

                    children.append(
                        Individual(
                            genome=genome,
                            result=coarse(
                                genome,
                                progress,
                            ),
                            island=island_id,
                            generation=generation,
                            origin=origin,
                            family=island.style.family,
                        )
                    )

                    progress.update()

                next_islands.append(
                    Island(
                        style=island.style,
                        rng=island.rng,
                        population=survivors(
                            island.population + children,
                            cfg.population,
                        ),
                    )
                )

        islands = next_islands

        migration_now = (
            len(islands) > 1
            and (
                burst_active
                or (
                    cfg.migration_interval > 0
                    and generation
                    % cfg.migration_interval == 0
                )
            )
        )

        if migration_now:
            migrant_count = (
                min(12, cfg.population)
                if burst_active
                else min(
                    cfg.migrants,
                    cfg.population,
                )
            )

            outgoing: list[list[Individual]] = []

            for island in islands:
                rank_and_crowding(island.population)

                selected = sorted(
                    island.population,
                    key=lambda item: (
                        item.rank,
                        -item.crowding,
                        style_score(
                            item,
                            island.style,
                        ),
                    ),
                )[:migrant_count]

                outgoing.append(
                    [
                        item.copy()
                        for item in selected
                    ]
                )

            # One ring through ALL islands. Ordinary and special islands have
            # the same genome, so their elites migrate and mate naturally.
            for source, migrants in enumerate(outgoing):
                destination = (
                    source + 1
                ) % len(islands)

                target = islands[destination]

                translated = [
                    Individual(
                        genome=item.genome.copy(),
                        result=dict(item.result),
                        island=destination,
                        generation=generation,
                        origin=(
                            "migrant_from_"
                            f"island_{source:02d}_"
                            f"{item.family}"
                        ),
                        family=target.style.family,
                        rank=item.rank,
                        crowding=item.crowding,
                    )
                    for item in migrants
                ]

                islands[destination] = Island(
                    style=target.style,
                    rng=target.rng,
                    population=survivors(
                        target.population + translated,
                        cfg.population,
                    ),
                )

        joint_now = current_best_joint_score()

        improved = (
            joint_now
            < best_joint_seen
            * (
                1.0
                - cfg.improvement_fraction
            )
        )

        if improved:
            best_joint_seen = joint_now
            stall_count = 0

            if burst_active:
                burst_remaining = 0
        else:
            stall_count += 1

        if burst_active and burst_remaining > 0:
            burst_remaining -= 1

        if (
            not burst_active
            and burst_remaining == 0
            and stall_count >= cfg.stall_generations
        ):
            burst_remaining = cfg.burst_generations
            stall_count = 0

            print(
                "Stagnation burst scheduled: "
                f"{cfg.burst_generations} generation(s), "
                f"starting at G{generation + 1:03d}.",
                flush=True,
            )

        update_archive(
            [
                item
                for island in islands
                for item in island.population
            ]
        )

        history.append(summary(generation))

        print(
            f"G{generation:03d}",
            history[-1],
            flush=True,
        )

        save_csv(
            history,
            cfg.output_dir / "history.csv",
        )

        save_csv(
            rejection_rows,
            cfg.output_dir
            / "rejection_summary.csv",
        )

        save_csv(
            elite_rows(islands, cfg),
            cfg.output_dir
            / "island_elites"
            / f"elites_gen_{generation:04d}.csv",
        )

        save_csv(
            [
                serialize(item)
                for island in islands
                for item in island.population
            ],
            cfg.output_dir
            / f"population_gen_{generation:04d}.csv",
        )

    animate_surfaces(cfg)

    selected_candidates = (
        sorted(
            archive,
            key=lambda item: item.result["dz_peak_m"],
        )[:cfg.fine_top]
        + sorted(
            archive,
            key=lambda item: item.result["barrier_ev"],
        )[:cfg.fine_top]
        + sorted(
            archive,
            key=lambda item: joint(item.result),
        )[:cfg.fine_top]
    )

    unique_candidates: dict[
        tuple[float, ...],
        Individual,
    ] = {
        tuple(
            np.round(candidate.genome, 12)
        ): candidate
        for candidate in selected_candidates
    }

    selected = list(unique_candidates.values())

    fine_rows: list[dict[str, Any]] = []

    for candidate_number, candidate in enumerate(
        tqdm(
            selected,
            desc="Fine validation",
            unit="candidate",
        ),
        start=1,
    ):
        result, data = evaluate(
            candidate.genome,
            cfg,
            stage="fine",
            retain=True,
        )

        row = {
            "candidate": candidate_number,
            **serialize(candidate),
            **{
                f"fine_{key}": value
                for key, value in result.items()
            },
        }

        fine_rows.append(row)

        if data is not None:
            directory = (
                cfg.output_dir
                / "fine"
                / f"candidate_{candidate_number:03d}"
            )

            title = (
                f"Fine candidate {candidate_number} | "
                f"family={candidate.family} | "
                f"origin={candidate.origin}"
            )

            fine_plots(
                data,
                result,
                directory,
                title,
                cfg,
            )

            (
                directory / "full_candidate.json"
            ).write_text(
                json.dumps(
                    row,
                    indent=2,
                    default=float,
                ),
                encoding="utf-8",
            )

    save_csv(
        fine_rows,
        cfg.output_dir / "fine_validation.csv",
    )

    valid_rows = [
        row
        for row in fine_rows
        if row["fine_valid"]
    ]

    verified_rows = [
        row
        for row in valid_rows
        if row["fine_verified"]
    ]

    candidates = (
        verified_rows
        or valid_rows
    )

    top5 = sorted(
        candidates,
        key=lambda row: max(
            row["fine_dz_peak_m"] / 3e-6,
            row["fine_barrier_ev"] / 1e-3,
        ),
    )[:5]

    (
        cfg.output_dir
        / "top5_lithography_candidates.json"
    ).write_text(
        json.dumps(
            top5,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    save_csv(
        top5,
        cfg.output_dir
        / "top5_lithography_candidates.csv",
    )

    (
        cfg.output_dir / "verified_hits.json"
    ).write_text(
        json.dumps(
            verified_rows,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print(
        f"Fine verified hits: {len(verified_rows)} | "
        "top-5 design records: "
        f"{cfg.output_dir / 'top5_lithography_candidates.json'}"
    )


if __name__ == "__main__":
    main()