"""Workflow 09d: mixed ordinary/special island GA for a planar C4v X-junction.

All islands use the same full movable-knot genome and the same parameter /
mask / BEM pipeline:

    genome -> parameters(genome) -> XJunctionParameters
           -> baseline_x_junction_rf_mask
           -> build_geometry_aware_quadtree_x_junction_bem
           -> FixedMeshBEM -> RF-null trace -> objectives.

Ordinary islands evolve the full movable-knot genome.

Special islands use the same full genome, but every candidate is projected by
special_07_repair().  This fixes the knots and all global template parameters
and leaves only these physical controls active:

    d_in(60 um), d_in(90 um), d_in(120 um), common a_out.

The resulting special geometry is still the normal connected C4v RF cross
defined by baseline_x_junction_rf_mask().  It is not a separate topology and
does not use a separate genome.

PowerShell smoke test:
    python workflows/11_adaptive_mesh.py --smoke --fast

Usual run:
    python workflows/09d_mixed_island_ga.py --backend auto
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
from core.geometry.junction_templates import (
    baseline_x_junction_rf_mask,
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


Family = Literal["ordinary", "special"]

# =============================================================================
# One shared full genome
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

# gap logits | seven inner offsets | seven outer offsets | global parameters
GAPS = slice(0, N_INTERVALS)
INNER = slice(GAPS.stop, GAPS.stop + N_KNOTS - 2)
OUTER = slice(INNER.stop, INNER.stop + N_KNOTS - 2)

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

# =============================================================================
# Special 07c--07g family: same genome, fixed projection
# =============================================================================

SPECIAL_LOCK_M = 150e-6

SPECIAL_INNER_SHIFT_M = -25e-6
SPECIAL_OUTER_SHIFT_M = 20e-6
SPECIAL_RF_START_M = 30e-6
SPECIAL_TAPER_LENGTH_M = 150e-6
SPECIAL_TAPER_POWER = 2.0

SPECIAL_BULGE_M = 0.0
SPECIAL_BULGE_CENTER_M = 70e-6
SPECIAL_BULGE_WIDTH_M = 25e-6

# The shared knots are:
# [0, 7, 15, 27, 43, 65, 92, 122, 150] um
#
# The seven interior controls correspond to:
# [7, 15, 27, 43, 65, 92, 122] um.
#
# We use 65, 92 and 122 um as the three physical special inner controls.
# They are called d_in(60), d_in(90), d_in(120) in the special-family
# nomenclature, while the actual controllable knots remain the shared full
# workflow-09 knots above.
SPECIAL_INNER_INDICES = np.array([4, 5, 6], dtype=int)

SPECIAL_INNER_MIN_M = -18e-6
SPECIAL_INNER_MAX_M = 4e-6

SPECIAL_OUTER_MIN_M = -10e-6
SPECIAL_OUTER_MAX_M = 10e-6


# =============================================================================
# Data structures
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
    family: Family


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
            return np.array([np.inf, np.inf], dtype=float)

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


class BEMField:
    """Adapter from FixedMeshBEM to the RF-null tracing interface."""

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
# Genome decode / construction
# =============================================================================

def repair(genome: np.ndarray) -> np.ndarray:
    """Validate and clip one shared full genome."""
    values = np.asarray(genome, dtype=float).copy()

    if values.shape != (N_GENES,):
        raise ValueError(
            f"Expected genome shape {(N_GENES,)}, got {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Genome contains non-finite values.")

    return np.clip(values, LOW, HIGH)


def decode(
    genome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode movable knots and the two piecewise-linear contour profiles."""
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

    return knots_m, inner_offsets_m, outer_offsets_m


def baseline() -> np.ndarray:
    """Conventional connected open-centre C4v cross as a full genome."""
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
    """Exact conventional cross: all movable contour offsets are zero."""
    return baseline()


def parameters(genome: np.ndarray):
    """Create the standard XJunctionParameters object from the full genome."""
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
        inner_edge_shift_at_centre_m=float(values[CENTER_IN]),
        outer_edge_shift_at_centre_m=float(values[CENTER_OUT]),
        rf_start_radius_override_m=float(values[START]),
        taper_length_m=float(values[LENGTH]),
        taper_power=float(values[POWER]),
        outer_bulge_amplitude_m=float(values[BULGE]),
        outer_bulge_center_m=float(values[BULGE_CENTER]),
        outer_bulge_sigma_m=float(values[BULGE_WIDTH]),
    )


# =============================================================================
# Special-family projection: still the same full genome
# =============================================================================

def special_07_repair(genome: np.ndarray) -> np.ndarray:
    """Project a full genome onto the constrained connected 07c--07g family.

    The returned vector remains a normal full N_GENES movable-knot genome:
    it can be serialized, passed to parameters(), placed in the common archive,
    crossed with ordinary candidates and evaluated by the normal BEM pipeline.

    Fixed:
    - shared knot locations;
    - lock;
    - central inner/outer shifts;
    - RF start;
    - taper length and taper power;
    - Gaussian outer bulge.

    Active:
    - three selected inner contour controls;
    - one common outer amplitude copied to the same selected controls.
    """
    source = repair(genome)
    child = baseline()

    # Fixed longitudinal knots and core extent.
    child[LOCK] = SPECIAL_LOCK_M
    child[GAPS] = fixed_knots_to_logits(
        FIXED_CORE_KNOTS_M,
        min_gap_m=MIN_GAP_M,
    )

    # Fixed continuous baseline-X template parameters.
    child[CENTER_IN] = SPECIAL_INNER_SHIFT_M
    child[CENTER_OUT] = SPECIAL_OUTER_SHIFT_M
    child[START] = SPECIAL_RF_START_M
    child[LENGTH] = SPECIAL_TAPER_LENGTH_M
    child[POWER] = SPECIAL_TAPER_POWER
    child[BULGE] = SPECIAL_BULGE_M
    child[BULGE_CENTER] = SPECIAL_BULGE_CENTER_M
    child[BULGE_WIDTH] = SPECIAL_BULGE_WIDTH_M

    # Keep only the designated inner controls.
    child[INNER] = 0.0
    child[INNER][SPECIAL_INNER_INDICES] = np.clip(
        source[INNER][SPECIAL_INNER_INDICES],
        SPECIAL_INNER_MIN_M,
        SPECIAL_INNER_MAX_M,
    )

    # One physical a_out shared by the permitted outer controls.
    a_out_m = float(
        np.clip(
            np.mean(source[OUTER][SPECIAL_INNER_INDICES]),
            SPECIAL_OUTER_MIN_M,
            SPECIAL_OUTER_MAX_M,
        )
    )

    child[OUTER] = 0.0
    child[OUTER][SPECIAL_INNER_INDICES] = a_out_m

    return repair(child)


def special_feature_seed(
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    """Random full-genome seed projected to the connected special family."""
    genome = baseline()

    mode = int(rng.integers(0, 5))

    if mode == 0:
        d90_um = float(rng.uniform(-16.0, -11.0))
        d60_um = float(rng.uniform(-0.75 * abs(d90_um), -0.40 * abs(d90_um)))
        d120_um = float(rng.uniform(-0.75 * abs(d90_um), -0.40 * abs(d90_um)))
        a_out_um = 0.0
        label = "special_triangle"

    elif mode == 1:
        level_um = float(rng.uniform(-13.0, -8.0))
        d60_um = float(level_um + rng.normal(0.0, 1.2))
        d90_um = float(level_um + rng.normal(0.0, 1.2))
        d120_um = float(level_um + rng.normal(0.0, 1.2))
        a_out_um = 0.0
        label = "special_broad_inner"

    elif mode == 2:
        d60_um = float(rng.uniform(-15.0, -9.0))
        d90_um = float(rng.uniform(-14.0, -9.0))
        d120_um = float(rng.uniform(-12.0, -7.0))
        a_out_um = float(rng.uniform(-10.0, 10.0))
        label = "special_coupled"

    elif mode == 3:
        d60_um = float(rng.uniform(-14.0, -8.0))
        d90_um = float(rng.uniform(-12.0, -8.0))
        d120_um = float(rng.uniform(-11.0, -6.0))
        a_out_um = float(rng.uniform(-10.0, -3.0))
        label = "special_outer_notch"

    else:
        d60_um = float(rng.uniform(-14.0, -8.0))
        d90_um = float(rng.uniform(-15.0, -9.0))
        d120_um = float(rng.uniform(-11.0, -6.0))
        a_out_um = float(rng.uniform(3.0, 10.0))
        label = "special_outer_bulge"

    genome[INNER] = 0.0
    genome[INNER][SPECIAL_INNER_INDICES] = (
        np.clip(
            [d60_um, d90_um, d120_um],
            SPECIAL_INNER_MIN_M * 1e6,
            SPECIAL_INNER_MAX_M * 1e6,
        )
        * 1e-6
    )

    genome[OUTER] = 0.0
    genome[OUTER][SPECIAL_INNER_INDICES] = (
        np.clip(
            a_out_um,
            SPECIAL_OUTER_MIN_M * 1e6,
            SPECIAL_OUTER_MAX_M * 1e6,
        )
        * 1e-6
    )

    return label, special_07_repair(genome)


def ordinary_seeds() -> list[tuple[str, np.ndarray]]:
    """Established ordinary full-genome seeds."""
    base = baseline()

    inner = base.copy()
    inner[INNER] = (
        np.array([-4, -8, -12, -12, -8, -4, -2], dtype=float)
        * 1e-6
    )

    outer = base.copy()
    outer[OUTER] = (
        np.array([2, 5, 9, 11, 8, 4, 2], dtype=float)
        * 1e-6
    )

    coupled = base.copy()
    coupled[INNER] = (
        np.array([-3, -7, -11, -13, -9, -5, -2], dtype=float)
        * 1e-6
    )
    coupled[OUTER] = (
        np.array([2, 5, 10, 12, 8, 4, 1], dtype=float)
        * 1e-6
    )

    packed = base.copy()
    packed[GAPS] = np.array(
        [1.4, 1.0, 0.6, 0.1, -0.3, -0.7, -1.0, -1.2],
        dtype=float,
    )
    packed[LOCK] = 130e-6

    return [
        ("ordinary_conventional_exact", conventional_cross()),
        ("ordinary_inner", repair(inner)),
        ("ordinary_outer", repair(outer)),
        ("ordinary_coupled", repair(coupled)),
        ("ordinary_packed_x", repair(packed)),
    ]


# =============================================================================
# Geometry / BEM evaluation
# =============================================================================

def rejected(
    reason: str,
    stage: str,
) -> dict[str, Any]:
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
    parameter_object: Any,
    *,
    fast: bool,
):
    """Build BEM model from the project's baseline connected RF mask."""
    if fast:
        return build_geometry_aware_quadtree_x_junction_bem(
            parameter_object,
            central_half_extent_m=145e-6,
            central_max_cell_m=46e-6,
            boundary_max_cell_m=17e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=8e-6,
        )

    return build_geometry_aware_quadtree_x_junction_bem(
        parameter_object,
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
):
    """Run geometry checks, fixed-mesh BEM, RF-null trace and objectives."""
    values = repair(genome)

    try:
        parameter_object = parameters(values)

        report = check_x_junction_manufacturability(parameter_object)

        if not report.valid:
            return (
                rejected(
                    "geometry: " + "; ".join(map(str, report.messages)),
                    stage,
                ),
                None,
            )

        longitudinal_m = np.linspace(
            0.0,
            parameter_object.arm_length_m,
            2001,
        )

        inner_m, outer_m = parameter_object.rail_boundaries_m(longitudinal_m)

        if np.min(inner_m) <= 0.5e-6:
            return rejected("inner clearance", stage), None

        if np.min(outer_m - inner_m) < cfg.rail_min_width_m:
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

        charge = bem.solve_masks(model.bem.electrode_voltages_v)
        field = BEMField(bem, charge)

        trace_points = (
            cfg.fine_points
            if stage == "fine"
            else cfg.coarse_points
        )

        trace = trace_rf_transverse_minimum(
            field,
            np.linspace(-350e-6, 350e-6, trace_points),
            initial_y_m=0.0,
            initial_z_m=cfg.target_z_m,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )

        converged = np.asarray(trace.converged, dtype=bool)
        z_m = np.asarray(trace.z_m, dtype=float)
        y_m = np.asarray(trace.y_m, dtype=float)

        if (
            not trace.valid
            or converged.shape != (trace_points,)
            or not np.all(converged)
        ):
            return rejected("incomplete trace", stage), None

        if not np.all(np.isfinite(z_m)) or not np.all(np.isfinite(y_m)):
            return rejected("nonfinite trace", stage), None

        pseudo_ev = np.asarray(
            pseudopotential_profile_ev(
                field,
                trace,
                rf_voltage_peak_v=cfg.rf_peak_v,
                rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
            ),
            dtype=float,
        )

        barrier = compute_barrier_metrics(
            pseudo_ev,
            trace,
        )

        if (
            not barrier.valid
            or pseudo_ev.shape != (trace_points,)
            or not np.all(np.isfinite(pseudo_ev))
        ):
            return rejected("invalid pseudopotential", stage), None

        dz_m = z_m - cfg.target_z_m

        dz_peak_m = float(np.max(np.abs(dz_m)))
        dz_rms_m = float(np.sqrt(np.mean(dz_m**2)))
        lateral_peak_m = float(np.max(np.abs(y_m)))
        barrier_ev = float(barrier.barrier_height_ev)

        if not np.all(
            np.isfinite(
                [dz_peak_m, dz_rms_m, lateral_peak_m, barrier_ev]
            )
        ):
            return rejected("nonfinite metrics", stage), None

        result = {
            "valid": True,
            "verified": bool(
                stage == "fine"
                and dz_peak_m <= cfg.dz_limit_m
                and barrier_ev <= cfg.barrier_limit_ev
                and lateral_peak_m <= cfg.lateral_limit_m
            ),
            "reason": "ok",
            "stage": stage,
            "backend": bem.backend_name,
            "dz_peak_m": dz_peak_m,
            "dz_rms_m": dz_rms_m,
            "barrier_ev": barrier_ev,
            "lateral_peak_m": lateral_peak_m,
            "barrier_reference_ev": float(barrier.reference_energy_ev),
            "panels": int(model.n_panels),
            "points": int(trace_points),
            "converged": int(np.sum(converged)),
        }

        retained = (
            (model, trace, pseudo_ev, field, parameter_object)
            if retain
            else None
        )

        return result, retained

    except Exception as error:
        if (
            cfg.backend == "cuda"
            and "out of memory" in str(error).lower()
        ):
            raise RuntimeError(
                "CUDA OOM. Reduce --max-panels or use --fast."
            ) from error

        return (
            rejected(
                "geometry/BEM error "
                f"{type(error).__name__}: "
                f"{str(error).replace(chr(10), ' ').strip()}",
                stage,
            ),
            None,
        )


# =============================================================================
# NSGA-II
# =============================================================================

def dominates(
    left: Individual,
    right: Individual,
) -> bool:
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
    n = len(population)

    defeated: list[list[int]] = [[] for _ in population]
    losses = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for left_index in range(n):
        for right_index in range(left_index + 1, n):
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
        for index in range(n)
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

    for level, front in enumerate(fronts):
        for index in front:
            population[index].rank = level
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

        for objective_axis in range(2):
            ordered = sorted(
                valid,
                key=lambda index: population[index].objectives[objective_axis],
            )

            low = population[ordered[0]].objectives[objective_axis]
            high = population[ordered[-1]].objectives[objective_axis]

            if high <= low:
                continue

            population[ordered[0]].crowding = np.inf
            population[ordered[-1]].crowding = np.inf

            for position in range(1, len(ordered) - 1):
                index = ordered[position]

                if np.isfinite(population[index].crowding):
                    before = population[
                        ordered[position - 1]
                    ].objectives[objective_axis]

                    after = population[
                        ordered[position + 1]
                    ].objectives[objective_axis]

                    population[index].crowding += (
                        (after - before) / (high - low)
                    )

    return fronts


def survivors(
    pool: list[Individual],
    size: int,
) -> list[Individual]:
    selected: list[Individual] = []

    for front in rank_and_crowding(pool):
        ordered = sorted(
            front,
            key=lambda index: pool[index].crowding,
            reverse=True,
        )

        selected.extend(
            pool[index].copy()
            for index in ordered[: size - len(selected)]
        )

        if len(selected) == size:
            break

    return selected


def score(
    individual: Individual,
    style: IslandStyle,
) -> float:
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
    first, second = (
        population[int(index)]
        for index in rng.integers(0, len(population), 2)
    )

    if first.valid != second.valid:
        return first if first.valid else second

    if first.rank != second.rank:
        return first if first.rank < second.rank else second

    if first.crowding != second.crowding:
        return first if first.crowding > second.crowding else second

    return (
        first
        if score(first, style) <= score(second, style)
        else second
    )


# =============================================================================
# Genetic operators
# =============================================================================

def anneal(
    style: IslandStyle,
    fraction: float,
) -> float:
    if fraction <= 0.75:
        return style.explore

    q = (fraction - 0.75) / 0.25

    return (
        style.final
        + 0.5
        * (style.explore - style.final)
        * (1.0 + math.cos(math.pi * q))
    )


def ordinary_sigmas(
    style: IslandStyle,
    fraction: float,
) -> np.ndarray:
    scale = anneal(style, fraction)

    sigma = np.zeros(N_GENES, dtype=float)

    sigma[GAPS] = 0.34 * scale
    sigma[INNER] = 8e-6 * scale
    sigma[OUTER] = 8e-6 * scale

    sigma[LOCK] = 22e-6 * scale
    sigma[CENTER_IN:CENTER_OUT + 1] = 9e-6 * scale

    sigma[START] = 10e-6 * scale
    sigma[LENGTH] = 24e-6 * scale
    sigma[POWER] = 0.35 * scale

    sigma[BULGE] = 10e-6 * scale
    sigma[BULGE_CENTER] = 18e-6 * scale
    sigma[BULGE_WIDTH] = 12e-6 * scale

    if style.emphasis == "inner":
        sigma[INNER] *= 1.8

    elif style.emphasis == "outer":
        sigma[OUTER] *= 1.8

    elif style.emphasis == "x":
        sigma[GAPS] *= 2.3
        sigma[LOCK] *= 1.7

    elif style.emphasis == "inner_x":
        sigma[INNER] *= 1.7
        sigma[GAPS] *= 1.5

    elif style.emphasis == "outer_x":
        sigma[OUTER] *= 1.7
        sigma[GAPS] *= 1.5

    elif style.emphasis == "central":
        sigma[CENTER_IN:START + 1] *= 2.0

    elif style.emphasis == "wide":
        sigma *= 1.35

    elif style.emphasis == "local":
        sigma *= 0.55

    return sigma


def mutate_special_07_family(
    genome: np.ndarray,
    style: IslandStyle,
    rng: np.random.Generator,
    fraction: float,
    *,
    burst: bool,
    burst_mutation_multiplier: float,
) -> np.ndarray:
    """Mutate only the active special controls, then re-project."""
    child = special_07_repair(genome)

    scale = anneal(style, fraction)

    inner_sigma_m = 2.0e-6 * scale
    outer_sigma_m = 1.5e-6 * scale
    probability = 0.58

    if burst:
        inner_sigma_m *= burst_mutation_multiplier
        outer_sigma_m *= burst_mutation_multiplier
        probability = 0.90

    inner_values_m = child[INNER][SPECIAL_INNER_INDICES].copy()

    active = rng.random(len(SPECIAL_INNER_INDICES)) < probability

    if np.any(active):
        inner_values_m[active] += rng.normal(
            0.0,
            inner_sigma_m,
            size=int(np.sum(active)),
        )

    child[INNER][SPECIAL_INNER_INDICES] = np.clip(
        inner_values_m,
        SPECIAL_INNER_MIN_M,
        SPECIAL_INNER_MAX_M,
    )

    a_out_m = float(
        np.mean(child[OUTER][SPECIAL_INNER_INDICES])
    )

    if rng.random() < 0.80 * probability:
        a_out_m += float(rng.normal(0.0, outer_sigma_m))

    a_out_m = float(
        np.clip(
            a_out_m,
            SPECIAL_OUTER_MIN_M,
            SPECIAL_OUTER_MAX_M,
        )
    )

    child[OUTER] = 0.0
    child[OUTER][SPECIAL_INNER_INDICES] = a_out_m

    return special_07_repair(child)


def mutate(
    genome: np.ndarray,
    style: IslandStyle,
    rng: np.random.Generator,
    fraction: float,
    *,
    burst: bool = False,
    burst_mutation_multiplier: float = 1.0,
) -> np.ndarray:
    if style.family == "special":
        return mutate_special_07_family(
            genome,
            style,
            rng,
            fraction,
            burst=burst,
            burst_mutation_multiplier=burst_mutation_multiplier,
        )

    child = repair(genome)
    sigma = ordinary_sigmas(style, fraction)

    if burst:
        sigma *= burst_mutation_multiplier

    chance = np.full(N_GENES, 0.55)

    if style.emphasis in ("x", "inner_x", "outer_x"):
        chance[GAPS] = 0.88

    if style.emphasis == "inner":
        chance[INNER] = 0.88

    if style.emphasis == "outer":
        chance[OUTER] = 0.88

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


def block_crossover(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block crossover remains valid because all candidates share N_GENES."""
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
# Island styles
# =============================================================================

def make_styles(
    ordinary_count: int,
    special_count: int,
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
        for index in range(ordinary_count)
    ]

    special = [
        special_templates[index % len(special_templates)]
        for index in range(special_count)
    ]

    return ordinary + special


# =============================================================================
# Serialization / reports
# =============================================================================

def serialize(
    individual: Individual,
) -> dict[str, Any]:
    knots_m, inner_m, outer_m = decode(individual.genome)

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


def load_seed_genome(
    path: Path,
    *,
    objective: str = "barrier",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the best saved full genome from a previous workflow report.

    The returned vector is still the standard N_GENES genome. It can be
    seeded directly into an ordinary island or projected to a special island.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Seed report not found: {path.resolve()}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        candidates = [payload]
    elif isinstance(payload, list):
        candidates = payload
    else:
        raise ValueError(
            "Expected a JSON object or a list of candidate records."
        )

    if not candidates:
        raise ValueError(f"No candidates in {path}")

    def value(row: dict[str, Any], key: str) -> float:
        raw = row.get(key, np.inf)

        try:
            result = float(raw)
        except (TypeError, ValueError):
            return np.inf

        return result if np.isfinite(result) else np.inf

    if objective == "barrier":
        # Fine result has priority. Fall back to coarse if this is an
        # older report without fine_ fields.
        candidates.sort(
            key=lambda row: (
                value(
                    row,
                    "fine_barrier_ev",
                )
                if "fine_barrier_ev" in row
                else value(row, "barrier_ev"),
                value(
                    row,
                    "fine_dz_peak_m",
                )
                if "fine_dz_peak_m" in row
                else value(row, "dz_peak_m"),
            )
        )

    elif objective == "joint":
        candidates.sort(
            key=lambda row: max(
                (
                    value(row, "fine_dz_peak_m")
                    if "fine_dz_peak_m" in row
                    else value(row, "dz_peak_m")
                )
                / 3e-6,
                (
                    value(row, "fine_barrier_ev")
                    if "fine_barrier_ev" in row
                    else value(row, "barrier_ev")
                )
                / 1e-3,
            )
        )

    else:
        raise ValueError(
            f"Unknown seed objective: {objective!r}"
        )

    best = candidates[0]

    raw_genome = best.get("genome")

    if raw_genome is None:
        raise KeyError(
            "Candidate record has no 'genome' field."
        )

    if isinstance(raw_genome, str):
        raw_genome = json.loads(raw_genome)

    genome = repair(
        np.asarray(raw_genome, dtype=float)
    )

    return genome, best

def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = sorted(set().union(*(row.keys() for row in rows)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, allow_nan=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def elite_rows(
    islands: list[Island],
    cfg: Settings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for island in islands:
        rank_and_crowding(island.population)

        best = sorted(
            island.population,
            key=lambda item: (
                item.rank,
                -item.crowding,
                score(item, island.style),
            ),
        )[: cfg.retain_per_island]

        rows.extend(serialize(item) for item in best)

    return rows


# =============================================================================
# Geometry visualizations
# =============================================================================

def draw_geometry(
    axis: Any,
    row: dict[str, Any],
    label: str,
) -> None:
    raw_genome = row["genome"]

    if isinstance(raw_genome, str):
        raw_genome = json.loads(raw_genome)

    genome = repair(np.asarray(raw_genome, dtype=float))
    parameter_object = parameters(genome)

    longitudinal_m = np.linspace(0.0, 260e-6, 900)
    inner_m, outer_m = parameter_object.rail_boundaries_m(longitudinal_m)

    for theta in (
        0.0,
        np.pi / 2.0,
        np.pi,
        3.0 * np.pi / 2.0,
    ):
        cosine = np.cos(theta)
        sine = np.sin(theta)

        x_inner_m = longitudinal_m * cosine - inner_m * sine
        y_inner_m = longitudinal_m * sine + inner_m * cosine

        x_outer_m = longitudinal_m * cosine - outer_m * sine
        y_outer_m = longitudinal_m * sine + outer_m * cosine

        axis.fill(
            np.r_[x_inner_m, x_outer_m[::-1]] * 1e6,
            np.r_[y_inner_m, y_outer_m[::-1]] * 1e6,
            color="#db4f4f",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.90,
        )

    knots_m, _, _ = decode(genome)
    knot_inner_m, knot_outer_m = parameter_object.rail_boundaries_m(knots_m)

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

    axis.axhline(0.0, color="0.7", linewidth=0.35)
    axis.axvline(0.0, color="0.7", linewidth=0.35)

    axis.set(
        xlim=(-260.0, 260.0),
        ylim=(-260.0, 260.0),
        aspect="equal",
    )

    axis.set_xticks([])
    axis.set_yticks([])

    axis.set_title(
        f"{label} [{row.get('family', 'ordinary')}]\n"
        f"dz={float(row['dz_peak_m']) * 1e6:.2f} um, "
        f"U={float(row['barrier_ev']) * 1e3:.2f} meV",
        fontsize=8,
    )


def plot_mask_vs_bem(
    parameter_object: Any,
    model: Any,
    path: Path,
    *,
    title_prefix: str,
) -> None:
    """Your requested continuous-mask versus BEM-quadtree diagnostic."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    n = 600
    xs_m = np.linspace(-250e-6, 250e-6, n)
    ys_m = np.linspace(-250e-6, 250e-6, n)

    x_grid_m, y_grid_m = np.meshgrid(
        xs_m,
        ys_m,
        indexing="xy",
    )

    mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        parameter_object,
    )

    axes[0].imshow(
        mask,
        extent=[
            xs_m[0] * 1e6,
            xs_m[-1] * 1e6,
            ys_m[0] * 1e6,
            ys_m[-1] * 1e6,
        ],
        origin="lower",
        cmap="RdYlBu_r",
        aspect="equal",
    )

    axes[0].set(
        xlim=(-250, 250),
        ylim=(-250, 250),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=f"{title_prefix}: continuous RF mask (600×600)",
    )
    axes[0].grid(alpha=0.15)

    panels_m = np.asarray(model.bem.panels_m)
    rf = np.asarray(model.bem.electrode_voltages_v) > 0.5

    rectangles = [
        Rectangle(
            (panel[0] * 1e6, panel[2] * 1e6),
            (panel[1] - panel[0]) * 1e6,
            (panel[3] - panel[2]) * 1e6,
        )
        for panel in panels_m
    ]

    collection = PatchCollection(
        rectangles,
        array=rf.astype(float),
        cmap="RdYlBu_r",
        edgecolor="none",
    )

    axes[1].add_collection(collection)

    axes[1].set(
        xlim=(-250, 250),
        ylim=(-250, 250),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=f"{title_prefix}: BEM quadtree ({model.n_panels} panels)",
    )
    axes[1].grid(alpha=0.15)

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def pseudopotential_surface_ev(
    field: BEMField,
    cfg: Settings,
    *,
    extent_m: float = 230e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates_m = np.linspace(
        -extent_m,
        extent_m,
        cfg.surface_points,
    )

    x_grid_m, y_grid_m = np.meshgrid(
        coordinates_m,
        coordinates_m,
        indexing="xy",
    )

    x_flat_m = x_grid_m.ravel()
    y_flat_m = y_grid_m.ravel()
    z_flat_m = np.full_like(x_flat_m, cfg.target_z_m)

    values = field.bem.field_batch(
        x_flat_m[None, :],
        y_flat_m[None, :],
        z_flat_m[None, :],
        field.charge,
        candidate_chunk=1,
    )

    electric_field = field.bem.asnumpy(values)[0]
    electric_field_sq = np.sum(electric_field**2, axis=1)

    elementary_charge_c = 1.602176634e-19
    ca40_mass_kg = 39.96259098 * 1.66053906660e-27

    potential_ev = (
        elementary_charge_c
        * cfg.rf_peak_v**2
        * electric_field_sq
        / (
            4.0
            * ca40_mass_kg
            * RF_ANGULAR_FREQUENCY_RAD_S**2
        )
    )

    return (
        x_grid_m,
        y_grid_m,
        potential_ev.reshape(x_grid_m.shape),
    )


def fine_plots(
    data: tuple[Any, Any, np.ndarray, BEMField, Any],
    result: dict[str, Any],
    directory: Path,
    title: str,
    cfg: Settings,
) -> None:
    model, trace, pseudo_ev, field, parameter_object = data

    directory.mkdir(parents=True, exist_ok=True)

    plot_mask_vs_bem(
        parameter_object,
        model,
        directory / "mask_vs_bem.png",
        title_prefix=title,
    )

    x_grid_m, y_grid_m, surface_ev = pseudopotential_surface_ev(field, cfg)

    surface_mev = surface_ev * 1e3
    finite = surface_mev[np.isfinite(surface_mev)]

    vmax = (
        max(float(np.percentile(finite, 97.0)), 1e-6)
        if finite.size
        else 1.0
    )

    figure, axis = plt.subplots(figsize=(8.0, 7.0))

    image = axis.contourf(
        x_grid_m * 1e6,
        y_grid_m * 1e6,
        np.clip(surface_mev, 0.0, vmax),
        levels=60,
        cmap="magma",
    )

    axis.plot(
        np.asarray(trace.x_m) * 1e6,
        np.asarray(trace.y_m) * 1e6,
        color="cyan",
        linewidth=1.2,
    )

    axis.set(
        xlim=(-230.0, 230.0),
        ylim=(-230.0, 230.0),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=(
            f"{title}\n"
            f"RF pseudopotential at z={cfg.target_z_m * 1e6:.1f} µm"
        ),
    )

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
        (np.asarray(trace.z_m) - TARGET_ION_HEIGHT_M) * 1e6,
    )
    axes[0].axhline(3.0, color="r", linestyle="--")
    axes[0].axhline(-3.0, color="r", linestyle="--")
    axes[0].set_ylabel("z - target [µm]")

    axes[1].plot(
        x_um,
        (pseudo_ev - result["barrier_reference_ev"]) * 1e3,
    )
    axes[1].axhline(1.0, color="r", linestyle="--")
    axes[1].set_ylabel("RF pseudo-ref [meV]")

    axes[2].plot(
        x_um,
        np.asarray(trace.y_m) * 1e6,
    )
    axes[2].set(
        xlabel="x [µm]",
        ylabel="y [µm]",
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    figure.savefig(
        directory / "profiles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def animate_surfaces(cfg: Settings) -> None:
    """Build top-5 geometry GIFs from each saved generation's island elites."""
    try:
        import pandas as pd
    except ImportError:
        print("GIF generation skipped: pandas is not installed.")
        return

    files = sorted(
        (cfg.output_dir / "island_elites").glob("elites_gen_*.csv")
    )[:: max(1, cfg.animation_stride)]

    if not files:
        return

    frames = [
        (
            int(file.stem.rsplit("_", 1)[-1]),
            pd.read_csv(file),
        )
        for file in files
    ]

    target = cfg.output_dir / "surface_animations"
    target.mkdir(parents=True, exist_ok=True)

    for island_id in range(cfg.islands):
        choices: list[tuple[int, Any]] = []

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
                ascending=[True, False, True, True],
            ).head(5)

            choices.append((generation, subset))

        if not any(not subset.empty for _, subset in choices):
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

            for index, axis in enumerate(axes):
                if index < len(rows):
                    draw_geometry(
                        axis,
                        rows[index],
                        f"Top {index + 1}",
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

            family = rows[0]["family"] if rows else "unknown"

            figure.suptitle(
                f"Island {island_id:02d} [{family}], "
                f"top-5 connected-C4v candidates, generation {generation}",
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
                target / f"island_{island_id:02d}_top5.gif",
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
    command = argparse.ArgumentParser(description=__doc__)

    command.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    command.add_argument("--ordinary-islands", type=int, default=21)
    command.add_argument("--special-islands", type=int, default=9)

    command.add_argument("--population", type=int, default=20)
    command.add_argument("--offspring", type=int, default=8)
    command.add_argument("--generations", type=int, default=100)

    command.add_argument("--migration-interval", type=int, default=3)
    command.add_argument("--migrants", type=int, default=3)
    command.add_argument("--retain-per-island", type=int, default=5)

    command.add_argument("--coarse-points", type=int, default=61)
    command.add_argument("--fine-points", type=int, default=181)
    command.add_argument("--fine-top", type=int, default=25)
    command.add_argument("--max-panels", type=int, default=7000)

    command.add_argument("--fast", action="store_true")

    command.add_argument("--animation-stride", type=int, default=2)
    command.add_argument("--surface-points", type=int, default=101)

    command.add_argument("--stall-generations", type=int, default=5)
    command.add_argument("--burst-generations", type=int, default=3)
    command.add_argument("--improvement-fraction", type=float, default=0.01)
    command.add_argument(
        "--burst-mutation-multiplier",
        type=float,
        default=3.0,
    )

    command.add_argument("--seed", type=int, default=20260926)

    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/11_adaptive_mesh"),
    )

    command.add_argument("--smoke", action="store_true")

    command.add_argument(
    "--seed-report",
    type=Path,
    default=Path(
        "reports/11_mixed_island_ga/"
        "top5_lithography_candidates.json"
    ),
    help=(
        "Previous report containing full genomes. "
        "The best barrier candidate is injected into ordinary island 0."
    ),
    )

    command.add_argument(
        "--disable-best-seed",
        action="store_true",
        help="Do not inject the best candidate from --seed-report.",
    )
    command.add_argument(
        "--animate-bem",
        action="store_true",
        help=(
            "Create exact RF-mask versus BEM-quadtree GIFs "
            "for the best candidate of every island."
        ),
    )

    command.add_argument(
        "--animate-pseudopotential",
        action="store_true",
        help=(
            "Create an RF pseudopotential evolution GIF for "
            "the best barrier candidate."
        ),
    )

    command.add_argument(
        "--pseudopotential-animation-stride",
        type=int,
        default=10,
        help=(
            "Use every Nth saved generation for the expensive "
            "pseudopotential animation."
        ),
    )

    command.add_argument(
        "--pseudopotential-animation-points",
        type=int,
        default=61,
        help=(
            "Grid side for each pseudopotential-animation frame."
        ),
    )
    return command


# =============================================================================
# Main
# =============================================================================

def main() -> None:
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

    if (
        arguments.ordinary_islands < 1
        or arguments.special_islands < 1
    ):
        raise ValueError(
            "Both ordinary and special island counts must be positive."
        )

    if (
        arguments.population < 8
        or arguments.offspring < 1
        or arguments.generations < 1
    ):
        raise ValueError(
            "Require population >= 8 and offspring/generations >= 1."
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
        burst_mutation_multiplier=arguments.burst_mutation_multiplier,
        output_dir=arguments.output_dir,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # =========================================================================
    # Load the best candidate from the previous run.
    #
    # This is done once, before Initial BEM population construction.
    # =========================================================================

    best_seed_genome: np.ndarray | None = None
    best_seed_row: dict[str, Any] | None = None

    if not arguments.disable_best_seed:
        try:
            best_seed_genome, best_seed_row = load_seed_genome(
                arguments.seed_report,
                objective="barrier",
            )

            barrier_ev = float(
                best_seed_row.get(
                    "fine_barrier_ev",
                    best_seed_row.get("barrier_ev", np.inf),
                )
            )

            dz_peak_m = float(
                best_seed_row.get(
                    "fine_dz_peak_m",
                    best_seed_row.get("dz_peak_m", np.inf),
                )
            )

            print(
                "Loaded best prior seed: "
                f"{arguments.seed_report} | "
                f"barrier={barrier_ev * 1e3:.6f} meV | "
                f"dz={dz_peak_m * 1e6:.3f} um",
                flush=True,
            )

        except Exception as error:
            print(
                "Warning: previous best seed was not loaded: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )


    styles = make_styles(
        cfg.ordinary_islands,
        cfg.special_islands,
    )

    (cfg.output_dir / "settings.json").write_text(
        json.dumps(
            {
                "settings": asdict(cfg),
                "backend": backend.as_dict(),
                "genome_genes": N_GENES,
                "special_projection": {
                    "same_full_genome": True,
                    "fixed_knots_m": FIXED_CORE_KNOTS_M.tolist(),
                    "active_inner_indices": SPECIAL_INNER_INDICES.tolist(),
                    "active_special_controls": [
                        "d_in_60",
                        "d_in_90",
                        "d_in_120",
                        "common_a_out",
                    ],
                },
                "rf_mask": "baseline_x_junction_rf_mask",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Workflow 09d | backend={backend.selected} | "
        f"device={backend.device_name}"
    )
    print(
        f"Islands: {cfg.islands} = "
        f"{cfg.ordinary_islands} ordinary + "
        f"{cfg.special_islands} special."
    )
    print(f"All candidates use one shared {N_GENES}-gene genome.")

    master_rng = np.random.default_rng(cfg.seed)
    ordinary_seed_list = ordinary_seeds()

    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    best_height: tuple[np.ndarray, dict[str, Any]] | None = None
    best_barrier: tuple[np.ndarray, dict[str, Any]] | None = None
    best_joint: tuple[np.ndarray, dict[str, Any]] | None = None

    def joint_score(result: dict[str, Any]) -> float:
        return max(
            result["dz_peak_m"] / 10e-6,
            result["barrier_ev"] / 10e-3,
        )

    def coarse(
        genome: np.ndarray,
        progress: Any | None = None,
    ) -> dict[str, Any]:
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
                or result["dz_peak_m"] < best_height[1]["dz_peak_m"]
            ):
                best_height = (values.copy(), result.copy())

            if (
                best_barrier is None
                or result["barrier_ev"] < best_barrier[1]["barrier_ev"]
            ):
                best_barrier = (values.copy(), result.copy())

            if (
                best_joint is None
                or joint_score(result) < joint_score(best_joint[1])
            ):
                best_joint = (values.copy(), result.copy())

            (cfg.output_dir / "best_live_coarse.json").write_text(
                json.dumps(
                    {
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
                    },
                    indent=2,
                    default=float,
                ),
                encoding="utf-8",
            )

        if progress is not None and best_joint is not None:
            progress.set_postfix(
                {
                    "dz": (
                        f"{best_joint[1]['dz_peak_m'] * 1e6:.2f} um"
                    ),
                    "U": (
                        f"{best_joint[1]['barrier_ev'] * 1e3:.2f} meV"
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
            rng = np.random.default_rng(
                int(master_rng.integers(0, 2**32 - 1))
            )

            population: list[Individual] = []

            if style.family == "ordinary":
                if (
                    island_id == 0
                    and best_seed_genome is not None
                ):
                    population.append(
                        Individual(
                            genome=best_seed_genome.copy(),
                            result=coarse(
                                best_seed_genome,
                                progress,
                            ),
                            island=island_id,
                            generation=0,
                            origin="seed_from_11_mixed_island_ga_best",
                            family="ordinary",
                        )
                    )

                    progress.update()
                if island_id == 0 and len(population) < cfg.population:
                    genome = conventional_cross()

                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(genome, progress),
                            island=island_id,
                            generation=0,
                            origin="ordinary_exact_conventional_cross",
                            family="ordinary",
                        )
                    )

                    progress.update()

                for label, seed in ordinary_seed_list:
                    if len(population) >= cfg.population:
                        break

                    genome = repair(
                        seed
                        + rng.normal(
                            0.0,
                            (HIGH - LOW) * 0.012 * style.explore,
                        )
                    )

                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(genome, progress),
                            island=island_id,
                            generation=0,
                            origin=f"ordinary_seed_{label}",
                            family="ordinary",
                        )
                    )
                    progress.update()

            else:
                while len(population) < cfg.population:
                    label, genome = special_feature_seed(rng)

                    population.append(
                        Individual(
                            genome=genome,
                            result=coarse(genome, progress),
                            island=island_id,
                            generation=0,
                            origin=label,
                            family="special",
                        )
                    )
                    progress.update()

            while len(population) < cfg.population:
                if style.family == "special":
                    label, genome = special_feature_seed(rng)

                else:
                    label, seed = ordinary_seed_list[
                        int(rng.integers(len(ordinary_seed_list)))
                    ]

                    genome = repair(
                        seed
                        + rng.normal(
                            0.0,
                            (HIGH - LOW) * 0.020 * style.explore,
                        )
                    )

                population.append(
                    Individual(
                        genome=genome,
                        result=coarse(genome, progress),
                        island=island_id,
                        generation=0,
                        origin=f"initial_{label}",
                        family=style.family,
                    )
                )
                progress.update()

            rank_and_crowding(population)

            islands.append(
                Island(
                    style=style,
                    rng=rng,
                    population=population,
                )
            )

    if not any(
        individual.valid
        for island in islands
        for individual in island.population
    ):
        reasons = Counter(
            individual.result["reason"]
            for island in islands
            for individual in island.population
        )

        raise RuntimeError(
            "No valid initial geometry. "
            f"Most common rejection reasons: {reasons.most_common(20)!r}"
        )

    archive: list[Individual] = []
    history: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    best_joint_seen = np.inf
    stall_count = 0
    burst_remaining = 0
    search_mode = "NORMAL"

    def update_archive(
        new_items: list[Individual],
    ) -> None:
        nonlocal archive

        unique = {
            tuple(np.round(item.genome, 12)): item.copy()
            for item in archive + new_items
            if item.valid
        }

        archive = list(unique.values())

        if archive:
            rank_and_crowding(archive)

            archive.sort(
                key=lambda item: (
                    item.rank,
                    -item.crowding,
                )
            )

            archive = archive[:1000]

    def current_best_joint_score() -> float:
        candidates = [
            individual
            for island in islands
            for individual in island.population
            if individual.valid
        ]

        return min(
            (joint_score(individual.result) for individual in candidates),
            default=np.inf,
        )

    def summary(generation: int) -> dict[str, Any]:
        all_items = [
            individual
            for island in islands
            for individual in island.population
        ]

        valid = [
            individual
            for individual in all_items
            if individual.valid
        ]

        rejects = Counter(
            individual.result["reason"]
            for individual in all_items
            if not individual.valid
        )

        for reason, count in rejects.items():
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
                "invalid": len(all_items),
                "archive": len(archive),
                "unique_evaluations": len(cache),
                "rejects": "; ".join(
                    f"{reason}:{count}"
                    for reason, count in rejects.most_common(4)
                ),
            }

        best_dz = min(
            valid,
            key=lambda item: item.result["dz_peak_m"],
        )

        best_barrier_item = min(
            valid,
            key=lambda item: item.result["barrier_ev"],
        )

        best_compromise = min(
            valid,
            key=lambda item: joint_score(item.result),
        )

        return {
            "generation": generation,
            "mode": search_mode,
            "stall_count": stall_count,
            "burst_remaining": burst_remaining,
            "joint_score": current_best_joint_score(),
            "valid": len(valid),
            "invalid": len(all_items) - len(valid),
            "archive": len(archive),
            "best_dz_um": best_dz.result["dz_peak_m"] * 1e6,
            "best_U_mev": best_barrier_item.result["barrier_ev"] * 1e3,
            "best_joint_dz_um": (
                best_compromise.result["dz_peak_m"] * 1e6
            ),
            "best_joint_U_mev": (
                best_compromise.result["barrier_ev"] * 1e3
            ),
            "unique_evaluations": len(cache),
            "rejects": "; ".join(
                f"{reason}:{count}"
                for reason, count in rejects.most_common(4)
            ),
        }

    update_archive(
        [
            individual
            for island in islands
            for individual in island.population
        ]
    )

    history.append(summary(0))

    save_csv(
        elite_rows(islands, cfg),
        cfg.output_dir / "island_elites" / "elites_gen_0000.csv",
    )

    print("G000", history[-1], flush=True)

    for generation in range(1, cfg.generations + 1):
        burst_active = burst_remaining > 0
        search_mode = "BURST" if burst_active else "NORMAL"

        next_islands: list[Island] = []

        with tqdm(
            total=cfg.islands * cfg.offspring,
            desc=f"Generation {generation} [{search_mode}]",
            unit="candidate",
        ) as progress:
            for island_id, island in enumerate(islands):
                rank_and_crowding(island.population)

                elite = sorted(
                    island.population,
                    key=lambda item: (
                        item.rank,
                        -item.crowding,
                        score(item, island.style),
                    ),
                )[: max(2, min(5, len(island.population)))]

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

                            child_genome = parent.genome.copy()
                            origin = "top_direct_mutation"

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

                            child_genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )

                            origin = "pareto_block_crossover"

                        elif random_value < 0.96:
                            left = elite[
                                int(island.rng.integers(len(elite)))
                            ]

                            right = island.population[
                                int(
                                    island.rng.integers(
                                        len(island.population)
                                    )
                                )
                            ]

                            child_genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )

                            origin = "elite_random_crossover"

                        else:
                            if island.style.family == "special":
                                label, child_genome = special_feature_seed(
                                    island.rng
                                )

                            else:
                                label, child_genome = ordinary_seed_list[
                                    int(
                                        island.rng.integers(
                                            len(ordinary_seed_list)
                                        )
                                    )
                                ]

                            origin = f"normal_family_restart_{label}"

                    else:
                        if random_value < 0.18:
                            parent = elite[
                                int(
                                    island.rng.integers(
                                        min(2, len(elite))
                                    )
                                )
                            ]

                            child_genome = parent.genome.copy()
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

                            child_genome = block_crossover(
                                left.genome,
                                right.genome,
                                island.rng,
                            )

                            origin = "burst_pareto_crossover"

                        elif random_value < 0.72:
                            donor_island = islands[
                                int(island.rng.integers(len(islands)))
                            ]

                            donor = donor_island.population[
                                int(
                                    island.rng.integers(
                                        len(donor_island.population)
                                    )
                                )
                            ]

                            parent = elite[
                                int(island.rng.integers(len(elite)))
                            ]

                            child_genome = block_crossover(
                                parent.genome,
                                donor.genome,
                                island.rng,
                            )

                            origin = "burst_cross_island_mixing"

                        else:
                            if island.rng.random() < 0.50:
                                label, child_genome = special_feature_seed(
                                    island.rng
                                )

                            else:
                                label, child_genome = ordinary_seed_list[
                                    int(
                                        island.rng.integers(
                                            len(ordinary_seed_list)
                                        )
                                    )
                                ]

                            child_genome = repair(
                                child_genome
                                + island.rng.normal(
                                    0.0,
                                    (HIGH - LOW) * 0.08,
                                )
                            )

                            origin = f"burst_mixed_restart_{label}"

                    # The target family determines the legal subspace.
                    if island.style.family == "special":
                        child_genome = special_07_repair(child_genome)

                    child_genome = mutate(
                        child_genome,
                        island.style,
                        island.rng,
                        generation / cfg.generations,
                        burst=burst_active,
                        burst_mutation_multiplier=cfg.burst_mutation_multiplier,
                    )

                    # Final mandatory projection after crossover/mutation.
                    if island.style.family == "special":
                        child_genome = special_07_repair(child_genome)

                    children.append(
                        Individual(
                            genome=child_genome,
                            result=coarse(child_genome, progress),
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
                    and generation % cfg.migration_interval == 0
                )
            )
        )

        if migration_now:
            migrant_count = (
                min(12, cfg.population)
                if burst_active
                else min(cfg.migrants, cfg.population)
            )

            outgoing: list[list[Individual]] = []

            for island in islands:
                rank_and_crowding(island.population)

                selected = sorted(
                    island.population,
                    key=lambda item: (
                        item.rank,
                        -item.crowding,
                        score(item, island.style),
                    ),
                )[:migrant_count]

                outgoing.append([item.copy() for item in selected])

            for source, migrants in enumerate(outgoing):
                destination = (source + 1) % len(islands)
                target = islands[destination]

                incoming: list[Individual] = []

                for migrant in migrants:
                    if target.style.family == "special":
                        # Projection creates a new physical geometry.
                        # Therefore its previous BEM result is invalid.
                        projected_genome = special_07_repair(migrant.genome)
                        projected_result = coarse(projected_genome)

                        incoming.append(
                            Individual(
                                genome=projected_genome,
                                result=projected_result,
                                island=destination,
                                generation=generation,
                                origin=(
                                    f"migrant_from_{source:02d}_"
                                    f"{migrant.family}_"
                                    "projected_to_special"
                                ),
                                family="special",
                            )
                        )

                    else:
                        # Special -> ordinary and ordinary -> ordinary keep
                        # the same full genome and identical geometry.
                        incoming.append(
                            Individual(
                                genome=migrant.genome.copy(),
                                result=dict(migrant.result),
                                island=destination,
                                generation=generation,
                                origin=(
                                    f"migrant_from_{source:02d}_"
                                    f"{migrant.family}_to_ordinary"
                                ),
                                family="ordinary",
                                rank=migrant.rank,
                                crowding=migrant.crowding,
                            )
                        )

                islands[destination] = Island(
                    style=target.style,
                    rng=target.rng,
                    population=survivors(
                        target.population + incoming,
                        cfg.population,
                    ),
                )

        current_joint = current_best_joint_score()

        improved = (
            current_joint
            < best_joint_seen * (1.0 - cfg.improvement_fraction)
        )

        if improved:
            best_joint_seen = current_joint
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
                individual
                for island in islands
                for individual in island.population
            ]
        )

        history.append(summary(generation))

        print(
            f"G{generation:03d}",
            history[-1],
            flush=True,
        )

        save_csv(history, cfg.output_dir / "history.csv")

        save_csv(
            rejection_rows,
            cfg.output_dir / "rejection_summary.csv",
        )

        save_csv(
            elite_rows(islands, cfg),
            cfg.output_dir
            / "island_elites"
            / f"elites_gen_{generation:04d}.csv",
        )

        save_csv(
            [
                serialize(individual)
                for island in islands
                for individual in island.population
            ],
            cfg.output_dir / f"population_gen_{generation:04d}.csv",
        )

    animate_surfaces(cfg)

    selected = (
        sorted(
            archive,
            key=lambda item: item.result["dz_peak_m"],
        )[: cfg.fine_top]
        + sorted(
            archive,
            key=lambda item: item.result["barrier_ev"],
        )[: cfg.fine_top]
        + sorted(
            archive,
            key=lambda item: joint_score(item.result),
        )[: cfg.fine_top]
    )

    unique_selected = {
        tuple(np.round(item.genome, 12)): item
        for item in selected
    }

    fine_rows: list[dict[str, Any]] = []

    for number, item in enumerate(
        tqdm(
            list(unique_selected.values()),
            desc="Fine validation",
            unit="candidate",
        ),
        start=1,
    ):
        result, data = evaluate(
            item.genome,
            cfg,
            stage="fine",
            retain=True,
        )

        row = {
            "candidate": number,
            **serialize(item),
            **{
                "fine_" + key: value
                for key, value in result.items()
            },
        }

        fine_rows.append(row)

        if data is not None:
            directory = (
                cfg.output_dir
                / "fine"
                / f"candidate_{number:03d}"
            )

            title = (
                f"Fine candidate {number} | "
                f"family={item.family} | "
                f"origin={item.origin}"
            )

            fine_plots(
                data,
                result,
                directory,
                title,
                cfg,
            )

            (directory / "full_candidate.json").write_text(
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

    selection_pool = verified_rows or valid_rows

    top5 = sorted(
        selection_pool,
        key=lambda row: max(
            row["fine_dz_peak_m"] / 3e-6,
            row["fine_barrier_ev"] / 1e-3,
        ),
    )[:5]

    (cfg.output_dir / "top5_lithography_candidates.json").write_text(
        json.dumps(
            top5,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    save_csv(
        top5,
        cfg.output_dir / "top5_lithography_candidates.csv",
    )

    (cfg.output_dir / "verified_hits.json").write_text(
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