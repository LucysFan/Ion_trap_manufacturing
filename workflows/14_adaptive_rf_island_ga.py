"""
Workflow 14: adaptive RF-only research Island GA for a planar C4v
surface-electrode X-junction.

The optimiser uses the existing movable-knot cubic-spline geometry, the
project BEM solver and three prior elite basins A/B/C.  It does not reject
multi-minimum profiles.  It minimises four continuous RF-only quantities:

    1. absolute pseudopotential excursion relative to the arm reference;
    2. total variation integral of the pseudopotential profile;
    3. maximum longitudinal pseudopotential gradient;
    4. RF-null height excursion.

Significant extrema, transverse branch consistency and normalized RF
confinement are retained as diagnostics/constraints.  The nominal 1 meV
value is an aspiration and objective normalization scale, not a promise and
not a geometry rejection threshold.

Mesh strategy
-------------
Population screening uses one common geometry-aware quadtree policy.  Elite
candidates are re-evaluated on progressively finer policies.  Refinement
stops when the quantities of interest converge.  This is observable-driven
adaptive verification.  True per-panel residual refinement requires a
refinement API in core.geometry.mask_builder and is therefore not claimed.

Run from repository root:

    python workflows/14_adaptive_rf_island_ga.py --smoke --fast

Research run:

    python workflows/14_adaptive_rf_island_ga.py --backend cuda \
      --population 12 --offspring 5 --generations 80 \
      --fine-top 24 --adaptive-top 10 --adaptive-levels 4 \
      --max-panels 9000 --output-dir reports/14_adaptive_rf
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

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.junction_templates import baseline_x_junction_rf_mask
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.geometry.movable_knot_xjunction import (
    MovableKnotContour,
    build_movable_knot_parameters,
    fixed_knots_to_logits,
    softmax_gap_knots,
)

SeedKind = Literal[
    "A", "B", "C", "AB", "AC", "BC", "ABC", "GLOBAL",
]

Emphasis = Literal[
    "centre", "transition", "arm", "height", "barrier",
    "gradient", "confinement", "balanced", "wide",
]

# =============================================================================
# Shared full genome
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

GENOME_BLOCKS = (
    GAPS,
    INNER,
    OUTER,
    slice(LOCK, N_GENES),
)

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
# Transport zones
# =============================================================================

CENTRE_LIMIT_M = 45e-6
TRANSITION_LIMIT_M = 160e-6
ARM_LIMIT_M = 320e-6

REFERENCE_MIN_M = 260e-6
REFERENCE_MAX_M = 320e-6

# =============================================================================
# Search and final targets
# =============================================================================

SEARCH_DZ_LIMIT_M = 3.0e-6
SEARCH_EXCURSION_LIMIT_EV = 1.0e-3
SEARCH_LATERAL_LIMIT_M = 3.0e-6
SEARCH_TV_LIMIT_EV = 2.5e-3
SEARCH_GRADIENT_LIMIT_EV_M = 8.0
SEARCH_CONFINEMENT_MIN_RATIO = 0.20
SEARCH_BRANCH_MISMATCH_LIMIT_M = 3.0e-6

FINAL_DZ_LIMIT_M = 1.0e-6
FINAL_EXCURSION_LIMIT_EV = 0.10e-3
FINAL_LATERAL_LIMIT_M = 1.0e-6
FINAL_TV_LIMIT_EV = 1.0e-3
FINAL_GRADIENT_LIMIT_EV_M = 3.0
FINAL_CONFINEMENT_MIN_RATIO = 0.35
FINAL_BRANCH_MISMATCH_LIMIT_M = 1.0e-6


@dataclass(frozen=True)
class Settings:
    backend: str

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
    adaptive_top: int
    adaptive_levels: int

    max_panels: int
    fast: bool

    animation_stride: int
    surface_points: int

    stall_generations: int
    burst_generations: int
    improvement_fraction: float
    burst_mutation_multiplier: float

    mesh_excursion_tolerance_ev: float
    mesh_tv_tolerance_ev: float
    mesh_dz_tolerance_m: float
    mesh_confinement_tolerance: float

    output_dir: Path

    rf_peak_v: float = 100.0
    target_z_m: float = TARGET_ION_HEIGHT_M

    search_dz_limit_m: float = SEARCH_DZ_LIMIT_M
    search_excursion_limit_ev: float = SEARCH_EXCURSION_LIMIT_EV
    search_lateral_limit_m: float = SEARCH_LATERAL_LIMIT_M
    search_tv_limit_ev: float = SEARCH_TV_LIMIT_EV
    search_gradient_limit_ev_m: float = SEARCH_GRADIENT_LIMIT_EV_M
    search_confinement_min_ratio: float = SEARCH_CONFINEMENT_MIN_RATIO
    search_branch_mismatch_limit_m: float = SEARCH_BRANCH_MISMATCH_LIMIT_M

    final_dz_limit_m: float = FINAL_DZ_LIMIT_M
    final_excursion_limit_ev: float = FINAL_EXCURSION_LIMIT_EV
    final_lateral_limit_m: float = FINAL_LATERAL_LIMIT_M
    final_tv_limit_ev: float = FINAL_TV_LIMIT_EV
    final_gradient_limit_ev_m: float = FINAL_GRADIENT_LIMIT_EV_M
    final_confinement_min_ratio: float = FINAL_CONFINEMENT_MIN_RATIO
    final_branch_mismatch_limit_m: float = FINAL_BRANCH_MISMATCH_LIMIT_M


@dataclass(frozen=True)
class IslandStyle:
    name: str
    seed_kind: SeedKind
    emphasis: Emphasis
    explore_scale: float
    final_scale: float


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    style_name: str
    seed_kind: SeedKind
    emphasis: Emphasis
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result["valid"])

    @property
    def search_feasible(self) -> bool:
        return bool(self.result.get("search_feasible", False))

    @property
    def final_feasible(self) -> bool:
        return bool(self.result.get("final_feasible", False))

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf], dtype=float)

        return np.array(
            [
                float(self.result.get("excursion_global_ratio_search", np.inf)),
                float(self.result.get("tv_ratio_search", np.inf)),
                float(self.result.get("gradient_ratio_search", np.inf)),
                float(self.result.get("dz_global_ratio_search", np.inf)),
            ],
            dtype=float,
        )


    @property
    def violation(self) -> float:
        """
        Return total search-constraint violation.

        Invalid candidates are always worse than valid candidates.
        Older rejected() result formats may not contain the current
        search_constraint_violation key, therefore use a safe fallback.
        """
        if not self.valid:
            return np.inf

        raw_value = self.result.get(
            "search_constraint_violation",
            np.inf,
        )

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return np.inf

        return value if np.isfinite(value) else np.inf

    def copy(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(),
            result=dict(self.result),
            island=self.island,
            generation=self.generation,
            origin=self.origin,
            style_name=self.style_name,
            seed_kind=self.seed_kind,
            emphasis=self.emphasis,
            rank=self.rank,
            crowding=self.crowding,
        )


@dataclass
class Island:
    style: IslandStyle
    rng: np.random.Generator
    population: list[Individual]


class BEMField:
    """Adapter from FixedMeshBEM to RF-null tracing interface."""

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

    def electric_field_batch(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x_m, dtype=float).ravel()
        y = np.asarray(y_m, dtype=float).ravel()
        z = np.asarray(z_m, dtype=float).ravel()
        if not (x.shape == y.shape == z.shape):
            raise ValueError("Batch coordinates must have identical shapes.")
        values = self.bem.field_batch(
            x[None, :], y[None, :], z[None, :], self.charge,
            candidate_chunk=1,
        )
        return np.asarray(self.bem.asnumpy(values)[0], dtype=float)


# =============================================================================
# Genome helpers
# =============================================================================

def repair(genome: np.ndarray) -> np.ndarray:
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


def parameters(genome: np.ndarray) -> Any:
    values = repair(genome)

    knots_m, inner_offsets_m, outer_offsets_m = decode(
        values
    )

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
# Known elite seeds
# =============================================================================

def elite_seeds() -> dict[str, np.ndarray]:
    """
    Three distinct known ordinary basins.

    A: best joint candidate.
    B: best barrier candidate.
    C: diverse candidate from previous Pareto archive.
    """

    genome_A = np.array(
        [
            -2.693919933820525,
            -3.8791883884875884,
            -2.3107549237131555,
            -2.6635360680776348,
            -2.604364450724748,
            -1.0188113709792743,
            -2.295002976432831,
            -1.58956654563482,

            -4.671047050058288e-06,
            -6.628886993159777e-06,
            -1.4230855848703506e-05,
            -9.938723998480415e-06,
            -6.968225587669311e-07,
            -8.992668411738875e-06,
            -4.4706657882594386e-07,

            8.96957498891864e-06,
            1.2337918405266222e-05,
            1.5843821583578765e-05,
            6.160598233461362e-06,
            -3.0745376990661804e-06,
            6.330512570923556e-07,
            8.470178316320093e-07,

            1.6992351073699268e-04,
            -3.017299702847008e-05,
            1.9768896889529104e-05,
            2.477974346604927e-05,
            1.5294601025453086e-04,
            2.058947594763603,
            1.3531585219616183e-07,
            2.053237855986011e-05,
            2.1517821592870727e-05,
        ],
        dtype=float,
    )

    genome_B = np.array(
        [
            -2.709933932853829,
            -3.8337678030869418,
            -2.3338229158144213,
            -2.6934269345805704,
            -2.5421057953892996,
            -0.997926753075712,
            -2.295002976432831,
            -1.6903460656946008,

            -6.331026783805033e-06,
            -5.890457246891707e-06,
            -1.5281547606642362e-05,
            -9.999924830276548e-06,
            -1.26739540498944e-06,
            -9.912303077128267e-06,
            -1.3597748541155676e-06,

            7.274234673725851e-06,
            9.817866241204519e-06,
            1.4897250517529313e-05,
            6.478090564381574e-06,
            -3.81292082739843e-06,
            -1.9943656647684993e-08,
            2.796813400808457e-06,

            1.6687669509666937e-04,
            -2.9448708514443475e-05,
            2.135454064244456e-05,
            2.5211293026688254e-05,
            1.5024507179705118e-04,
            2.001091036207649,
            -9.751371789496804e-07,
            2.263872675032877e-05,
            2.191668209459116e-05,
        ],
        dtype=float,
    )

    genome_C = np.array(
        [
            -3.5878694899167303,
            -3.153762938162855,
            -2.713733453435762,
            -2.3309751634792555,
            -1.942648043149124,
            -1.4907367463890762,
            -1.4675270878437812,
            -1.761484342471865,

            -3.5052820174793393e-06,
            -9.070629157996265e-06,
            -1.1218134594172648e-05,
            -1.5222896079720358e-05,
            -2.757105287824218e-06,
            -5.559592001521011e-06,
            -5.054037406570247e-06,

            -1.3584388230842257e-06,
            -6.336830747342028e-08,
            7.49031343173437e-06,
            1.1462476127269313e-05,
            6.7139106982945445e-06,
            5.139106510379214e-06,
            1.7667972654596989e-06,

            1.5279141515234126e-04,
            -2.9765839678324105e-05,
            1.952631811890536e-05,
            2.627562347274212e-05,
            1.4496027635234118e-04,
            1.9605311344426817,
            -1.4640821799977548e-06,
            7.499354408876493e-05,
            2.569149473853044e-05,
        ],
        dtype=float,
    )

    return {
        "A": repair(genome_A),
        "B": repair(genome_B),
        "C": repair(genome_C),
    }


# =============================================================================
# Seed mixing
# =============================================================================

def local_sigmas(
    emphasis: Emphasis,
    scale: float,
) -> np.ndarray:
    """
    Local trust-region mutation scales.

    The values are intentionally small. Broad global mutation would destroy
    the known 8/8 meV/um basins before the BEM evaluator can exploit them.
    """
    sigma = np.zeros(N_GENES, dtype=float)

    sigma[GAPS] = 0.06
    sigma[INNER] = 1.20e-6
    sigma[OUTER] = 1.20e-6

    sigma[LOCK] = 2.5e-6

    sigma[CENTER_IN] = 0.8e-6
    sigma[CENTER_OUT] = 0.8e-6

    sigma[START] = 1.5e-6
    sigma[LENGTH] = 6.0e-6
    sigma[POWER] = 0.06

    sigma[BULGE] = 0.7e-6
    sigma[BULGE_CENTER] = 3.0e-6
    sigma[BULGE_WIDTH] = 2.0e-6

    if emphasis == "centre":
        sigma[INNER] *= 1.70
        sigma[CENTER_IN:CENTER_OUT + 1] *= 2.00
        sigma[START] *= 1.50

    elif emphasis == "transition":
        sigma[INNER] *= 1.50
        sigma[OUTER] *= 1.50
        sigma[START:LENGTH + 1] *= 2.00

    elif emphasis == "arm":
        sigma[OUTER] *= 2.00
        sigma[LENGTH] *= 2.00
        sigma[POWER] *= 1.50
        sigma[BULGE:BULGE_WIDTH + 1] *= 2.00

    elif emphasis == "height":
        sigma[INNER] *= 1.80
        sigma[CENTER_IN:CENTER_OUT + 1] *= 1.80
        sigma[START] *= 1.30

    elif emphasis == "barrier":
        sigma[OUTER] *= 1.80
        sigma[LENGTH] *= 1.50
        sigma[POWER] *= 1.40
        sigma[BULGE:BULGE_WIDTH + 1] *= 1.50

    elif emphasis == "gradient":
        sigma[GAPS] *= 1.35
        sigma[INNER] *= 1.45
        sigma[OUTER] *= 1.45
        sigma[LENGTH:POWER + 1] *= 1.45

    elif emphasis == "confinement":
        sigma[INNER] *= 1.45
        sigma[OUTER] *= 1.65
        sigma[CENTER_IN:CENTER_OUT + 1] *= 1.50

    elif emphasis == "wide":
        sigma *= 1.80

    return sigma * scale


def local_mutation(
    genome: np.ndarray,
    emphasis: Emphasis,
    rng: np.random.Generator,
    scale: float,
    probability: float = 0.55,
) -> np.ndarray:
    child = repair(genome)

    sigma = local_sigmas(emphasis, scale)

    active = rng.random(N_GENES) < probability

    child[active] += rng.normal(
        0.0,
        sigma[active],
    )

    return repair(child)


def two_parent_block_mix(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Copy whole meaningful genome blocks from one of two known basins.

    This avoids averaging incompatible contour structures gene by gene.
    """
    child = np.empty(N_GENES, dtype=float)

    for block in GENOME_BLOCKS:
        parent = left if rng.random() < 0.5 else right

        child[block] = parent[block]

    return repair(child)


def three_parent_block_mix(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    parents = (first, second, third)

    child = np.empty(N_GENES, dtype=float)

    for block in GENOME_BLOCKS:
        parent = parents[
            int(rng.integers(0, len(parents)))
        ]

        child[block] = parent[block]

    return repair(child)


def blended_block_mix(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mild arithmetic block crossover around known elite genomes.

    Alpha is intentionally restricted close to [0, 1] so it remains local.
    """
    child = np.empty(N_GENES, dtype=float)

    for block in GENOME_BLOCKS:
        alpha = float(rng.uniform(0.15, 0.85))

        child[block] = (
            alpha * left[block]
            + (1.0 - alpha) * right[block]
        )

    return repair(child)


def global_random_genome(rng: np.random.Generator) -> np.ndarray:
    """Broad but manufacturability-aware restart without prior elite data."""
    child = (LOW + HIGH) * 0.5
    child[GAPS] = rng.normal(0.0, 1.35, N_INTERVALS)
    child[INNER] = rng.normal(-6e-6, 9e-6, N_KNOTS - 2)
    child[OUTER] = rng.normal(4e-6, 9e-6, N_KNOTS - 2)
    child[LOCK] = rng.uniform(115e-6, 175e-6)
    child[CENTER_IN] = rng.uniform(-42e-6, -8e-6)
    child[CENTER_OUT] = rng.uniform(0.0e-6, 45e-6)
    child[START] = rng.uniform(12e-6, 48e-6)
    child[LENGTH] = rng.uniform(90e-6, 250e-6)
    child[POWER] = rng.uniform(1.0, 4.4)
    child[BULGE] = rng.uniform(-18e-6, 18e-6)
    child[BULGE_CENTER] = rng.uniform(25e-6, 125e-6)
    child[BULGE_WIDTH] = rng.uniform(12e-6, 65e-6)
    return repair(child)


def seed_from_kind(
    kind: SeedKind,
    seeds: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    A = seeds["A"]
    B = seeds["B"]
    C = seeds["C"]

    if kind == "GLOBAL":
        return "global_random", global_random_genome(rng)

    if kind == "A":
        return "seed_A", A.copy()

    if kind == "B":
        return "seed_B", B.copy()

    if kind == "C":
        return "seed_C", C.copy()

    if kind == "AB":
        return "mix_A_B", two_parent_block_mix(A, B, rng)

    if kind == "AC":
        return "mix_A_C", two_parent_block_mix(A, C, rng)

    if kind == "BC":
        return "mix_B_C", two_parent_block_mix(B, C, rng)

    return "mix_A_B_C", three_parent_block_mix(
        A,
        B,
        C,
        rng,
    )


# =============================================================================
# BEM and zonal evaluation
# =============================================================================

def rejected(
    reason: str,
    stage: str,
    cfg: Settings,
) -> dict[str, Any]:
    return {
    "valid": False,
    "search_feasible": False,
    "final_feasible": False,
    "verified": False,

    "reason": reason,
    "stage": stage,

    "dz_peak_m": np.inf,
    "dz_rms_m": np.inf,
    "lateral_peak_m": np.inf,

    "barrier_ev": np.inf,
    "barrier_reference_ev": np.nan,

    "peak_ev": np.inf,
    "dip_ev": np.inf,
    "excursion_ev": np.inf,
    "total_variation_ev": np.inf,
    "gradient_peak_ev_m": np.inf,
    "n_significant_minima": 0,
    "n_significant_maxima": 0,
    "confinement_min_ratio": 0.0,
    "confinement_cv": np.inf,
    "branch_mismatch_m": np.inf,
    "mesh_level": -1,
    "mesh_converged": False,
    "numerical_uncertainty_ev": np.inf,

    "dz_centre_peak_m": np.inf,
    "dz_transition_peak_m": np.inf,
    "dz_arm_peak_m": np.inf,

    "peak_centre_ev": np.inf,
    "peak_transition_ev": np.inf,
    "peak_arm_ev": np.inf,

    "dip_centre_ev": np.inf,
    "dip_transition_ev": np.inf,
    "dip_arm_ev": np.inf,

    "excursion_centre_ev": np.inf,
    "excursion_transition_ev": np.inf,
    "excursion_arm_ev": np.inf,

    "z_reference_m": np.nan,
    "u_reference_ev": np.nan,

    "dz_global_ratio_search": np.inf,
    "excursion_global_ratio_search": np.inf,
    "tv_ratio_search": np.inf,
    "gradient_ratio_search": np.inf,
    "confinement_violation_search": np.inf,
    "branch_ratio_search": np.inf,
    "lateral_ratio_search": np.inf,

    "search_constraint_violation": np.inf,

    "dz_global_ratio_final": np.inf,
    "excursion_global_ratio_final": np.inf,
    "lateral_ratio_final": np.inf,

    "panels": 0,
    "points": 0,
    "converged": 0,
}
def mesh_policy(level: int, fast: bool = False) -> dict[str, float]:
    """Nested geometry-aware quadtree policies; level 0 is GA screening."""
    policies = [
        dict(central_half_extent_m=145e-6, central_max_cell_m=52e-6,
             boundary_max_cell_m=20e-6, outer_max_cell_m=220e-6,
             min_cell_m=10e-6),
        dict(central_half_extent_m=180e-6, central_max_cell_m=30e-6,
             boundary_max_cell_m=10e-6, outer_max_cell_m=180e-6,
             min_cell_m=5e-6),
        dict(central_half_extent_m=205e-6, central_max_cell_m=22e-6,
             boundary_max_cell_m=7e-6, outer_max_cell_m=145e-6,
             min_cell_m=3.5e-6),
        dict(central_half_extent_m=225e-6, central_max_cell_m=16e-6,
             boundary_max_cell_m=5e-6, outer_max_cell_m=120e-6,
             min_cell_m=2.5e-6),
        dict(central_half_extent_m=245e-6, central_max_cell_m=12e-6,
             boundary_max_cell_m=3.5e-6, outer_max_cell_m=100e-6,
             min_cell_m=1.8e-6),
    ]
    chosen = dict(policies[min(max(int(level), 0), len(policies) - 1)])
    if fast and level == 0:
        chosen.update(
            central_half_extent_m=130e-6,
            central_max_cell_m=70e-6,
            boundary_max_cell_m=28e-6,
            outer_max_cell_m=240e-6,
            min_cell_m=14e-6,
        )
    return chosen


def build_mesh(parameter_object: Any, level: int, fast: bool) -> Any:
    return build_geometry_aware_quadtree_x_junction_bem(
        parameter_object, **mesh_policy(level, fast=fast)
    )


def zone_masks(
    x_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_m = np.abs(np.asarray(x_m, dtype=float))

    centre = s_m <= CENTRE_LIMIT_M

    transition = (
        (s_m > CENTRE_LIMIT_M)
        & (s_m <= TRANSITION_LIMIT_M)
    )

    arm = (
        (s_m > TRANSITION_LIMIT_M)
        & (s_m <= ARM_LIMIT_M)
    )

    reference = (
        (s_m >= REFERENCE_MIN_M)
        & (s_m <= REFERENCE_MAX_M)
    )

    return centre, transition, arm, reference


def zone_energy_metrics(
    relative_ev: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(relative_ev[mask], dtype=float)

    peak_ev = float(np.max(values))

    dip_ev = float(-np.min(values))

    excursion_ev = float(np.max(np.abs(values)))

    return peak_ev, dip_ev, excursion_ev


def profile_metrics(x_m: np.ndarray, relative_ev: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x_m, dtype=float)
    u = np.asarray(relative_ev, dtype=float)
    edge_order = 2 if len(x) >= 3 else 1
    gradient = np.gradient(u, x, edge_order=edge_order)
    total_variation_ev = float(np.trapz(np.abs(gradient), x))
    gradient_peak_ev_m = float(np.max(np.abs(gradient)))

    smooth = u.copy()
    if len(u) >= 5:
        smooth[1:-1] = 0.25 * u[:-2] + 0.50 * u[1:-1] + 0.25 * u[2:]
    span = float(np.ptp(smooth))
    threshold = max(0.05 * span, 1e-6)
    minima, maxima = [], []
    window = max(3, len(u) // 14)
    for i in range(1, len(u) - 1):
        left = smooth[max(0, i-window):i]
        right = smooth[i+1:min(len(u), i+window+1)]
        if not len(left) or not len(right):
            continue
        if smooth[i] <= smooth[i-1] and smooth[i] <= smooth[i+1]:
            prominence = min(float(np.max(left)-smooth[i]),
                             float(np.max(right)-smooth[i]))
            if prominence >= threshold:
                minima.append((i, prominence))
        if smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]:
            prominence = min(float(smooth[i]-np.min(left)),
                             float(smooth[i]-np.min(right)))
            if prominence >= threshold:
                maxima.append((i, prominence))
    return {
        "gradient_ev_m": gradient,
        "total_variation_ev": total_variation_ev,
        "gradient_peak_ev_m": gradient_peak_ev_m,
        "n_significant_minima": len(minima),
        "n_significant_maxima": len(maxima),
        "extrema_prominence_threshold_ev": threshold,
    }


def transverse_confinement_profile(
    field: BEMField,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    step_m: float = 0.5e-6,
) -> np.ndarray:
    """Return squared smallest singular value of d(Ey,Ez)/d(y,z)."""
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    n = len(x)
    xb = np.tile(x, 4)
    yb = np.concatenate([y+step_m, y-step_m, y, y])
    zb = np.concatenate([z, z, z+step_m, z-step_m])
    e = field.electric_field_batch(xb, yb, zb)
    ep_y, em_y, ep_z, em_z = np.split(e, 4)
    d_dy = (ep_y[:, 1:3] - em_y[:, 1:3]) / (2.0*step_m)
    d_dz = (ep_z[:, 1:3] - em_z[:, 1:3]) / (2.0*step_m)
    jac = np.empty((n, 2, 2), dtype=float)
    jac[:, :, 0] = d_dy
    jac[:, :, 1] = d_dz
    singular = np.linalg.svd(jac, compute_uv=False)
    return np.square(singular[:, -1])


def reverse_branch_mismatch(
    field: BEMField,
    x_m: np.ndarray,
    y_forward_m: np.ndarray,
    z_forward_m: np.ndarray,
    target_z_m: float,
) -> tuple[float, Any]:
    reverse = trace_rf_transverse_minimum(
        field, np.asarray(x_m)[::-1], initial_y_m=0.0,
        initial_z_m=target_z_m, residual_tolerance_v_m=1e-3,
        max_transverse_shift_m=25e-6,
    )
    if not reverse.valid:
        return np.inf, reverse
    yr = np.asarray(reverse.y_m)[::-1]
    zr = np.asarray(reverse.z_m)[::-1]
    mismatch = np.hypot(yr-y_forward_m, zr-z_forward_m)
    return float(np.max(mismatch)), reverse


def evaluate(
    genome: np.ndarray,
    cfg: Settings,
    stage: str = "coarse",
    retain: bool = False,
    mesh_level: int | None = None,
) -> tuple[dict[str, Any], Any | None]:
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
                    + "; ".join(map(str, report.messages)),
                    stage,
                    cfg,
                ),
                None,
            )

        longitudinal_m = np.linspace(
            0.0,
            parameter_object.arm_length_m,
            2001,
        )

        inner_m, outer_m = parameter_object.rail_boundaries_m(
            longitudinal_m
        )

        if np.min(inner_m) <= 0.5e-6:
            return rejected(
                "inner clearance",
                stage,
                cfg,
            ), None

        if np.min(outer_m - inner_m) < 18e-6:
            return rejected(
                "rail width",
                stage,
                cfg,
            ), None

        level = (0 if stage == "coarse" else 1) if mesh_level is None else int(mesh_level)
        model = build_mesh(
            parameter_object,
            level=level,
            fast=cfg.fast and stage == "coarse",
        )

        if model.n_panels > cfg.max_panels:
            return rejected(
                "panel limit",
                stage,
                cfg,
            ), None

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

        x_m = np.asarray(trace.x_m, dtype=float)
        y_m = np.asarray(trace.y_m, dtype=float)
        z_m = np.asarray(trace.z_m, dtype=float)

        if (
            not trace.valid
            or converged.shape != (points,)
            or not np.all(converged)
        ):
            return rejected(
                "incomplete trace",
                stage,
                cfg,
            ), None

        if not (
            np.all(np.isfinite(x_m))
            and np.all(np.isfinite(y_m))
            and np.all(np.isfinite(z_m))
        ):
            return rejected(
                "nonfinite trace",
                stage,
                cfg,
            ), None

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

        # -------------------------------------------------------------------------
        # Validate the existing project barrier calculation first.
        # -------------------------------------------------------------------------
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
                cfg,
            ), None

        # -------------------------------------------------------------------------
        # Spatial zones of the transport path.
        #
        # The masks are used for diagnostics and zonal objectives. They do NOT
        # define a new energy reference. That is important for compatibility with
        # old workflow results.
        # -------------------------------------------------------------------------
        centre_mask, transition_mask, arm_mask, reference_mask = zone_masks(
            x_m
        )

        if not (
            np.any(centre_mask)
            and np.any(transition_mask)
            and np.any(arm_mask)
            and np.any(reference_mask)
        ):
            return rejected(
                "incomplete zone coverage",
                stage,
                cfg,
            ), None

        # -------------------------------------------------------------------------
        # Height reference.
        #
        # Height is referenced to the far arm. This answers the practical question:
        # how much does the RF-null leave the linear-arm height during transport?
        # -------------------------------------------------------------------------
        z_reference_m = float(
            np.median(z_m[reference_mask])
        )

        dz_profile_m = z_m - z_reference_m

        dz_centre_peak_m = float(
            np.max(
                np.abs(
                    dz_profile_m[centre_mask]
                )
            )
        )

        dz_transition_peak_m = float(
            np.max(
                np.abs(
                    dz_profile_m[transition_mask]
                )
            )
        )

        dz_arm_peak_m = float(
            np.max(
                np.abs(
                    dz_profile_m[arm_mask]
                )
            )
        )

        dz_peak_m = float(
            np.max(
                np.abs(dz_profile_m)
            )
        )

        dz_rms_m = float(
            np.sqrt(
                np.mean(dz_profile_m**2)
            )
        )

        # -------------------------------------------------------------------------
        # Energy reference.
        #
        # CRITICAL:
        # Do not use:
        #
        #     median(pseudo_ev[reference_mask])
        #
        # because it changes the definition of energy zero and makes new results
        # incomparable with historical A/B/C barrier_ev values.
        #
        # Use the exact reference selected by the project's existing barrier metric.
        # -------------------------------------------------------------------------
        u_reference_ev = float(
            barrier.reference_energy_ev
        )

        u_relative_ev = pseudo_ev - u_reference_ev

        # -------------------------------------------------------------------------
        # Existing, historical metric.
        #
        # Keep this unchanged. It is what produced the known values:
        #
        #     A ~ 8.647 meV
        #     B ~ 8.388 meV
        #
        # and it remains the compatibility metric for all old workflows.
        # -------------------------------------------------------------------------
        barrier_ev = float(
            barrier.barrier_height_ev
        )

        # -------------------------------------------------------------------------
        # New stricter energy-flatness metrics.
        #
        # peak_ev      = positive hill above U_ref.
        # dip_ev       = positive depth of a well below U_ref.
        # excursion_ev = worst absolute energy deviation from U_ref.
        #
        # Therefore a central well can no longer be treated as a free improvement.
        # -------------------------------------------------------------------------
        peak_ev = float(
            np.max(u_relative_ev)
        )

        dip_ev = float(
            -np.min(u_relative_ev)
        )

        excursion_ev = float(
            np.max(
                np.abs(u_relative_ev)
            )
        )

        profile = profile_metrics(x_m, u_relative_ev)
        total_variation_ev = float(profile["total_variation_ev"])
        gradient_peak_ev_m = float(profile["gradient_peak_ev_m"])
        n_significant_minima = int(profile["n_significant_minima"])
        n_significant_maxima = int(profile["n_significant_maxima"])

        branch_mismatch_m, reverse_trace = reverse_branch_mismatch(
            field, x_m, y_m, z_m, cfg.target_z_m
        )

        confinement = transverse_confinement_profile(
            field, x_m, y_m, z_m
        )
        confinement_reference = float(np.median(confinement[reference_mask]))
        if confinement_reference <= 0.0 or not np.isfinite(confinement_reference):
            return rejected("invalid confinement reference", stage, cfg), None
        confinement_ratio_profile = confinement / confinement_reference
        confinement_min_ratio = float(np.min(confinement_ratio_profile))
        confinement_cv = float(
            np.std(confinement_ratio_profile) /
            max(np.mean(confinement_ratio_profile), 1e-15)
        )

        # -------------------------------------------------------------------------
        # Zonal energy metrics, all measured with the SAME U_ref.
        # -------------------------------------------------------------------------
        def zone_energy_metrics(
            relative_energy_ev: np.ndarray,
            mask: np.ndarray,
        ) -> tuple[float, float, float]:
            """
            Return:
                peak_ev      : maximum U - U_ref in zone
                dip_ev       : maximum U_ref - U in zone
                excursion_ev : max(abs(U - U_ref)) in zone
            """
            values_ev = np.asarray(
                relative_energy_ev[mask],
                dtype=float,
            )

            peak_ev = float(
                np.max(values_ev)
            )

            dip_ev = float(
                -np.min(values_ev)
            )

            excursion_ev = float(
                np.max(
                    np.abs(values_ev)
                )
            )

            return peak_ev, dip_ev, excursion_ev


        (
            peak_centre_ev,
            dip_centre_ev,
            excursion_centre_ev,
        ) = zone_energy_metrics(
            u_relative_ev,
            centre_mask,
        )

        (
            peak_transition_ev,
            dip_transition_ev,
            excursion_transition_ev,
        ) = zone_energy_metrics(
            u_relative_ev,
            transition_mask,
        )

        (
            peak_arm_ev,
            dip_arm_ev,
            excursion_arm_ev,
        ) = zone_energy_metrics(
            u_relative_ev,
            arm_mask,
        )

        # -------------------------------------------------------------------------
        # Lateral displacement. In a correct C4v geometry this should stay close to
        # numerical zero, but it is retained as a safety metric.
        # -------------------------------------------------------------------------
        lateral_peak_m = float(
            np.max(
                np.abs(y_m)
            )
        )

        # -------------------------------------------------------------------------
        # Final finite-value gate.
        # -------------------------------------------------------------------------
        metrics = np.array(
            [
                dz_peak_m,
                dz_rms_m,
                lateral_peak_m,

                barrier_ev,
                peak_ev,
                dip_ev,
                excursion_ev,
                total_variation_ev,
                gradient_peak_ev_m,
                branch_mismatch_m,
                confinement_min_ratio,
                confinement_cv,

                dz_centre_peak_m,
                dz_transition_peak_m,
                dz_arm_peak_m,

                peak_centre_ev,
                peak_transition_ev,
                peak_arm_ev,

                dip_centre_ev,
                dip_transition_ev,
                dip_arm_ev,

                excursion_centre_ev,
                excursion_transition_ev,
                excursion_arm_ev,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(metrics)):
            return rejected(
                "nonfinite metrics",
                stage,
                cfg,
            ), None
        dz_zone_ratios_search = np.array(
            [
                dz_centre_peak_m / cfg.search_dz_limit_m,
                dz_transition_peak_m / cfg.search_dz_limit_m,
                dz_arm_peak_m / cfg.search_dz_limit_m,
            ],
            dtype=float,
        )

        excursion_zone_ratios_search = np.array(
            [
                excursion_centre_ev
                / cfg.search_excursion_limit_ev,

                excursion_transition_ev
                / cfg.search_excursion_limit_ev,

                excursion_arm_ev
                / cfg.search_excursion_limit_ev,
            ],
            dtype=float,
        )

        lateral_ratio_search = (
            lateral_peak_m
            / cfg.search_lateral_limit_m
        )
        tv_ratio_search = total_variation_ev / cfg.search_tv_limit_ev
        gradient_ratio_search = (
            gradient_peak_ev_m / cfg.search_gradient_limit_ev_m
        )
        branch_ratio_search = (
            branch_mismatch_m / cfg.search_branch_mismatch_limit_m
        )
        confinement_violation_search = max(
            0.0,
            (cfg.search_confinement_min_ratio - confinement_min_ratio)
            / cfg.search_confinement_min_ratio,
        )

        dz_violation_search = float(
            np.sum(
                np.maximum(
                    0.0,
                    dz_zone_ratios_search - 1.0,
                )
            )
        )

        excursion_violation_search = float(
            np.sum(
                np.maximum(
                    0.0,
                    excursion_zone_ratios_search - 1.0,
                )
            )
        )

        lateral_violation_search = max(
            0.0,
            lateral_ratio_search - 1.0,
        )

        search_constraint_violation = float(
            dz_violation_search
            + excursion_violation_search
            + lateral_violation_search
            + max(0.0, tv_ratio_search - 1.0)
            + max(0.0, gradient_ratio_search - 1.0)
            + max(0.0, branch_ratio_search - 1.0)
            + confinement_violation_search
        )

        search_feasible = bool(
            np.all(dz_zone_ratios_search <= 1.0)
            and np.all(
                excursion_zone_ratios_search <= 1.0
            )
            and lateral_ratio_search <= 1.0
            and tv_ratio_search <= 1.0
            and gradient_ratio_search <= 1.0
            and branch_ratio_search <= 1.0
            and confinement_min_ratio >= cfg.search_confinement_min_ratio
        )

        dz_zone_ratios_final = np.array(
            [
                dz_centre_peak_m / cfg.final_dz_limit_m,
                dz_transition_peak_m / cfg.final_dz_limit_m,
                dz_arm_peak_m / cfg.final_dz_limit_m,
            ],
            dtype=float,
        )

        excursion_zone_ratios_final = np.array(
            [
                excursion_centre_ev
                / cfg.final_excursion_limit_ev,

                excursion_transition_ev
                / cfg.final_excursion_limit_ev,

                excursion_arm_ev
                / cfg.final_excursion_limit_ev,
            ],
            dtype=float,
        )

        lateral_ratio_final = (
            lateral_peak_m
            / cfg.final_lateral_limit_m
        )
        tv_ratio_final = total_variation_ev / cfg.final_tv_limit_ev
        gradient_ratio_final = gradient_peak_ev_m / cfg.final_gradient_limit_ev_m
        branch_ratio_final = branch_mismatch_m / cfg.final_branch_mismatch_limit_m

        final_feasible = bool(
            np.all(dz_zone_ratios_final <= 1.0)
            and np.all(
                excursion_zone_ratios_final <= 1.0
            )
            and lateral_ratio_final <= 1.0
            and tv_ratio_final <= 1.0
            and gradient_ratio_final <= 1.0
            and branch_ratio_final <= 1.0
            and confinement_min_ratio >= cfg.final_confinement_min_ratio
        )

        result = {
        "valid": True,
        "search_feasible": search_feasible,
        "final_feasible": final_feasible,
        "verified": bool(stage in ("fine", "adaptive") and final_feasible),
        "reason": "ok",
        "stage": stage,
        "backend": bem.backend_name,
        "mesh_level": int(level),
        "mesh_converged": False,
        "numerical_uncertainty_ev": np.nan,

        # -----------------------------------------------------------------
        # Existing compatible metrics. Do not change their meaning.
        # -----------------------------------------------------------------
        "barrier_ev": barrier_ev,
        "barrier_reference_ev": u_reference_ev,

        "dz_peak_m": dz_peak_m,
        "dz_rms_m": dz_rms_m,
        "lateral_peak_m": lateral_peak_m,

        # -----------------------------------------------------------------
        # New global flatness diagnostics.
        # -----------------------------------------------------------------
        "peak_ev": peak_ev,
        "dip_ev": dip_ev,
        "excursion_ev": excursion_ev,
        "total_variation_ev": total_variation_ev,
        "gradient_peak_ev_m": gradient_peak_ev_m,
        "n_significant_minima": n_significant_minima,
        "n_significant_maxima": n_significant_maxima,
        "branch_mismatch_m": branch_mismatch_m,
        "confinement_min_ratio": confinement_min_ratio,
        "confinement_cv": confinement_cv,

        "dz_global_ratio_search": float(np.max(dz_zone_ratios_search)),
        "excursion_global_ratio_search": float(np.max(excursion_zone_ratios_search)),
        "tv_ratio_search": tv_ratio_search,
        "gradient_ratio_search": gradient_ratio_search,
        "branch_ratio_search": branch_ratio_search,
        "confinement_violation_search": confinement_violation_search,
        "lateral_ratio_search": lateral_ratio_search,
        "search_constraint_violation": search_constraint_violation,

        "dz_global_ratio_final": float(np.max(dz_zone_ratios_final)),
        "excursion_global_ratio_final": float(np.max(excursion_zone_ratios_final)),
        "tv_ratio_final": tv_ratio_final,
        "gradient_ratio_final": gradient_ratio_final,
        "branch_ratio_final": branch_ratio_final,
        "lateral_ratio_final": lateral_ratio_final,

        # -----------------------------------------------------------------
        # Height diagnostics by spatial zone.
        # -----------------------------------------------------------------
        "dz_centre_peak_m": dz_centre_peak_m,
        "dz_transition_peak_m": dz_transition_peak_m,
        "dz_arm_peak_m": dz_arm_peak_m,

        # -----------------------------------------------------------------
        # Positive energy hills by zone.
        # -----------------------------------------------------------------
        "peak_centre_ev": peak_centre_ev,
        "peak_transition_ev": peak_transition_ev,
        "peak_arm_ev": peak_arm_ev,

        # -----------------------------------------------------------------
        # Energy-well depths by zone.
        # -----------------------------------------------------------------
        "dip_centre_ev": dip_centre_ev,
        "dip_transition_ev": dip_transition_ev,
        "dip_arm_ev": dip_arm_ev,

        # -----------------------------------------------------------------
        # Worst absolute energy deviations by zone.
        # -----------------------------------------------------------------
        "excursion_centre_ev": excursion_centre_ev,
        "excursion_transition_ev": excursion_transition_ev,
        "excursion_arm_ev": excursion_arm_ev,

        # -----------------------------------------------------------------
        # Reference values.
        # -----------------------------------------------------------------
        "z_reference_m": z_reference_m,
        "u_reference_ev": u_reference_ev,

        "panels": int(model.n_panels),
        "points": int(points),
        "converged": int(np.sum(converged)),
    }
        data = (
            (
                model,
                trace,
                pseudo_ev,
                field,
                parameter_object,
                {
                    "centre_mask": centre_mask,
                    "transition_mask": transition_mask,
                    "arm_mask": arm_mask,
                    "reference_mask": reference_mask,
                    "z_reference_m": z_reference_m,
                    "u_reference_ev": u_reference_ev,
                    "gradient_ev_m": np.asarray(profile["gradient_ev_m"]),
                    "confinement_ratio": confinement_ratio_profile,
                    "reverse_trace": reverse_trace,
                },
            )
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
                "CUDA OOM. Reduce --max-panels "
                "or use --fast."
            ) from error

        return rejected(
            "geometry/BEM error "
            f"{type(error).__name__}: "
            f"{str(error).replace(chr(10), ' ').strip()}",
            stage,
            cfg,
        ), None


# =============================================================================
# Zonal scores and NSGA-II
# =============================================================================

def zonal_score(
    result: dict[str, Any],
    emphasis: Emphasis,
) -> float:
    if not result["valid"]:
        return np.inf

    dz_c = (
        result["dz_centre_peak_m"]
        / SEARCH_DZ_LIMIT_M
    )

    dz_t = (
        result["dz_transition_peak_m"]
        / SEARCH_DZ_LIMIT_M
    )

    dz_a = (
        result["dz_arm_peak_m"]
        / SEARCH_DZ_LIMIT_M
    )

    u_c = (
        result["excursion_centre_ev"]
        / SEARCH_EXCURSION_LIMIT_EV
    )

    u_t = (
        result["excursion_transition_ev"]
        / SEARCH_EXCURSION_LIMIT_EV
    )

    u_a = (
        result["excursion_arm_ev"]
        / SEARCH_EXCURSION_LIMIT_EV
    )
    tv = float(result.get("tv_ratio_search", np.inf))
    grad = float(result.get("gradient_ratio_search", np.inf))
    conf = float(result.get("confinement_violation_search", np.inf))

    if emphasis == "gradient":
        return float(max(tv, grad, 0.45*max(dz_c, dz_t, dz_a),
                         0.45*max(u_c, u_t, u_a)))
    if emphasis == "confinement":
        return float(max(conf, 0.55*max(dz_c, dz_t, dz_a),
                         0.55*max(u_c, u_t, u_a)))

    if emphasis == "centre":
        return float(
            max(
                dz_c,
                u_c,
                0.45 * dz_t,
                0.45 * u_t,
                0.20 * dz_a,
                0.20 * u_a,
            )
        )

    if emphasis == "transition":
        return float(
            max(
                dz_t,
                u_t,
                0.55 * dz_c,
                0.55 * u_c,
                0.45 * dz_a,
                0.45 * u_a,
            )
        )

    if emphasis == "arm":
        return float(
            max(
                dz_a,
                u_a,
                0.55 * dz_t,
                0.55 * u_t,
                0.25 * dz_c,
                0.25 * u_c,
            )
        )

    if emphasis == "height":
        return float(
            max(
                dz_c,
                dz_t,
                dz_a,
                0.30 * u_c,
                0.30 * u_t,
                0.30 * u_a,
            )
        )

    if emphasis == "barrier":
        return float(
            max(
                u_c,
                u_t,
                u_a,
                0.30 * dz_c,
                0.30 * dz_t,
                0.30 * dz_a,
            )
        )

    if emphasis == "wide":
        return float(
            max(
                dz_c,
                dz_t,
                dz_a,
                u_c,
                u_t,
                u_a,
            )
        )

    return float(
        max(dz_c, dz_t, dz_a, u_c, u_t, u_a, tv, grad, conf)
    )


def dominates(
    left: Individual,
    right: Individual,
) -> bool:
    if left.valid != right.valid:
        return left.valid

    if not left.valid:
        return False

    if (
        left.search_feasible
        != right.search_feasible
    ):
        return left.search_feasible

    if not left.search_feasible:
        return left.violation < right.violation

    return bool(
        np.all(left.objectives <= right.objectives)
        and np.any(left.objectives < right.objectives)
    )


def rank_and_crowding(
    population: list[Individual],
) -> list[list[int]]:
    count = len(population)

    defeated = [[] for _ in population]

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

        feasible = [
            index
            for index in front
            if population[index].search_feasible
        ]

        if len(feasible) <= 2:
            for index in feasible:
                population[index].crowding = np.inf

            continue

        for axis in range(len(population[feasible[0]].objectives)):
            ordered = sorted(
                feasible,
                key=lambda index: (
                    population[index].objectives[axis]
                ),
            )

            low = population[ordered[0]].objectives[
                axis
            ]

            high = population[ordered[-1]].objectives[
                axis
            ]

            if high <= low:
                continue

            population[ordered[0]].crowding = np.inf
            population[ordered[-1]].crowding = np.inf

            for position in range(1, len(ordered) - 1):
                index = ordered[position]

                if np.isfinite(
                    population[index].crowding
                ):
                    before = population[
                        ordered[position - 1]
                    ].objectives[axis]

                    after = population[
                        ordered[position + 1]
                    ].objectives[axis]

                    population[index].crowding += (
                        (after - before)
                        / (high - low)
                    )

    return fronts


def survivors(
    pool: list[Individual],
    size: int,
    emphasis: Emphasis,
) -> list[Individual]:
    selected: list[Individual] = []

    for front in rank_and_crowding(pool):
        ordered = sorted(
            front,
            key=lambda index: (
                not pool[index].search_feasible,
                pool[index].violation,
                -pool[index].crowding,
                zonal_score(
                    pool[index].result,
                    emphasis,
                ),
            ),
        )

        selected.extend(
            pool[index].copy()
            for index in ordered[
                : size - len(selected)
            ]
        )

        if len(selected) == size:
            break

    return selected


def tournament(
    population: list[Individual],
    emphasis: Emphasis,
    rng: np.random.Generator,
) -> Individual:
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

    if (
        first.search_feasible
        != second.search_feasible
    ):
        return (
            first
            if first.search_feasible
            else second
        )

    if (
        not first.search_feasible
        and first.violation != second.violation
    ):
        return (
            first
            if first.violation < second.violation
            else second
        )

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
        if zonal_score(
            first.result,
            emphasis,
        )
        <= zonal_score(
            second.result,
            emphasis,
        )
        else second
    )


# =============================================================================
# Island setup
# =============================================================================

def make_island_styles() -> list[IslandStyle]:
    """Seeded local basins, block recombination and seed-free global restarts."""
    emphases: tuple[Emphasis, ...] = (
        "centre", "transition", "arm", "height", "barrier",
        "gradient", "confinement", "balanced", "wide",
    )
    layout: list[tuple[SeedKind, int]] = [
        ("A", 3), ("B", 3), ("C", 3),
        ("AB", 3), ("AC", 2), ("BC", 2), ("ABC", 2),
        ("GLOBAL", 6),
    ]
    styles: list[IslandStyle] = []
    for seed_kind, amount in layout:
        for index in range(amount):
            emphasis = emphases[len(styles) % len(emphases)]
            if seed_kind in ("A", "B", "C"):
                explore, final = 1.0, 0.22
            elif seed_kind == "GLOBAL":
                explore, final = 2.1, 0.55
            else:
                explore, final = 1.3, 0.34
            styles.append(IslandStyle(
                name=f"{seed_kind.lower()}_{emphasis}_{index:02d}",
                seed_kind=seed_kind, emphasis=emphasis,
                explore_scale=explore, final_scale=final,
            ))
    return styles


def anneal_scale(
    style: IslandStyle,
    generation_fraction: float,
) -> float:
    if generation_fraction <= 0.75:
        return style.explore_scale

    q = (generation_fraction - 0.75) / 0.25

    return (
        style.final_scale
        + 0.5
        * (
            style.explore_scale
            - style.final_scale
        )
        * (1.0 + math.cos(math.pi * q))
    )


def serialize(
    individual: Individual,
) -> dict[str, Any]:
    knots_m, inner_m, outer_m = decode(
        individual.genome
    )

    return {
        "island": individual.island,
        "generation": individual.generation,
        "origin": individual.origin,

        "style_name": individual.style_name,
        "seed_kind": individual.seed_kind,
        "emphasis": individual.emphasis,

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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = sorted(
        set().union(
            *(row.keys() for row in rows)
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
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
    rows: list[dict[str, Any]] = []

    for island in islands:
        rank_and_crowding(island.population)

        best = sorted(
            island.population,
            key=lambda item: (
                not item.search_feasible,
                item.violation,
                item.rank,
                -item.crowding,
                zonal_score(
                    item.result,
                    island.style.emphasis,
                ),
            ),
        )[: cfg.retain_per_island]

        rows.extend(
            serialize(item)
            for item in best
        )

    return rows


# =============================================================================
# Visualization
# =============================================================================

def draw_geometry(
    axis: Any,
    row: dict[str, Any],
    label: str,
) -> None:
    raw_genome = row["genome"]

    if isinstance(raw_genome, str):
        raw_genome = json.loads(raw_genome)

    genome = repair(
        np.asarray(raw_genome, dtype=float)
    )

    parameter_object = parameters(genome)

    s_m = np.linspace(0.0, 260e-6, 900)

    inner_m, outer_m = parameter_object.rail_boundaries_m(
        s_m
    )

    for theta in (
        0.0,
        np.pi / 2.0,
        np.pi,
        3.0 * np.pi / 2.0,
    ):
        cosine = np.cos(theta)
        sine = np.sin(theta)

        x_inner = s_m * cosine - inner_m * sine
        y_inner = s_m * sine + inner_m * cosine

        x_outer = s_m * cosine - outer_m * sine
        y_outer = s_m * sine + outer_m * cosine

        axis.fill(
            np.r_[x_inner, x_outer[::-1]] * 1e6,
            np.r_[y_inner, y_outer[::-1]] * 1e6,
            color="#db4f4f",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.90,
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
        f"{label}\n"
        f"dz={float(row['dz_peak_m']) * 1e6:.2f} um, "
        f"Uexc={float(row['excursion_ev']) * 1e3:.2f} meV",
        fontsize=8,
    )


def animate_surfaces(
    cfg: Settings,
    island_count: int,
) -> None:
    try:
        import pandas as pd
    except ImportError:
        print(
            "GIF generation skipped: pandas is not installed."
        )
        return

    files = sorted(
        (
            cfg.output_dir
            / "island_elites"
        ).glob("elites_gen_*.csv")
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

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    for island_id in range(island_count):
        choices: list[tuple[int, Any]] = []

        for generation, table in frames:
            subset = table[
                table["island"] == island_id
            ].sort_values(
                [
                    "search_feasible",
                    "search_constraint_violation",
                    "rank",
                    "crowding",
                    "dz_peak_m",
                    "excursion_ev",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                ],
            ).head(5)

            choices.append((generation, subset))

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
                        "no elite",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )

                    axis.set_axis_off()

            if rows:
                title = (
                    f"Island {island_id:02d} | "
                    f"{rows[0]['style_name']} | "
                    f"G{generation}"
                )
            else:
                title = (
                    f"Island {island_id:02d} | "
                    f"G{generation}"
                )

            figure.suptitle(
                title,
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
                / f"island_{island_id:02d}_top5.gif",
                writer=PillowWriter(fps=2),
                dpi=115,
            )

        except Exception as error:
            print(
                f"GIF skipped for island {island_id}: "
                f"{type(error).__name__}: {error}"
            )

        plt.close(figure)


def plot_mask_vs_bem(
    parameter_object: Any,
    model: Any,
    path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    coordinates_m = np.linspace(
        -250e-6,
        250e-6,
        600,
    )

    x_grid_m, y_grid_m = np.meshgrid(
        coordinates_m,
        coordinates_m,
        indexing="xy",
    )

    mask = baseline_x_junction_rf_mask(
        x_grid_m,
        y_grid_m,
        parameter_object,
    )

    axes[0].imshow(
        mask,
        extent=[-250, 250, -250, 250],
        origin="lower",
        cmap="RdYlBu_r",
        aspect="equal",
    )

    axes[0].set(
        title=f"{title}: continuous RF mask",
        xlabel="x [um]",
        ylabel="y [um]",
    )

    panels_m = np.asarray(model.bem.panels_m)

    rf = (
        np.asarray(
            model.bem.electrode_voltages_v
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
        title=(
            f"{title}: BEM quadtree "
            f"({model.n_panels} panels)"
        ),
        xlabel="x [um]",
        ylabel="y [um]",
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def pseudopotential_surface_ev(
    field: BEMField,
    cfg: Settings,
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

    z_flat_m = np.full_like(
        x_flat_m,
        cfg.target_z_m,
    )

    values = field.bem.field_batch(
        x_flat_m[None, :],
        y_flat_m[None, :],
        z_flat_m[None, :],
        field.charge,
        candidate_chunk=1,
    )

    electric_field = field.bem.asnumpy(values)[0]

    field_squared = np.sum(
        electric_field**2,
        axis=1,
    )

    elementary_charge_c = 1.602176634e-19

    ca40_mass_kg = (
        39.96259098
        * 1.66053906660e-27
    )

    potential_ev = (
        elementary_charge_c
        * cfg.rf_peak_v**2
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
        potential_ev.reshape(x_grid_m.shape),
    )


def shade_zones(
    axes: list[Any],
) -> None:
    zones = [
        (-320.0, -160.0, "#DDEBFF", 0.28),
        (-160.0, -45.0, "#FFF2CC", 0.28),
        (-45.0, 45.0, "#FCE4D6", 0.35),
        (45.0, 160.0, "#FFF2CC", 0.28),
        (160.0, 320.0, "#DDEBFF", 0.28),
    ]

    for axis in axes:
        for left, right, color, alpha in zones:
            axis.axvspan(
                left,
                right,
                color=color,
                alpha=alpha,
                zorder=0,
            )


def add_zone_labels(
    axis: Any,
) -> None:
    low, high = axis.get_ylim()

    y = high - 0.05 * (high - low)

    for x, label in [
        (-240, "left arm"),
        (-100, "transition"),
        (0, "centre"),
        (100, "transition"),
        (240, "right arm"),
    ]:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="0.25",
        )


def fine_plots(
    data: tuple[
        Any,
        Any,
        np.ndarray,
        BEMField,
        Any,
        dict[str, Any],
    ],
    result: dict[str, Any],
    directory: Path,
    title: str,
    cfg: Settings,
) -> None:
    (
        model,
        trace,
        pseudo_ev,
        field,
        parameter_object,
        zone_data,
    ) = data

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_mask_vs_bem(
        parameter_object,
        model,
        directory / "mask_vs_bem.png",
        title,
    )

    x_grid_m, y_grid_m, surface_ev = (
        pseudopotential_surface_ev(
            field,
            cfg,
        )
    )

    surface_mev = surface_ev * 1e3
    plot_interactive_surface(
        x_grid_m, y_grid_m, surface_ev,
        directory / "pseudopotential_3d.html", title,
    )

    finite = surface_mev[
        np.isfinite(surface_mev)
    ]

    vmax = (
        max(
            float(np.percentile(finite, 97.0)),
            1e-6,
        )
        if finite.size
        else 1.0
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 7.0)
    )

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
        xlabel="x [um]",
        ylabel="y [um]",
        title=(
            f"{title}\n"
            f"RF pseudopotential at "
            f"z={cfg.target_z_m * 1e6:.1f} um"
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

    dz_relative_um = (
        np.asarray(trace.z_m)
        - zone_data["z_reference_m"]
    ) * 1e6

    u_relative_mev = (
        pseudo_ev
        - zone_data["u_reference_ev"]
    ) * 1e3

    figure, axes = plt.subplots(
        5,
        1,
        sharex=True,
        figsize=(11.0, 14.0),
    )

    shade_zones(list(axes))

    axes[0].plot(
        x_um,
        dz_relative_um,
        color="#1f77b4",
    )

    axes[0].axhline(
        cfg.search_dz_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
        label="search target",
    )

    axes[0].axhline(
        -cfg.search_dz_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[0].axhline(
        cfg.final_dz_limit_m * 1e6,
        color="crimson",
        linestyle="--",
        label="final target",
    )

    axes[0].axhline(
        -cfg.final_dz_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[0].set_ylabel(
        "z - arm ref [um]"
    )

    axes[0].legend(
        loc="lower right",
        fontsize=8,
    )

    axes[1].plot(
        x_um,
        u_relative_mev,
        color="#d62728",
    )

    axes[1].axhline(
        cfg.search_excursion_limit_ev * 1e3,
        color="darkorange",
        linestyle=":",
    )

    axes[1].axhline(
        -cfg.search_excursion_limit_ev * 1e3,
        color="darkorange",
        linestyle=":",
    )

    axes[1].axhline(
        cfg.final_excursion_limit_ev * 1e3,
        color="crimson",
        linestyle="--",
    )

    axes[1].axhline(
        -cfg.final_excursion_limit_ev * 1e3,
        color="crimson",
        linestyle="--",
    )

    axes[1].set_ylabel(
        "U - arm ref [meV]"
    )

    axes[2].plot(
        x_um,
        np.asarray(zone_data["gradient_ev_m"]),
        color="#9467bd",
    )
    axes[2].set_ylabel("dU/dx [eV/m]")

    axes[3].plot(
        x_um,
        np.asarray(zone_data["confinement_ratio"]),
        color="#8c564b",
    )
    axes[3].axhline(cfg.search_confinement_min_ratio, color="darkorange", linestyle=":")
    axes[3].set_ylabel("RF confinement / arm")

    axes[4].plot(
        x_um,
        np.asarray(trace.y_m) * 1e6,
        color="#2ca02c",
    )

    axes[4].axhline(
        cfg.search_lateral_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[4].axhline(
        -cfg.search_lateral_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[4].axhline(
        cfg.final_lateral_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[4].axhline(
        -cfg.final_lateral_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[4].set(
        xlabel="x [um]",
        ylabel="y [um]",
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    add_zone_labels(axes[0])

    figure.suptitle(
        f"{title}\n"
        f"dz centre / transition / arm = "
        f"{result['dz_centre_peak_m'] * 1e6:.3f} / "
        f"{result['dz_transition_peak_m'] * 1e6:.3f} / "
        f"{result['dz_arm_peak_m'] * 1e6:.3f} um\n"
        f"Uexc centre / transition / arm = "
        f"{result['excursion_centre_ev'] * 1e3:.4f} / "
        f"{result['excursion_transition_ev'] * 1e3:.4f} / "
        f"{result['excursion_arm_ev'] * 1e3:.4f} meV"
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.90)
    )

    figure.savefig(
        directory / "profiles_zonal.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_convergence(
    history: list[dict[str, Any]],
    output_dir: Path,
    cfg: Settings,
) -> None:
    if not history:
        return

    generation = np.asarray(
        [row["generation"] for row in history],
        dtype=int,
    )

    best_dz_um = np.asarray(
        [
            row.get("best_dz_um", np.nan)
            for row in history
        ],
        dtype=float,
    )

    best_excursion_mev = np.asarray(
        [
            row.get("best_excursion_mev", np.nan)
            for row in history
        ],
        dtype=float,
    )

    joint_centre_dz_um = np.asarray(
        [
            row.get(
                "joint_dz_centre_um",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    joint_transition_dz_um = np.asarray(
        [
            row.get(
                "joint_dz_transition_um",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    joint_arm_dz_um = np.asarray(
        [
            row.get(
                "joint_dz_arm_um",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    joint_centre_u_mev = np.asarray(
        [
            row.get(
                "joint_U_centre_mev",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    joint_transition_u_mev = np.asarray(
        [
            row.get(
                "joint_U_transition_mev",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    joint_arm_u_mev = np.asarray(
        [
            row.get(
                "joint_U_arm_mev",
                np.nan,
            )
            for row in history
        ],
        dtype=float,
    )

    feasible = np.asarray(
        [
            row.get("search_feasible", 0)
            for row in history
        ],
        dtype=float,
    )

    valid = np.asarray(
        [
            row.get("valid", 0)
            for row in history
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        constrained_layout=True,
    )

    axes[0, 0].plot(
        generation,
        best_dz_um,
        marker="o",
        markersize=3,
        label="best global dz",
    )

    axes[0, 0].axhline(
        cfg.search_dz_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
        label="search target",
    )

    axes[0, 0].axhline(
        cfg.final_dz_limit_m * 1e6,
        color="crimson",
        linestyle="--",
        label="final target",
    )

    axes[0, 0].set(
        title="Global dz convergence",
        xlabel="generation",
        ylabel="dz peak [um]",
        yscale="log",
    )

    axes[0, 0].grid(alpha=0.30)
    axes[0, 0].legend()

    axes[0, 1].plot(
        generation,
        best_excursion_mev,
        marker="o",
        markersize=3,
        label="best global U excursion",
    )

    axes[0, 1].axhline(
        cfg.search_excursion_limit_ev * 1e3,
        color="darkorange",
        linestyle=":",
        label="search target",
    )

    axes[0, 1].axhline(
        cfg.final_excursion_limit_ev * 1e3,
        color="crimson",
        linestyle="--",
        label="final target",
    )

    axes[0, 1].set(
        title="Global energy-excursion convergence",
        xlabel="generation",
        ylabel="U excursion [meV]",
        yscale="log",
    )

    axes[0, 1].grid(alpha=0.30)
    axes[0, 1].legend()

    axes[1, 0].plot(
        generation,
        joint_centre_dz_um,
        marker="o",
        markersize=3,
        label="centre",
    )

    axes[1, 0].plot(
        generation,
        joint_transition_dz_um,
        marker="s",
        markersize=3,
        label="transition",
    )

    axes[1, 0].plot(
        generation,
        joint_arm_dz_um,
        marker="^",
        markersize=3,
        label="arm",
    )

    axes[1, 0].axhline(
        cfg.search_dz_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[1, 0].axhline(
        cfg.final_dz_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[1, 0].set(
        title="Joint candidate dz by zone",
        xlabel="generation",
        ylabel="dz [um]",
        yscale="log",
    )

    axes[1, 0].grid(alpha=0.30)
    axes[1, 0].legend()

    axes[1, 1].plot(
        generation,
        joint_centre_u_mev,
        marker="o",
        markersize=3,
        label="centre",
    )

    axes[1, 1].plot(
        generation,
        joint_transition_u_mev,
        marker="s",
        markersize=3,
        label="transition",
    )

    axes[1, 1].plot(
        generation,
        joint_arm_u_mev,
        marker="^",
        markersize=3,
        label="arm",
    )

    axes[1, 1].axhline(
        cfg.search_excursion_limit_ev * 1e3,
        color="darkorange",
        linestyle=":",
    )

    axes[1, 1].axhline(
        cfg.final_excursion_limit_ev * 1e3,
        color="crimson",
        linestyle="--",
    )

    axes[1, 1].set(
        title="Joint candidate U excursion by zone",
        xlabel="generation",
        ylabel="U excursion [meV]",
        yscale="log",
    )

    axes[1, 1].grid(alpha=0.30)
    axes[1, 1].legend()

    figure.savefig(
        output_dir / "convergence_zonal.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    axis.plot(
        generation,
        valid,
        marker="o",
        markersize=3,
        label="valid",
    )

    axis.plot(
        generation,
        feasible,
        marker="o",
        markersize=3,
        label="search feasible",
    )

    axis.set(
        title="Population validity and feasibility",
        xlabel="generation",
        ylabel="individuals",
    )

    axis.grid(alpha=0.30)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir / "population_state.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_pareto(
    archive: list[Individual],
    output_dir: Path,
    cfg: Settings,
) -> None:
    valid = [
        item
        for item in archive
        if item.valid
    ]

    if not valid:
        return

    dz_um = np.array(
        [
            item.result["dz_peak_m"] * 1e6
            for item in valid
        ]
    )

    excursion_mev = np.array(
        [
            item.result["excursion_ev"] * 1e3
            for item in valid
        ]
    )

    kinds = np.array(
        [
            item.seed_kind
            for item in valid
        ]
    )

    search_feasible = np.array(
        [
            item.search_feasible
            for item in valid
        ]
    )

    colors = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
        "AB": "#9467bd",
        "AC": "#8c564b",
        "BC": "#e377c2",
        "ABC": "#7f7f7f",
    }

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    for kind in colors:
        mask = kinds == kind

        if np.any(mask):
            axis.scatter(
                dz_um[mask],
                excursion_mev[mask],
                c=colors[kind],
                alpha=0.65,
                s=28,
                label=kind,
            )

    if np.any(search_feasible):
        axis.scatter(
            dz_um[search_feasible],
            excursion_mev[search_feasible],
            c="#111111",
            marker="*",
            s=160,
            label="search feasible",
        )

    axis.axvline(
        cfg.search_dz_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
        label="search dz",
    )

    axis.axhline(
        cfg.search_excursion_limit_ev * 1e3,
        color="darkorange",
        linestyle=":",
        label="search U",
    )

    axis.axvline(
        cfg.final_dz_limit_m * 1e6,
        color="crimson",
        linestyle="--",
        label="final dz",
    )

    axis.axhline(
        cfg.final_excursion_limit_ev * 1e3,
        color="crimson",
        linestyle="--",
        label="final U",
    )

    axis.set(
        xlabel="global dz peak [um]",
        ylabel="global U excursion [meV]",
        xscale="log",
        yscale="log",
        title="Seeded zonal island archive",
    )

    axis.grid(alpha=0.30)
    axis.legend(
        ncol=2,
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        output_dir / "pareto_archive.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# Main workflow
# =============================================================================

def adaptive_verify(
    genome: np.ndarray,
    cfg: Settings,
    retain: bool = True,
) -> tuple[dict[str, Any], Any | None, list[dict[str, Any]]]:
    """Refine nested BEM policies until RF quantities of interest stabilize."""
    records: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    best_result: dict[str, Any] = rejected("adaptive not run", "adaptive", cfg)
    best_data: Any | None = None
    max_level = min(max(cfg.adaptive_levels, 1), 5)

    for level in range(1, max_level + 1):
        result, data = evaluate(
            genome, cfg, stage="adaptive", retain=retain,
            mesh_level=level,
        )
        result["mesh_level"] = level
        row = {
            "mesh_level": level,
            "panels": result.get("panels", 0),
            "valid": result.get("valid", False),
            "reason": result.get("reason", ""),
            "excursion_ev": result.get("excursion_ev", np.inf),
            "total_variation_ev": result.get("total_variation_ev", np.inf),
            "gradient_peak_ev_m": result.get("gradient_peak_ev_m", np.inf),
            "dz_peak_m": result.get("dz_peak_m", np.inf),
            "confinement_min_ratio": result.get("confinement_min_ratio", 0.0),
        }
        if not result.get("valid", False):
            records.append(row)
            if previous is not None:
                best_result = dict(previous)
                best_result["reason"] = (
                    "last converged mesh retained; finer level failed: "
                    + str(result.get("reason", "unknown"))
                )
                best_result["mesh_converged"] = False
                best_result["verified"] = False
            break

        if previous is not None:
            d_exc = abs(float(result["excursion_ev"]) - float(previous["excursion_ev"]))
            d_tv = abs(float(result["total_variation_ev"]) - float(previous["total_variation_ev"]))
            d_dz = abs(float(result["dz_peak_m"]) - float(previous["dz_peak_m"]))
            d_conf = abs(float(result["confinement_min_ratio"]) - float(previous["confinement_min_ratio"]))
            converged = bool(
                d_exc <= cfg.mesh_excursion_tolerance_ev
                and d_tv <= cfg.mesh_tv_tolerance_ev
                and d_dz <= cfg.mesh_dz_tolerance_m
                and d_conf <= cfg.mesh_confinement_tolerance
            )
            uncertainty_ev = max(d_exc, 0.5*d_tv)
            row.update(delta_excursion_ev=d_exc, delta_tv_ev=d_tv,
                       delta_dz_m=d_dz, delta_confinement=d_conf,
                       mesh_converged=converged,
                       numerical_uncertainty_ev=uncertainty_ev)
            result["mesh_converged"] = converged
            result["numerical_uncertainty_ev"] = uncertainty_ev
            result["conservative_excursion_ev"] = float(result["excursion_ev"]) + uncertainty_ev
            result["verified"] = bool(converged and result.get("final_feasible", False))
        else:
            row.update(mesh_converged=False, numerical_uncertainty_ev=np.inf)
            result["mesh_converged"] = False
            result["numerical_uncertainty_ev"] = np.inf
            result["conservative_excursion_ev"] = np.inf
            result["verified"] = False

        records.append(row)
        best_result, best_data = result, data
        if result["mesh_converged"]:
            break
        previous = result

    return best_result, best_data, records


def plot_mesh_convergence(records: list[dict[str, Any]], path: Path) -> None:
    valid = [r for r in records if r.get("valid", False)]
    if not valid:
        return
    panels = np.asarray([r["panels"] for r in valid], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(panels, [r["excursion_ev"]*1e3 for r in valid], "o-")
    axes[0, 0].set_ylabel("excursion [meV]")
    axes[0, 1].plot(panels, [r["total_variation_ev"]*1e3 for r in valid], "o-")
    axes[0, 1].set_ylabel("TV(U) [meV]")
    axes[1, 0].plot(panels, [r["dz_peak_m"]*1e6 for r in valid], "o-")
    axes[1, 0].set_ylabel("height excursion [um]")
    axes[1, 1].plot(panels, [r["confinement_min_ratio"] for r in valid], "o-")
    axes[1, 1].set_ylabel("minimum confinement ratio")
    for ax in axes.ravel():
        ax.set_xlabel("BEM panels")
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_surface(
    x_grid_m: np.ndarray, y_grid_m: np.ndarray, surface_ev: np.ndarray,
    path: Path, title: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    fig = go.Figure(data=[go.Surface(
        x=x_grid_m*1e6, y=y_grid_m*1e6, z=surface_ev*1e3,
        colorscale="Magma", colorbar=dict(title="meV"),
    )])
    fig.update_layout(title=title, scene=dict(
        xaxis_title="x [um]", yaxis_title="y [um]",
        zaxis_title="RF pseudopotential [meV]",
    ))
    fig.write_html(path, include_plotlyjs="cdn")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=__doc__
    )

    command.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    command.add_argument(
        "--population",
        type=int,
        default=10,
    )

    command.add_argument(
        "--offspring",
        type=int,
        default=3,
    )

    command.add_argument(
        "--generations",
        type=int,
        default=20,
    )

    command.add_argument(
        "--migration-interval",
        type=int,
        default=3,
    )

    command.add_argument(
        "--migrants",
        type=int,
        default=2,
    )

    command.add_argument(
        "--retain-per-island",
        type=int,
        default=5,
    )

    command.add_argument(
        "--coarse-points",
        type=int,
        default=51,
    )

    command.add_argument(
        "--fine-points",
        type=int,
        default=181,
    )

    command.add_argument(
        "--fine-top",
        type=int,
        default=20,
    )

    command.add_argument("--adaptive-top", type=int, default=10)
    command.add_argument("--adaptive-levels", type=int, default=4)
    command.add_argument("--mesh-excursion-tol-mev", type=float, default=0.03)
    command.add_argument("--mesh-tv-tol-mev", type=float, default=0.08)
    command.add_argument("--mesh-dz-tol-um", type=float, default=0.10)
    command.add_argument("--mesh-confinement-tol", type=float, default=0.04)

    command.add_argument(
        "--max-panels",
        type=int,
        default=9000,
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
        default=4,
    )

    command.add_argument(
        "--burst-generations",
        type=int,
        default=2,
    )

    command.add_argument(
        "--improvement-fraction",
        type=float,
        default=0.01,
    )

    command.add_argument(
        "--burst-mutation-multiplier",
        type=float,
        default=2.5,
    )

    command.add_argument(
        "--seed",
        type=int,
        default=20260927,
    )

    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/13_seeded_zonal_islands"
        ),
    )

    command.add_argument(
        "--smoke",
        action="store_true",
    )

    return command


def main() -> None:
    arguments = parser().parse_args()

    if arguments.smoke:
        arguments.population = 8
        arguments.offspring = 2
        arguments.generations = 2
        arguments.migration_interval = 1
        arguments.migrants = 1
        arguments.retain_per_island = 3
        arguments.coarse_points = 31
        arguments.fine_points = 61
        arguments.fine_top = 6
        arguments.adaptive_top = 2
        arguments.adaptive_levels = 2
        arguments.max_panels = 5000
        arguments.fast = True
        arguments.animation_stride = 1
        arguments.surface_points = 61
        arguments.stall_generations = 1
        arguments.burst_generations = 1

    if (
        arguments.population < 8
        or arguments.offspring < 1
        or arguments.generations < 1
    ):
        raise ValueError(
            "Require population >= 8 and positive "
            "offspring/generations."
        )

    backend = available_backend(arguments.backend)

    if not backend.available:
        raise RuntimeError(backend.reason)

    cfg = Settings(
        backend=backend.selected,

        population=arguments.population,
        offspring=arguments.offspring,
        generations=arguments.generations,

        migration_interval=(
            arguments.migration_interval
        ),

        migrants=arguments.migrants,

        retain_per_island=(
            arguments.retain_per_island
        ),

        seed=arguments.seed,

        coarse_points=arguments.coarse_points,
        fine_points=arguments.fine_points,
        fine_top=arguments.fine_top,
        adaptive_top=arguments.adaptive_top,
        adaptive_levels=arguments.adaptive_levels,

        max_panels=arguments.max_panels,
        fast=arguments.fast,

        animation_stride=arguments.animation_stride,
        surface_points=arguments.surface_points,

        stall_generations=arguments.stall_generations,
        burst_generations=arguments.burst_generations,
        improvement_fraction=(
            arguments.improvement_fraction
        ),

        burst_mutation_multiplier=(
            arguments.burst_mutation_multiplier
        ),
        mesh_excursion_tolerance_ev=arguments.mesh_excursion_tol_mev * 1e-3,
        mesh_tv_tolerance_ev=arguments.mesh_tv_tol_mev * 1e-3,
        mesh_dz_tolerance_m=arguments.mesh_dz_tol_um * 1e-6,
        mesh_confinement_tolerance=arguments.mesh_confinement_tol,

        output_dir=arguments.output_dir,
    )

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = make_island_styles()

    backend_payload = (
        backend.as_dict()
        if hasattr(backend, "as_dict")
        else {
            "selected": backend.selected,
            "available": backend.available,
        }
    )

    (cfg.output_dir / "settings.json").write_text(
        json.dumps(
            {
                "settings": asdict(cfg),
                "backend": backend_payload,
                "islands": [
                    asdict(style)
                    for style in styles
                ],

                "genome_genes": N_GENES,
                "objectives": ["excursion", "total_variation", "gradient_peak", "height_excursion"],
                "one_mev_is_normalization_not_promise": True,
                "mesh_adaptation": "observable-driven nested quadtree policies",

                "search_targets": {
                    "dz_um": (
                        cfg.search_dz_limit_m * 1e6
                    ),
                    "excursion_mev": (
                        cfg.search_excursion_limit_ev
                        * 1e3
                    ),
                    "lateral_um": (
                        cfg.search_lateral_limit_m
                        * 1e6
                    ),
                },

                "final_targets": {
                    "dz_um": (
                        cfg.final_dz_limit_m * 1e6
                    ),
                    "excursion_mev": (
                        cfg.final_excursion_limit_ev
                        * 1e3
                    ),
                    "lateral_um": (
                        cfg.final_lateral_limit_m
                        * 1e6
                    ),
                },

                "transport_zones_um": {
                    "centre_abs_x_le": (
                        CENTRE_LIMIT_M * 1e6
                    ),
                    "transition_abs_x": [
                        CENTRE_LIMIT_M * 1e6,
                        TRANSITION_LIMIT_M * 1e6,
                    ],
                    "arm_abs_x": [
                        TRANSITION_LIMIT_M * 1e6,
                        ARM_LIMIT_M * 1e6,
                    ],
                    "reference_abs_x": [
                        REFERENCE_MIN_M * 1e6,
                        REFERENCE_MAX_M * 1e6,
                    ],
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Workflow 13 | backend={backend.selected}"
    )

    print(
        f"Islands: {len(styles)} | "
        f"population={cfg.population} | "
        f"offspring={cfg.offspring}"
    )

    print(
        "Search targets: "
        f"dz < {cfg.search_dz_limit_m * 1e6:.2f} um, "
        f"Uexc < {cfg.search_excursion_limit_ev * 1e3:.2f} meV"
    )

    print(
        "Final targets: "
        f"dz < {cfg.final_dz_limit_m * 1e6:.2f} um, "
        f"Uexc < {cfg.final_excursion_limit_ev * 1e3:.2f} meV"
    )

    seeds = elite_seeds()

    master_rng = np.random.default_rng(cfg.seed)

    cache: dict[
        tuple[float, ...],
        dict[str, Any],
    ] = {}

    archive: list[Individual] = []

    history: list[dict[str, Any]] = []

    rejection_rows: list[dict[str, Any]] = []

    def coarse(
        genome: np.ndarray,
        progress: Any | None = None,
    ) -> dict[str, Any]:
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

        if progress is not None:
            if result["valid"]:
                progress.set_postfix(
                    dz=(
                        f"{result['dz_peak_m'] * 1e6:.2f}"
                        "um"
                    ),
                    U=(
                        f"{result['excursion_ev'] * 1e3:.2f}"
                        "meV"
                    ),
                    refresh=False,
                )

            else:
                progress.set_postfix(
                    state="invalid",
                    refresh=False,
                )

        return result

    islands: list[Island] = []

    with tqdm(
        total=len(styles) * cfg.population,
        desc="Initial seeded BEM",
        unit="candidate",
    ) as progress:
        for island_id, style in enumerate(styles):
            rng = np.random.default_rng(
                int(
                    master_rng.integers(
                        0,
                        2**32 - 1,
                    )
                )
            )

            population: list[Individual] = []

            # First individual is exact source seed or exact block mix.
            label, genome = seed_from_kind(
                style.seed_kind,
                seeds,
                rng,
            )

            population.append(
                Individual(
                    genome=genome.copy(),
                    result=coarse(
                        genome,
                        progress,
                    ),
                    island=island_id,
                    generation=0,
                    origin=label,
                    style_name=style.name,
                    seed_kind=style.seed_kind,
                    emphasis=style.emphasis,
                )
            )

            progress.update()

            # Remaining initial candidates are local perturbations.
            while len(population) < cfg.population:
                label, base = seed_from_kind(
                    style.seed_kind,
                    seeds,
                    rng,
                )

                genome = local_mutation(
                    base,
                    style.emphasis,
                    rng,
                    scale=style.explore_scale,
                    probability=0.75,
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
                        origin=f"initial_{label}",
                        style_name=style.name,
                        seed_kind=style.seed_kind,
                        emphasis=style.emphasis,
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
        item.valid
        for island in islands
        for item in island.population
    ):
        reasons = Counter(
            item.result["reason"]
            for island in islands
            for item in island.population
        )

        raise RuntimeError(
            "No valid initial seeded geometry. "
            f"Reasons: {reasons.most_common(20)!r}"
        )

    best_joint_seen = np.inf
    stall_count = 0
    burst_remaining = 0
    search_mode = "NORMAL"

    def global_joint_score(
        result: dict[str, Any],
    ) -> float:
        """
        Safe global minimax score.

        Supports:
        - current zonal result schema;
        - older result dictionaries;
        - rejected / invalid candidates.

        Lower is better.
        """
        if not bool(result.get("valid", False)):
            return np.inf

        def finite_float(
            key: str,
            default: float = np.inf,
        ) -> float:
            raw_value = result.get(key, default)

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return default

            return value if np.isfinite(value) else default

        # Preferred current zonal schema.
        dz_ratio = finite_float(
            "dz_global_ratio_search",
            default=np.nan,
        )

        excursion_ratio = finite_float(
            "excursion_global_ratio_search",
            default=np.nan,
        )

        lateral_ratio = finite_float(
            "lateral_ratio_search",
            default=np.nan,
        )

        if (
            np.isfinite(dz_ratio)
            and np.isfinite(excursion_ratio)
            and np.isfinite(lateral_ratio)
        ):
            return float(max(
                dz_ratio, excursion_ratio, lateral_ratio,
                finite_float("tv_ratio_search", 0.0),
                finite_float("gradient_ratio_search", 0.0),
                finite_float("branch_ratio_search", 0.0),
                finite_float("confinement_violation_search", 0.0),
            ))

        # Fallback for old workflow result schema:
        # barrier_ev and dz_peak_m are the historical metrics.
        dz_peak_m = finite_float(
            "dz_peak_m",
            default=np.inf,
        )

        barrier_ev = finite_float(
            "barrier_ev",
            default=np.inf,
        )

        lateral_peak_m = finite_float(
            "lateral_peak_m",
            default=0.0,
        )

        return float(
            max(
                dz_peak_m / SEARCH_DZ_LIMIT_M,
                barrier_ev / SEARCH_EXCURSION_LIMIT_EV,
                lateral_peak_m / SEARCH_LATERAL_LIMIT_M,
            )
        )

    def update_archive(
        items: list[Individual],
    ) -> None:
        nonlocal archive

        unique = {
            tuple(np.round(item.genome, 12)): item.copy()
            for item in archive + items
            if item.valid
        }

        archive = list(unique.values())

        rank_and_crowding(archive)

        archive.sort(
            key=lambda item: (
                not item.search_feasible,
                item.violation,
                item.rank,
                -item.crowding,
                global_joint_score(item.result),
            )
        )

        archive = archive[:2000]

    def summary(
        generation: int,
    ) -> dict[str, Any]:
        all_items = [
            item
            for island in islands
            for item in island.population
        ]

        valid = [
            item
            for item in all_items
            if item.valid
        ]

        search_feasible = [
            item
            for item in valid
            if item.search_feasible
        ]

        rejected_items = [
            item
            for item in all_items
            if not item.valid
        ]

        reasons = Counter(
            item.result["reason"]
            for item in rejected_items
        )

        for reason, count in reasons.items():
            rejection_rows.append(
                {
                    "generation": generation,
                    "reason": reason,
                    "count": count,
                }
            )

        row: dict[str, Any] = {
            "generation": generation,
            "mode": search_mode,

            "stall_count": stall_count,
            "burst_remaining": burst_remaining,

            "valid": len(valid),
            "search_feasible": len(search_feasible),
            "invalid": len(rejected_items),

            "archive": len(archive),
            "unique_evaluations": len(cache),

            "rejects": "; ".join(
                f"{key}:{value}"
                for key, value in reasons.most_common(4)
            ),
        }

        if not valid:
            return row

        best_dz = min(
            valid,
            key=lambda item: item.result["dz_peak_m"],
        )

        best_excursion = min(
            valid,
            key=lambda item: item.result["excursion_ev"],
        )

        joint = min(
            valid,
            key=lambda item: global_joint_score(
                item.result
            ),
        )

        row.update(
            {
                "best_dz_um": (
                    best_dz.result["dz_peak_m"]
                    * 1e6
                ),

                "best_excursion_mev": (
                    best_excursion.result[
                        "excursion_ev"
                    ]
                    * 1e3
                ),

                "joint_dz_um": (
                    joint.result["dz_peak_m"]
                    * 1e6
                ),

                "joint_excursion_mev": (
                    joint.result["excursion_ev"]
                    * 1e3
                ),

                "joint_score": global_joint_score(
                    joint.result
                ),

                "joint_violation": joint.violation,

                "joint_seed_kind": joint.seed_kind,
                "joint_emphasis": joint.emphasis,
                "joint_style": joint.style_name,

                "joint_dz_centre_um": (
                    joint.result[
                        "dz_centre_peak_m"
                    ]
                    * 1e6
                ),

                "joint_dz_transition_um": (
                    joint.result[
                        "dz_transition_peak_m"
                    ]
                    * 1e6
                ),

                "joint_dz_arm_um": (
                    joint.result[
                        "dz_arm_peak_m"
                    ]
                    * 1e6
                ),

                "joint_U_centre_mev": (
                    joint.result[
                        "excursion_centre_ev"
                    ]
                    * 1e3
                ),

                "joint_U_transition_mev": (
                    joint.result[
                        "excursion_transition_ev"
                    ]
                    * 1e3
                ),

                "joint_U_arm_mev": (
                    joint.result[
                        "excursion_arm_ev"
                    ]
                    * 1e3
                ),
            }
        )

        return row

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

    print("G000", history[-1], flush=True)

    for generation in range(
        1,
        cfg.generations + 1,
    ):
        fraction = generation / cfg.generations

        burst_active = burst_remaining > 0

        search_mode = (
            "BURST"
            if burst_active
            else "NORMAL"
        )

        next_islands: list[Island] = []

        with tqdm(
            total=len(islands) * cfg.offspring,
            desc=(
                f"Generation {generation} "
                f"{search_mode}"
            ),
            unit="candidate",
        ) as progress:
            for island_id, island in enumerate(islands):
                rank_and_crowding(island.population)

                elite = sorted(
                    island.population,
                    key=lambda item: (
                        not item.search_feasible,
                        item.violation,
                        item.rank,
                        -item.crowding,
                        zonal_score(
                            item.result,
                            island.style.emphasis,
                        ),
                    ),
                )[
                    : max(
                        2,
                        min(
                            5,
                            len(island.population),
                        ),
                    )
                ]

                children: list[Individual] = []

                scale = anneal_scale(
                    island.style,
                    fraction,
                )

                while len(children) < cfg.offspring:
                    value = island.rng.random()

                    if not burst_active:
                        if value < 0.55:
                            parent = elite[
                                0
                                if (
                                    len(elite) == 1
                                    or island.rng.random() < 0.80
                                )
                                else 1
                            ]

                            genome = parent.genome.copy()

                            origin = "elite_local_mutation"

                        elif value < 0.78:
                            left = tournament(
                                island.population,
                                island.style.emphasis,
                                island.rng,
                            )

                            right = tournament(
                                island.population,
                                island.style.emphasis,
                                island.rng,
                            )

                            genome = blended_block_mix(
                                left.genome,
                                right.genome,
                                island.rng,
                            )

                            origin = "local_blended_crossover"

                        elif value < 0.92:
                            left = tournament(
                                island.population,
                                island.style.emphasis,
                                island.rng,
                            )

                            right = tournament(
                                island.population,
                                island.style.emphasis,
                                island.rng,
                            )

                            genome = two_parent_block_mix(
                                left.genome,
                                right.genome,
                                island.rng,
                            )

                            origin = "local_block_crossover"

                        else:
                            label, genome = seed_from_kind(
                                island.style.seed_kind,
                                seeds,
                                island.rng,
                            )

                            origin = f"seed_restart_{label}"

                    else:
                        if value < 0.25:
                            parent = elite[
                                int(
                                    island.rng.integers(
                                        min(2, len(elite))
                                    )
                                )
                            ]

                            genome = parent.genome.copy()

                            origin = "burst_elite_mutation"

                        elif value < 0.58:
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

                            genome = blended_block_mix(
                                parent.genome,
                                donor.genome,
                                island.rng,
                            )

                            origin = "burst_cross_island"

                        elif value < 0.80:
                            label, genome = seed_from_kind(
                                island.style.seed_kind,
                                seeds,
                                island.rng,
                            )

                            origin = f"burst_seed_{label}"

                        else:
                            genome = three_parent_block_mix(
                                seeds["A"],
                                seeds["B"],
                                seeds["C"],
                                island.rng,
                            )

                            origin = "burst_ABC_mix"

                    mutation_probability = (
                        0.90
                        if burst_active
                        else 0.55
                    )

                    mutation_scale = (
                        scale
                        * (
                            cfg.burst_mutation_multiplier
                            if burst_active
                            else 1.0
                        )
                    )

                    genome = local_mutation(
                        genome,
                        island.style.emphasis,
                        island.rng,
                        scale=mutation_scale,
                        probability=mutation_probability,
                    )

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
                            style_name=island.style.name,
                            seed_kind=island.style.seed_kind,
                            emphasis=island.style.emphasis,
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
                            island.style.emphasis,
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
                    and (
                        generation
                        % cfg.migration_interval
                        == 0
                    )
                )
            )
        )

        if migration_now:
            migrant_count = min(
                (
                    8
                    if burst_active
                    else cfg.migrants
                ),
                cfg.population,
            )

            outgoing: list[list[Individual]] = []

            for island in islands:
                rank_and_crowding(island.population)

                migrants = sorted(
                    island.population,
                    key=lambda item: (
                        not item.search_feasible,
                        item.violation,
                        item.rank,
                        -item.crowding,
                        zonal_score(
                            item.result,
                            island.style.emphasis,
                        ),
                    ),
                )[:migrant_count]

                outgoing.append(migrants)

            for source, migrants in enumerate(outgoing):
                destination = (
                    source + 1
                ) % len(islands)

                target = islands[destination]

                incoming: list[Individual] = []

                for migrant in migrants:
                    incoming.append(
                        Individual(
                            genome=migrant.genome.copy(),
                            result=dict(migrant.result),
                            island=destination,
                            generation=generation,
                            origin=(
                                f"migrant_from_{source:02d}_"
                                f"{migrant.seed_kind}"
                            ),
                            style_name=target.style.name,
                            seed_kind=target.style.seed_kind,
                            emphasis=target.style.emphasis,
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
                        target.style.emphasis,
                    ),
                )

        valid = [
            item
            for island in islands
            for item in island.population
            if item.valid
        ]

        current_best = min(
            (
                global_joint_score(item.result)
                for item in valid
            ),
            default=np.inf,
        )

        improved = (
            current_best
            < best_joint_seen
            * (1.0 - cfg.improvement_fraction)
        )

        if improved:
            best_joint_seen = current_best
            stall_count = 0

            if burst_active:
                burst_remaining = 0

        else:
            stall_count += 1

        if (
            burst_active
            and burst_remaining > 0
        ):
            burst_remaining -= 1

        if (
            not burst_active
            and stall_count >= cfg.stall_generations
        ):
            burst_remaining = cfg.burst_generations
            stall_count = 0

            print(
                "Local stagnation burst scheduled: "
                f"{cfg.burst_generations} generations "
                f"after G{generation:03d}.",
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

    animate_surfaces(
        cfg,
        len(islands),
    )

    plot_convergence(
        history,
        cfg.output_dir,
        cfg,
    )

    plot_pareto(
        archive,
        cfg.output_dir,
        cfg,
    )

    candidates = [
        item
        for item in archive
        if item.valid
    ]

    candidates.sort(
        key=lambda item: (
            not item.search_feasible,
            item.violation,
            global_joint_score(item.result),
            item.result["dz_peak_m"],
            item.result["excursion_ev"],
        )
    )

    selected = candidates[: cfg.fine_top]

    fine_rows: list[dict[str, Any]] = []

    for number, item in enumerate(
        tqdm(
            selected,
            desc="Fine validation",
            unit="candidate",
        ),
        start=1,
    ):
        if number <= cfg.adaptive_top:
            result, data, mesh_records = adaptive_verify(
                item.genome, cfg, retain=True
            )
        else:
            result, data = evaluate(
                item.genome, cfg, stage="fine", retain=True, mesh_level=1
            )
            mesh_records = []

        row = {
            "candidate": number,
            **serialize(item),
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
                / f"candidate_{number:03d}"
            )

            if mesh_records:
                save_csv(mesh_records, directory / "mesh_convergence.csv")
                plot_mesh_convergence(mesh_records, directory / "mesh_convergence.png")

            title = (
                f"Fine candidate {number} | "
                f"{item.style_name} | "
                f"{item.origin}"
            )

            fine_plots(
                data,
                result,
                directory,
                title,
                cfg,
            )

            (
                directory
                / "full_candidate.json"
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

    verified = [
        row
        for row in fine_rows
        if bool(row.get("fine_verified", False))
    ]

    search_hits = [
        row
        for row in fine_rows
        if bool(
            row.get("fine_search_feasible", False)
        )
    ]

    selection_pool = verified or search_hits or fine_rows

    top5 = sorted(
        selection_pool,
        key=lambda row: (
            not bool(
                row.get(
                    "fine_final_feasible",
                    False,
                )
            ),
            not bool(
                row.get(
                    "fine_search_feasible",
                    False,
                )
            ),
            float(
                row.get(
                    "fine_search_constraint_violation",
                    np.inf,
                )
            ),
            float(row.get("fine_numerical_uncertainty_ev", np.inf)),
            max(
                float(
                    row.get(
                        "fine_dz_global_ratio_search",
                        np.inf,
                    )
                ),
                float(
                    row.get(
                        "fine_excursion_global_ratio_search",
                        np.inf,
                    )
                ),
            ),
        ),
    )[:5]

    (
        cfg.output_dir
        / "top5_seeded_candidates.json"
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
        / "top5_seeded_candidates.csv",
    )

    (
        cfg.output_dir
        / "verified_hits.json"
    ).write_text(
        json.dumps(
            verified,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    (
        cfg.output_dir
        / "search_hits.json"
    ).write_text(
        json.dumps(
            search_hits,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print(
        f"Search feasible fine hits: "
        f"{len(search_hits)}"
    )

    print(
        f"Strict final fine hits: "
        f"{len(verified)}"
    )

    print(
        "Top candidates: "
        f"{cfg.output_dir / 'top5_seeded_candidates.json'}"
    )


if __name__ == "__main__":
    main()