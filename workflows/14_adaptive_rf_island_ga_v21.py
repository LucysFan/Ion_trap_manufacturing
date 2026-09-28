"""
Workflow 14 v2.1: adaptive seeded zonal island GA for a planar C4v ion-trap X-junction.

The optimisation starts from three known good ordinary full-genome basins:

    A: best joint candidate
    B: best barrier candidate
    C: diverse ordinary candidate

Unlike broad random GA, this workflow is designed for local refinement and
structured recombination around known useful geometries.

Physical zoning
---------------
The transport path is divided into four regions:

    centre      : |x| <= 45 um
    transition  : 45 < |x| <= 160 um
    arm         : 160 < |x| <= 320 um
    reference   : 260 <= |x| <= 320 um

Pseudopotential objective
-------------------------
The workflow measures the absolute deviation relative to the arm reference:

    U_excursion(x) = abs(U_ps(x) - U_arm_reference)

Therefore both a positive barrier and a negative potential well are penalized.

Search target:
    dz_zone < 3 um
    U_excursion_zone < 1 meV

Strict final target:
    dz_zone < 1 um
    U_excursion_zone < 0.1 meV

Run from repository root:

    python workflows/13_seeded_zonal_islands.py --smoke --fast

Recommended first run:

    python workflows/13_seeded_zonal_islands.py `
      --backend auto `
      --population 10 `
      --offspring 3 `
      --generations 20 `
      --coarse-points 51 `
      --fine-points 181 `
      --fine-top 20 `
      --max-panels 6500 `
      --fast `
      --seed 20260927 `
      --output-dir reports/13_seeded_zonal_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go = None
    HAS_PLOTLY = False

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
    "A",
    "B",
    "C",
    "AB",
    "AC",
    "BC",
    "ABC",
]

Emphasis = Literal[
    "centre",
    "transition",
    "arm",
    "height",
    "barrier",
    "balanced",
    "wide",
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

FINAL_DZ_LIMIT_M = 1.0e-6
FINAL_EXCURSION_LIMIT_EV = 0.10e-3
FINAL_LATERAL_LIMIT_M = 1.0e-6

# Gradient targets are optimisation scales, not guaranteed physical bounds.
SEARCH_GRADIENT_L1_LIMIT_EV = 2.5e-3
SEARCH_GRADIENT_PEAK_LIMIT_EV_M = 20.0
FINAL_GRADIENT_L1_LIMIT_EV = 1.0e-3
FINAL_GRADIENT_PEAK_LIMIT_EV_M = 10.0

# Geometry precheck must agree with the project's manufacturability gate.
MIN_INNER_CLEARANCE_M = 10.0e-6
MIN_RAIL_WIDTH_M = 18.0e-6
GENOME_KEY_DECIMALS = 13


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

    search_dz_limit_m: float = SEARCH_DZ_LIMIT_M
    search_excursion_limit_ev: float = SEARCH_EXCURSION_LIMIT_EV
    search_lateral_limit_m: float = SEARCH_LATERAL_LIMIT_M

    final_dz_limit_m: float = FINAL_DZ_LIMIT_M
    final_excursion_limit_ev: float = FINAL_EXCURSION_LIMIT_EV
    final_lateral_limit_m: float = FINAL_LATERAL_LIMIT_M

    search_gradient_l1_limit_ev: float = SEARCH_GRADIENT_L1_LIMIT_EV
    search_gradient_peak_limit_ev_m: float = SEARCH_GRADIENT_PEAK_LIMIT_EV_M
    final_gradient_l1_limit_ev: float = FINAL_GRADIENT_L1_LIMIT_EV
    final_gradient_peak_limit_ev_m: float = FINAL_GRADIENT_PEAK_LIMIT_EV_M

    min_child_survivors: int = 1
    initial_geometry_attempts: int = 80
    child_geometry_attempts: int = 32
    adaptive_levels: int = 3
    adaptive_top: int = 10
    mesh_energy_tolerance_ev: float = 0.05e-3
    mesh_height_tolerance_m: float = 0.10e-6


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
                self.result.get("dz_global_ratio_search", np.inf),
                self.result.get("excursion_global_ratio_search", np.inf),
                self.result.get("gradient_l1_ratio_search", np.inf),
                self.result.get("gradient_peak_ratio_search", np.inf),
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
# Geometry validity, baseline discovery, seed projection, and diagnostics
# =============================================================================

def count_significant_extrema(
    profile_ev: np.ndarray,
    prominence_fraction: float = 0.05,
    absolute_floor_ev: float = 1.0e-6,
) -> tuple[int, int]:
    """Count robust internal minima/maxima after a short binomial smoothing."""
    raw = np.asarray(profile_ev, dtype=float)
    if raw.ndim != 1 or raw.size < 5 or not np.all(np.isfinite(raw)):
        return 0, 0
    smooth = raw.copy()
    smooth[1:-1] = 0.25 * raw[:-2] + 0.50 * raw[1:-1] + 0.25 * raw[2:]
    span = float(np.ptp(smooth))
    threshold = max(absolute_floor_ev, prominence_fraction * span)
    window = max(3, smooth.size // 20)
    minima = maxima = 0
    for index in range(1, smooth.size - 1):
        lo = max(0, index - window)
        hi = min(smooth.size, index + window + 1)
        left = smooth[lo:index]
        right = smooth[index + 1:hi]
        if left.size == 0 or right.size == 0:
            continue
        value = smooth[index]
        if value <= smooth[index - 1] and value <= smooth[index + 1]:
            prominence = min(float(np.max(left) - value), float(np.max(right) - value))
            minima += int(prominence >= threshold)
        if value >= smooth[index - 1] and value >= smooth[index + 1]:
            prominence = min(float(value - np.min(left)), float(value - np.min(right)))
            maxima += int(prominence >= threshold)
    return minima, maxima


def genome_key(genome: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(repair(genome), GENOME_KEY_DECIMALS))


def geometry_precheck(genome: np.ndarray) -> tuple[bool, str]:
    """Cheap gate used before any BEM allocation."""
    try:
        p = parameters(repair(genome))
        report = check_x_junction_manufacturability(p)
        if not report.valid:
            return False, "geometry: " + "; ".join(map(str, report.messages))
        longitudinal_m = np.linspace(0.0, p.arm_length_m, 2001)
        inner_m, outer_m = p.rail_boundaries_m(longitudinal_m)
        if not (np.all(np.isfinite(inner_m)) and np.all(np.isfinite(outer_m))):
            return False, "non-finite rail boundary"
        inner_clearance = float(np.min(inner_m))
        rail_width = float(np.min(outer_m - inner_m))
        if inner_clearance < MIN_INNER_CLEARANCE_M:
            return False, f"inner clearance {inner_clearance * 1e6:.3f} um"
        if rail_width < MIN_RAIL_WIDTH_M:
            return False, f"rail width {rail_width * 1e6:.3f} um"
        return True, "ok"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def default_baseline_genome() -> np.ndarray:
    """Neutral movable-knot genome; adjusted below if the local API rejects it."""
    genome = np.zeros(N_GENES, dtype=float)
    genome[GAPS] = fixed_knots_to_logits(FIXED_CORE_KNOTS_M, min_gap_m=MIN_GAP_M)
    genome[LOCK] = FIXED_CORE_KNOTS_M[-1]
    genome[CENTER_IN] = -25.0e-6
    genome[CENTER_OUT] = 20.0e-6
    genome[START] = 30.0e-6
    genome[LENGTH] = 150.0e-6
    genome[POWER] = 2.0
    genome[BULGE] = 0.0
    genome[BULGE_CENTER] = 70.0e-6
    genome[BULGE_WIDTH] = 25.0e-6
    return repair(genome)


def discover_valid_baseline(
    rng: np.random.Generator,
    attempts: int = 4000,
) -> np.ndarray:
    """Find a valid neutral basin without assuming historical seeds still fit."""
    base = default_baseline_genome()
    valid, _ = geometry_precheck(base)
    if valid:
        return base

    # First vary the parameters that directly control centre clearance/rail width.
    for center_in_um in (-3, -8, -15, -25, -35, -45):
        for center_out_um in (20, 30, 40, 50, 60):
            for start_um in (20, 30, 40, 50, 55):
                candidate = base.copy()
                candidate[CENTER_IN] = center_in_um * 1e-6
                candidate[CENTER_OUT] = center_out_um * 1e-6
                candidate[START] = start_um * 1e-6
                valid, _ = geometry_precheck(candidate)
                if valid:
                    return repair(candidate)

    # Broad but bounded Latin-like random restart.
    for _ in range(attempts):
        candidate = base.copy()
        candidate[GAPS] = rng.uniform(-2.0, 2.0, N_INTERVALS)
        candidate[INNER] = rng.uniform(-12e-6, 12e-6, N_KNOTS - 2)
        candidate[OUTER] = rng.uniform(-12e-6, 20e-6, N_KNOTS - 2)
        candidate[LOCK] = rng.uniform(LOCK_MIN_M, LOCK_MAX_M)
        candidate[CENTER_IN] = rng.uniform(-48e-6, -3e-6)
        candidate[CENTER_OUT] = rng.uniform(10e-6, 60e-6)
        candidate[START] = rng.uniform(12e-6, 54e-6)
        candidate[LENGTH] = rng.uniform(90e-6, 240e-6)
        candidate[POWER] = rng.uniform(1.0, 3.5)
        candidate[BULGE] = rng.uniform(-8e-6, 12e-6)
        candidate[BULGE_CENTER] = rng.uniform(25e-6, 125e-6)
        candidate[BULGE_WIDTH] = rng.uniform(15e-6, 60e-6)
        candidate = repair(candidate)
        valid, _ = geometry_precheck(candidate)
        if valid:
            return candidate
    raise RuntimeError(
        "Could not discover a manufacturable baseline. This usually means "
        "the genome bounds disagree with the current geometry API."
    )


def project_seed_to_valid(
    baseline: np.ndarray,
    seed: np.ndarray,
    iterations: int = 48,
    safety: float = 0.97,
) -> tuple[np.ndarray, float]:
    """Project an old seed onto the current feasible component by bisection."""
    baseline = repair(baseline)
    seed = repair(seed)
    if not geometry_precheck(baseline)[0]:
        raise ValueError("Projection baseline itself is invalid.")
    if geometry_precheck(seed)[0]:
        return seed, 1.0
    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        candidate = repair(baseline + middle * (seed - baseline))
        if geometry_precheck(candidate)[0]:
            low = middle
        else:
            high = middle
    alpha = safety * low
    projected = repair(baseline + alpha * (seed - baseline))
    if not geometry_precheck(projected)[0]:
        return baseline.copy(), 0.0
    return projected, alpha


def valid_mutation(
    base: np.ndarray,
    emphasis: Emphasis,
    rng: np.random.Generator,
    scale: float,
    probability: float,
    forbidden: set[tuple[float, ...]],
    attempts: int,
) -> tuple[np.ndarray, int]:
    """Resample/contract mutation until it is valid and unique."""
    trial_scale = scale
    for attempt in range(1, attempts + 1):
        candidate = local_mutation(base, emphasis, rng, trial_scale, probability)
        key = genome_key(candidate)
        if key not in forbidden and geometry_precheck(candidate)[0]:
            return candidate, attempt
        if attempt % 8 == 0:
            trial_scale *= 0.65
    # Deterministic tiny jitter avoids a silent duplicate fallback.
    for _ in range(100):
        candidate = repair(base + rng.normal(0.0, 1e-4, N_GENES) * (HIGH - LOW))
        key = genome_key(candidate)
        if key not in forbidden and geometry_precheck(candidate)[0]:
            return candidate, attempts + 1
    raise RuntimeError("Unable to create a unique valid genome near the current basin.")


def genome_diversity(population: list[Individual]) -> float:
    if len(population) < 2:
        return 0.0
    matrix = np.vstack([item.genome for item in population])
    span = np.maximum(HIGH - LOW, np.finfo(float).eps)
    normalized = (matrix - LOW) / span
    distances = []
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            distances.append(float(np.linalg.norm(normalized[left] - normalized[right])))
    return float(np.mean(distances)) if distances else 0.0

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


def seed_from_kind(
    kind: SeedKind,
    seeds: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    A = seeds["A"]
    B = seeds["B"]
    C = seeds["C"]

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
    "lateral_ratio_search": np.inf,

    "search_constraint_violation": np.inf,

    "dz_global_ratio_final": np.inf,
    "excursion_global_ratio_final": np.inf,
    "lateral_ratio_final": np.inf,

    "gradient_l1_ev": np.inf,
    "gradient_peak_ev_m": np.inf,
    "gradient_l1_ratio_search": np.inf,
    "gradient_peak_ratio_search": np.inf,
    "gradient_l1_ratio_final": np.inf,
    "gradient_peak_ratio_final": np.inf,
    "n_significant_minima": 0,
    "n_significant_maxima": 0,
    "mesh_level": -1,

    "panels": 0,
    "points": 0,
    "converged": 0,
}
def build_mesh(
    parameter_object: Any,
    level: int,
) -> Any:
    """Geometry-aware refinement ladder; level 0 is screening."""
    ladder = (
        dict(
            central_half_extent_m=130e-6,
            central_max_cell_m=70e-6,
            boundary_max_cell_m=28e-6,
            outer_max_cell_m=220e-6,
            min_cell_m=14e-6,
        ),
        dict(
            central_half_extent_m=180e-6,
            central_max_cell_m=42e-6,
            boundary_max_cell_m=16e-6,
            outer_max_cell_m=200e-6,
            min_cell_m=8e-6,
        ),
        dict(
            central_half_extent_m=210e-6,
            central_max_cell_m=30e-6,
            boundary_max_cell_m=10e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=5e-6,
        ),
        dict(
            central_half_extent_m=240e-6,
            central_max_cell_m=22e-6,
            boundary_max_cell_m=7e-6,
            outer_max_cell_m=160e-6,
            min_cell_m=3.5e-6,
        ),
    )
    return build_geometry_aware_quadtree_x_junction_bem(
        parameter_object,
        **ladder[min(max(int(level), 0), len(ladder) - 1)],
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


def evaluate(
    genome: np.ndarray,
    cfg: Settings,
    stage: str = "coarse",
    retain: bool = False,
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

        if np.min(inner_m) < MIN_INNER_CLEARANCE_M:
            return rejected(
                f"inner clearance < {MIN_INNER_CLEARANCE_M * 1e6:.1f} um",
                stage,
                cfg,
            ), None

        if np.min(outer_m - inner_m) < MIN_RAIL_WIDTH_M:
            return rejected(
                "rail width",
                stage,
                cfg,
            ), None

        if stage == "coarse":
            mesh_level = 0 if cfg.fast else 1
        elif stage.startswith("adaptive_"):
            mesh_level = int(stage.rsplit("_", 1)[1])
        else:
            mesh_level = 2

        model = build_mesh(parameter_object, mesh_level)

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
            cfg.coarse_points
            if stage == "coarse"
            else cfg.fine_points
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

        # RF-only smoothness metrics. Total variation has energy units;
        # the peak derivative has energy-per-length units.
        du_dx_ev_m = np.gradient(u_relative_ev, x_m)
        gradient_l1_ev = float(np.trapz(np.abs(du_dx_ev_m), x_m))
        gradient_peak_ev_m = float(np.max(np.abs(du_dx_ev_m)))

        n_significant_minima, n_significant_maxima = count_significant_extrema(
            u_relative_ev,
            prominence_fraction=0.05,
            absolute_floor_ev=1.0e-6,
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

                gradient_l1_ev,
                gradient_peak_ev_m,
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

        gradient_l1_ratio_search = (
            gradient_l1_ev / cfg.search_gradient_l1_limit_ev
        )
        gradient_peak_ratio_search = (
            gradient_peak_ev_m / cfg.search_gradient_peak_limit_ev_m
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

        gradient_violation_search = (
            max(0.0, gradient_l1_ratio_search - 1.0)
            + max(0.0, gradient_peak_ratio_search - 1.0)
        )

        search_constraint_violation = float(
            dz_violation_search
            + excursion_violation_search
            + lateral_violation_search
            + gradient_violation_search
        )

        search_feasible = bool(
            np.all(dz_zone_ratios_search <= 1.0)
            and np.all(excursion_zone_ratios_search <= 1.0)
            and lateral_ratio_search <= 1.0
            and gradient_l1_ratio_search <= 1.0
            and gradient_peak_ratio_search <= 1.0
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
        gradient_l1_ratio_final = (
            gradient_l1_ev / cfg.final_gradient_l1_limit_ev
        )
        gradient_peak_ratio_final = (
            gradient_peak_ev_m / cfg.final_gradient_peak_limit_ev_m
        )

        final_feasible = bool(
            np.all(dz_zone_ratios_final <= 1.0)
            and np.all(excursion_zone_ratios_final <= 1.0)
            and lateral_ratio_final <= 1.0
            and gradient_l1_ratio_final <= 1.0
            and gradient_peak_ratio_final <= 1.0
        )

        result = {
        "valid": True,
        "reason": "ok",
        "stage": stage,
        "backend": bem.backend_name,

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

        # Constraint state and all objective ratios must be returned.  Earlier
        # versions computed these values but silently dropped them here.
        "search_feasible": search_feasible,
        "final_feasible": final_feasible,
        "verified": False,
        "search_constraint_violation": search_constraint_violation,
        "dz_global_ratio_search": float(np.max(dz_zone_ratios_search)),
        "excursion_global_ratio_search": float(np.max(excursion_zone_ratios_search)),
        "lateral_ratio_search": float(lateral_ratio_search),
        "gradient_l1_ratio_search": float(gradient_l1_ratio_search),
        "gradient_peak_ratio_search": float(gradient_peak_ratio_search),
        "dz_global_ratio_final": float(np.max(dz_zone_ratios_final)),
        "excursion_global_ratio_final": float(np.max(excursion_zone_ratios_final)),
        "lateral_ratio_final": float(lateral_ratio_final),
        "gradient_l1_ratio_final": float(gradient_l1_ratio_final),
        "gradient_peak_ratio_final": float(gradient_peak_ratio_final),
        "gradient_l1_ev": gradient_l1_ev,
        "gradient_peak_ev_m": gradient_peak_ev_m,
        "n_significant_minima": int(n_significant_minima),
        "n_significant_maxima": int(n_significant_maxima),
        "mesh_level": int(mesh_level),

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


def adaptive_validate(
    genome: np.ndarray,
    cfg: Settings,
    retain: bool = True,
) -> tuple[dict[str, Any], Any | None, list[dict[str, Any]]]:
    """Refine until quantities of interest stabilize or the level budget ends."""
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    last_valid: dict[str, Any] | None = None
    last_data: Any | None = None
    converged = False

    levels = max(1, min(cfg.adaptive_levels, 4))
    for level in range(1, levels + 1):
        result, data = evaluate(
            genome,
            cfg,
            stage=f"adaptive_{level}",
            retain=retain,
        )
        row = {
            "level": level,
            "valid": bool(result.get("valid", False)),
            "reason": result.get("reason", ""),
            "panels": result.get("panels", 0),
            "excursion_ev": result.get("excursion_ev", np.inf),
            "gradient_l1_ev": result.get("gradient_l1_ev", np.inf),
            "dz_peak_m": result.get("dz_peak_m", np.inf),
        }
        if result.get("valid", False):
            last_valid = result
            last_data = data
            if previous is not None:
                d_energy = abs(result["excursion_ev"] - previous["excursion_ev"])
                d_height = abs(result["dz_peak_m"] - previous["dz_peak_m"])
                d_gradient = abs(result["gradient_l1_ev"] - previous["gradient_l1_ev"])
                row.update(
                    delta_excursion_ev=d_energy,
                    delta_dz_m=d_height,
                    delta_gradient_l1_ev=d_gradient,
                )
                converged = bool(
                    d_energy <= cfg.mesh_energy_tolerance_ev
                    and d_height <= cfg.mesh_height_tolerance_m
                    and d_gradient <= 2.0 * cfg.mesh_energy_tolerance_ev
                )
            previous = result
        rows.append(row)
        if converged:
            break
        if not result.get("valid", False) and last_valid is not None:
            break

    if last_valid is None:
        result = rejected("adaptive verification produced no valid level", "adaptive", cfg)
        result["mesh_converged"] = False
        result["mesh_uncertainty_ev"] = np.inf
        return result, None, rows

    uncertainty = np.inf
    if len(rows) >= 2 and "delta_excursion_ev" in rows[-1]:
        uncertainty = float(rows[-1]["delta_excursion_ev"])
    final = dict(last_valid)
    final["stage"] = "adaptive"
    final["mesh_converged"] = converged
    final["mesh_uncertainty_ev"] = uncertainty
    final["verified"] = bool(converged and final.get("final_feasible", False))
    if converged:
        final["reason"] = "ok"
    else:
        final["reason"] = "physical result valid; mesh not converged"
    return final, last_data, rows


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
        max(
            dz_c,
            dz_t,
            dz_a,
            u_c,
            u_t,
            u_a,
        )
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

        for axis in range(4):
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
    """
    Default: 24 islands.

    A local        : 4
    B local        : 4
    C local        : 4
    AB mix         : 4
    AC mix         : 3
    BC mix         : 3
    ABC mix        : 2
    Total          : 24
    """
    emphasis_cycle: tuple[Emphasis, ...] = (
        "centre",
        "transition",
        "arm",
        "balanced",
        "height",
        "barrier",
        "wide",
    )

    layout: list[tuple[SeedKind, int]] = [
        ("A", 4),
        ("B", 4),
        ("C", 4),
        ("AB", 4),
        ("AC", 3),
        ("BC", 3),
        ("ABC", 2),
    ]

    styles: list[IslandStyle] = []

    for seed_kind, amount in layout:
        for index in range(amount):
            emphasis = emphasis_cycle[
                len(styles) % len(emphasis_cycle)
            ]

            if seed_kind in ("A", "B", "C"):
                explore_scale = 1.00
                final_scale = 0.25

            elif seed_kind in ("AB", "AC", "BC"):
                explore_scale = 1.20
                final_scale = 0.35

            else:
                explore_scale = 1.40
                final_scale = 0.45

            styles.append(
                IslandStyle(
                    name=(
                        f"{seed_kind.lower()}_"
                        f"{emphasis}_"
                        f"{index:02d}"
                    ),
                    seed_kind=seed_kind,
                    emphasis=emphasis,
                    explore_scale=explore_scale,
                    final_scale=final_scale,
                )
            )

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


def electric_field_surface(
    field: BEMField,
    cfg: Settings,
    extent_m: float = 230e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates_m = np.linspace(-extent_m, extent_m, cfg.surface_points)
    x_grid_m, y_grid_m = np.meshgrid(coordinates_m, coordinates_m, indexing="xy")
    x_flat = x_grid_m.ravel()
    y_flat = y_grid_m.ravel()
    z_flat = np.full_like(x_flat, cfg.target_z_m)
    values = field.bem.field_batch(
        x_flat[None, :], y_flat[None, :], z_flat[None, :],
        field.charge, candidate_chunk=1,
    )
    electric = field.bem.asnumpy(values)[0]
    shape = x_grid_m.shape
    ex = electric[:, 0].reshape(shape)
    ey = electric[:, 1].reshape(shape)
    ez = electric[:, 2].reshape(shape)
    magnitude = np.sqrt(ex**2 + ey**2 + ez**2)
    return x_grid_m, y_grid_m, ex, ey, magnitude


def save_field_and_3d_plots(
    field: BEMField,
    cfg: Settings,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    surface_mev: np.ndarray,
    directory: Path,
    title: str,
) -> None:
    fx, fy, ex, ey, magnitude = electric_field_surface(field, cfg)
    finite = magnitude[np.isfinite(magnitude)]
    vmax = float(np.percentile(finite, 98.0)) if finite.size else 1.0

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.contourf(
        fx * 1e6, fy * 1e6, np.clip(magnitude, 0.0, vmax),
        levels=60, cmap="viridis",
    )
    stride = max(1, cfg.surface_points // 24)
    norm = np.hypot(ex, ey)
    safe = np.where(norm > 0.0, norm, 1.0)
    axis.quiver(
        fx[::stride, ::stride] * 1e6,
        fy[::stride, ::stride] * 1e6,
        (ex / safe)[::stride, ::stride],
        (ey / safe)[::stride, ::stride],
        color="white", alpha=0.72, scale=30.0, width=0.0025,
    )
    axis.set(
        xlabel="x [um]", ylabel="y [um]", aspect="equal",
        title=f"{title}\nRF electric-field magnitude and in-plane direction",
    )
    figure.colorbar(image, ax=axis, label="|E| [V/m per 1 V electrode solution]")
    figure.tight_layout()
    figure.savefig(directory / "rf_field_map.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    finite_u = surface_mev[np.isfinite(surface_mev)]
    clip_top = max(float(np.percentile(finite_u, 97.0)), 1e-9) if finite_u.size else 1.0
    clipped = np.clip(surface_mev, 0.0, clip_top)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x_grid_m * 1e6, y_grid_m * 1e6, clipped,
        cmap="magma", linewidth=0.0, antialiased=True,
        rcount=min(100, clipped.shape[0]), ccount=min(100, clipped.shape[1]),
    )
    axis.set(
        xlabel="x [um]", ylabel="y [um]", zlabel="U_RF [meV]",
        title=f"{title}\nRF pseudopotential surface (97th-percentile clipped)",
    )
    figure.tight_layout()
    figure.savefig(directory / "pseudopotential_3d.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    if HAS_PLOTLY:
        interactive = go.Figure(
            data=[go.Surface(
                x=x_grid_m * 1e6,
                y=y_grid_m * 1e6,
                z=clipped,
                colorscale="Magma",
                colorbar=dict(title="meV"),
            )]
        )
        interactive.update_layout(
            title=title,
            scene=dict(
                xaxis_title="x [um]",
                yaxis_title="y [um]",
                zaxis_title="U_RF [meV]",
                aspectmode="cube",
            ),
        )
        interactive.write_html(
            directory / "pseudopotential_3d_interactive.html",
            include_plotlyjs="cdn",
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

    save_field_and_3d_plots(
        field,
        cfg,
        x_grid_m,
        y_grid_m,
        surface_mev,
        directory,
        title,
    )

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
        4,
        1,
        sharex=True,
        figsize=(10.0, 9.5),
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

    du_dx_mev_um = np.gradient(u_relative_mev, x_um)
    axes[2].plot(x_um, du_dx_mev_um, color="#9467bd")
    axes[2].set_ylabel("dU/dx [meV/um]")

    axes[3].plot(
        x_um,
        np.asarray(trace.y_m) * 1e6,
        color="#2ca02c",
    )

    axes[3].axhline(cfg.search_lateral_limit_m * 1e6, color="darkorange", linestyle=":")
    axes[3].axhline(-cfg.search_lateral_limit_m * 1e6, color="darkorange", linestyle=":")
    axes[3].axhline(cfg.final_lateral_limit_m * 1e6, color="crimson", linestyle="--")
    axes[3].axhline(-cfg.final_lateral_limit_m * 1e6, color="crimson", linestyle="--")
    axes[3].set(xlabel="x [um]", ylabel="y [um]")

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
        default=30,
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

    command.add_argument(
        "--max-panels",
        type=int,
        default=6500,
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
        default=10,
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
        "--adaptive-levels",
        type=int,
        default=3,
    )

    command.add_argument(
        "--adaptive-top",
        type=int,
        default=10,
    )

    command.add_argument(
        "--min-child-survivors",
        type=int,
        default=1,
        help="Minimum new-generation members retained on each island.",
    )

    command.add_argument(
        "--initial-geometry-attempts",
        type=int,
        default=80,
    )

    command.add_argument(
        "--child-geometry-attempts",
        type=int,
        default=32,
    )

    command.add_argument(
        "--self-test",
        action="store_true",
        help="Check metrics, geometry projection, uniqueness, and mutation without BEM.",
    )

    command.add_argument(
        "--smoke",
        action="store_true",
    )

    return command


def run_self_test(seed: int) -> None:
    rng = np.random.default_rng(seed)
    baseline = discover_valid_baseline(rng)
    assert geometry_precheck(baseline)[0]
    projected = {}
    for name, old in elite_seeds().items():
        genome, alpha = project_seed_to_valid(baseline, old)
        assert geometry_precheck(genome)[0]
        projected[name] = alpha
    forbidden = {genome_key(baseline)}
    child, _ = valid_mutation(
        baseline, "wide", rng, 0.5, 0.8, forbidden, 80,
    )
    assert geometry_precheck(child)[0]
    assert genome_key(child) != genome_key(baseline)
    x = np.linspace(-1.0, 1.0, 101)
    profile = 1e-3 * (x**4 - 0.5 * x**2)
    minima, maxima = count_significant_extrema(profile)
    assert minima >= 1 and maxima >= 1
    print("SELF-TEST OK")
    print("seed projection alpha:", projected)
    print("unique valid mutation: OK")
    print("profile metrics: OK")


def main() -> None:
    arguments = parser().parse_args()

    if arguments.self_test:
        run_self_test(arguments.seed)
        return

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
        arguments.max_panels = 5000
        arguments.fast = True
        arguments.animation_stride = 1
        arguments.surface_points = 61
        arguments.stall_generations = 1
        arguments.burst_generations = 1
        arguments.adaptive_levels = 2
        arguments.adaptive_top = 3

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

        output_dir=arguments.output_dir,
        min_child_survivors=arguments.min_child_survivors,
        initial_geometry_attempts=arguments.initial_geometry_attempts,
        child_geometry_attempts=arguments.child_geometry_attempts,
        adaptive_levels=arguments.adaptive_levels,
        adaptive_top=arguments.adaptive_top,
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
        f"Workflow 14 v2.1 | backend={backend.selected}"
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

    raw_seeds = elite_seeds()

    master_rng = np.random.default_rng(cfg.seed)
    baseline = discover_valid_baseline(master_rng)
    baseline_ok, baseline_reason = geometry_precheck(baseline)
    if not baseline_ok:
        raise RuntimeError(f"Internal baseline failure: {baseline_reason}")

    seeds: dict[str, np.ndarray] = {}
    seed_projection: dict[str, float] = {}
    for name, raw_seed in raw_seeds.items():
        projected, alpha = project_seed_to_valid(baseline, raw_seed)
        seeds[name] = projected
        seed_projection[name] = alpha
        status = "accepted" if alpha >= 0.999999 else "projected"
        print(f"Seed {name}: {status}, alpha={alpha:.6f}", flush=True)

    cache: dict[
        tuple[float, ...],
        dict[str, Any],
    ] = {}

    cache_hits = 0
    cache_misses = 0

    archive: list[Individual] = []

    history: list[dict[str, Any]] = []

    rejection_rows: list[dict[str, Any]] = []

    def coarse(
        genome: np.ndarray,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        nonlocal cache_hits, cache_misses
        values = repair(genome)
        key = genome_key(values)

        if key in cache:
            cache_hits += 1
        else:
            cache_misses += 1
            valid_geometry, reason = geometry_precheck(values)
            if not valid_geometry:
                cache[key] = rejected(reason, "coarse", cfg)
            else:
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

            if not geometry_precheck(genome)[0]:
                genome, alpha = project_seed_to_valid(baseline, genome)
                label = f"{label}_projected_{alpha:.5f}"

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

                if not geometry_precheck(base)[0]:
                    base, _ = project_seed_to_valid(baseline, base)
                forbidden = {genome_key(item.genome) for item in population}
                try:
                    genome, _attempts = valid_mutation(
                        base,
                        style.emphasis,
                        rng,
                        scale=(0.55 if arguments.smoke else 1.0) * style.explore_scale,
                        probability=0.75,
                        forbidden=forbidden,
                        attempts=cfg.initial_geometry_attempts,
                    )
                except RuntimeError:
                    genome, _attempts = valid_mutation(
                        baseline,
                        "wide",
                        rng,
                        scale=0.5,
                        probability=0.75,
                        forbidden=forbidden,
                        attempts=cfg.initial_geometry_attempts,
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
            return float(
                max(
                    dz_ratio,
                    excursion_ratio,
                    lateral_ratio,
                    finite_float("gradient_l1_ratio_search", 0.0),
                    finite_float("gradient_peak_ratio_search", 0.0),
                )
            )

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
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "genome_diversity": genome_diversity(all_items),
            "new_generation_members": sum(
                item.generation == generation for item in all_items
            ) if generation > 0 else len(all_items),

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

                    if not geometry_precheck(genome)[0]:
                        genome, _ = project_seed_to_valid(baseline, genome)
                    forbidden = {
                        genome_key(item.genome)
                        for item in island.population + children
                    }
                    genome, _mutation_attempts = valid_mutation(
                        genome,
                        island.style.emphasis,
                        island.rng,
                        scale=mutation_scale,
                        probability=mutation_probability,
                        forbidden=forbidden,
                        attempts=cfg.child_geometry_attempts,
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

                selected_population = survivors(
                    island.population + children,
                    cfg.population,
                    island.style.emphasis,
                )

                # In tiny smoke populations pure elitism can hide all offspring.
                # Retain a bounded number of the best valid children so population
                # turnover is observable without sacrificing the best parent.
                required_children = min(
                    cfg.min_child_survivors,
                    len(children),
                    max(0, cfg.population - 1),
                )
                present_child_keys = {
                    genome_key(item.genome)
                    for item in selected_population
                    if item.generation == generation
                }
                ranked_children = survivors(
                    children,
                    len(children),
                    island.style.emphasis,
                )
                missing = [
                    child for child in ranked_children
                    if genome_key(child.genome) not in present_child_keys
                ]
                current_children = sum(
                    item.generation == generation for item in selected_population
                )
                replacements = max(0, required_children - current_children)
                if replacements and missing:
                    protected = min(
                        selected_population,
                        key=lambda item: (
                            not item.search_feasible,
                            item.violation,
                            item.rank,
                            zonal_score(item.result, island.style.emphasis),
                        ),
                    )
                    removable = sorted(
                        [item for item in selected_population if item is not protected],
                        key=lambda item: (
                            item.search_feasible,
                            -item.violation if np.isfinite(item.violation) else -np.inf,
                            -item.rank,
                            -zonal_score(item.result, island.style.emphasis),
                        ),
                    )
                    for child, old_item in zip(missing[:replacements], removable):
                        # Do not call list.remove on Individual: dataclass equality
                        # compares NumPy genomes and raises an ambiguous-truth error.
                        replace_index = next(
                            index for index, current in enumerate(selected_population)
                            if current is old_item
                        )
                        selected_population[replace_index] = child.copy()
                    rank_and_crowding(selected_population)

                next_islands.append(
                    Island(
                        style=island.style,
                        rng=island.rng,
                        population=selected_population,
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
            result, data, mesh_rows = adaptive_validate(
                item.genome,
                cfg,
                retain=True,
            )
        else:
            result, data = evaluate(
                item.genome,
                cfg,
                stage="fine",
                retain=True,
            )
            mesh_rows = []

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

            if mesh_rows:
                save_csv(mesh_rows, directory / "mesh_convergence.csv")

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