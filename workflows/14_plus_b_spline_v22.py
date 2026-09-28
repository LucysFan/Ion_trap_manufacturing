"""
Workflow 14+ v2.2: hybrid B-spline memetic island GA for a planar C4v ion-trap X-junction.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import least_squares
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
try:
    from config.numerical import RF_NULL_MAX_HEIGHT_M, RF_NULL_MIN_HEIGHT_M
except ImportError:
    RF_NULL_MIN_HEIGHT_M = 1.0e-9
    RF_NULL_MAX_HEIGHT_M = 500.0e-6
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.rf_null_trace import RFNullTrace, trace_rf_transverse_minimum
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

# Hybrid B-spline / memetic search controls.
BSPLINE_DEGREE = 3
BSPLINE_CONTROL_POINTS = 6
BSPLINE_SAMPLE_POINTS = 161
BSPLINE_ISLAND_FRACTION = 0.34
MEMETIC_ISLAND_FRACTION = 0.17
MEMETIC_INTERVAL = 2
MEMETIC_STEPS = 3
MEMETIC_CANDIDATES = 8
MEMETIC_RADIUS_START = 0.055
MEMETIC_RADIUS_END = 0.008
MIN_GENOME_DISTANCE = 2.5e-3
MIN_INNER_CLEARANCE_M = 0.5e-6
MIN_RAIL_WIDTH_M = 18e-6


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

    bspline_island_fraction: float = BSPLINE_ISLAND_FRACTION
    memetic_island_fraction: float = MEMETIC_ISLAND_FRACTION
    memetic_interval: int = MEMETIC_INTERVAL
    memetic_steps: int = MEMETIC_STEPS
    memetic_candidates: int = MEMETIC_CANDIDATES
    min_genome_distance: float = MIN_GENOME_DISTANCE
    spline_trace: bool = True
    spline_trace_anchors: int = 13
    spline_trace_max_nfev: int = 80


@dataclass(frozen=True)
class IslandStyle:
    name: str
    seed_kind: SeedKind
    emphasis: Emphasis
    explore_scale: float
    final_scale: float
    operator: Literal["full", "bspline", "memetic"] = "full"


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
                self.result["dz_global_ratio_search"],
                self.result["excursion_global_ratio_search"],
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
# Hybrid B-spline and memetic operators
# =============================================================================

def normalized_genome(genome: np.ndarray) -> np.ndarray:
    span = np.maximum(HIGH - LOW, np.finfo(float).eps)
    return (repair(genome) - LOW) / span


def denormalized_genome(values: np.ndarray) -> np.ndarray:
    return repair(LOW + np.clip(values, 0.0, 1.0) * (HIGH - LOW))


def genome_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((normalized_genome(left) - normalized_genome(right)) ** 2)))


def open_uniform_knots(n_control: int, degree: int = BSPLINE_DEGREE) -> np.ndarray:
    if n_control <= degree:
        raise ValueError("n_control must exceed spline degree")
    interior_count = n_control - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        if interior_count > 0 else np.empty(0)
    )
    return np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]


def spline_basis(sample_u: np.ndarray, n_control: int = BSPLINE_CONTROL_POINTS) -> np.ndarray:
    knots = open_uniform_knots(n_control)
    basis = np.empty((len(sample_u), n_control), dtype=float)
    for index in range(n_control):
        coefficients = np.zeros(n_control, dtype=float)
        coefficients[index] = 1.0
        basis[:, index] = BSpline(knots, coefficients, BSPLINE_DEGREE)(sample_u)
    return basis


def bspline_smooth_values(values: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Project a vector onto a low-dimensional clamped cubic B-spline basis."""
    original = np.asarray(values, dtype=float)
    if original.ndim != 1 or original.size < 4:
        return original.copy()
    u = np.linspace(0.0, 1.0, original.size)
    n_control = min(BSPLINE_CONTROL_POINTS, original.size)
    degree = min(BSPLINE_DEGREE, n_control - 1)
    if degree != BSPLINE_DEGREE:
        return original.copy()
    basis = spline_basis(u, n_control)
    coefficients, *_ = np.linalg.lstsq(basis, original, rcond=None)
    fitted = basis @ coefficients
    amount = float(np.clip(strength, 0.0, 1.0))
    return (1.0 - amount) * original + amount * fitted


def bspline_genome_operator(
    genome: np.ndarray,
    rng: np.random.Generator,
    scale: float,
    mutation_probability: float = 0.80,
) -> np.ndarray:
    """Low-dimensional smooth contour mutation mapped back to full genome."""
    child = repair(genome)
    normalized = normalized_genome(child)
    for block in (INNER, OUTER):
        values = normalized[block]
        u = np.linspace(0.0, 1.0, values.size)
        n_control = min(BSPLINE_CONTROL_POINTS, values.size)
        basis = spline_basis(u, n_control)
        coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
        active = rng.random(n_control) < mutation_probability
        if not np.any(active):
            active[int(rng.integers(n_control))] = True
        coefficients[active] += rng.normal(0.0, 0.055 * scale, np.sum(active))
        normalized[block] = np.clip(basis @ coefficients, 0.0, 1.0)
    # Knot logits and physical scalars retain full representation, but move more softly.
    normalized[GAPS] += rng.normal(0.0, 0.018 * scale, N_INTERVALS)
    scalar_slice = slice(LOCK, N_GENES)
    scalar_active = rng.random(N_GENES - LOCK) < 0.45
    normalized[scalar_slice][scalar_active] += rng.normal(
        0.0, 0.025 * scale, np.sum(scalar_active)
    )
    return denormalized_genome(normalized)


def spline_profile_extrema(
    x_m: np.ndarray,
    profile_ev: np.ndarray,
) -> tuple[int, int]:
    """Use a cubic interpolating spline only to locate extrema robustly."""
    x = np.asarray(x_m, dtype=float)
    values = np.asarray(profile_ev, dtype=float)
    if x.size < 5 or not np.all(np.isfinite(values)):
        return 0, 0
    spline = make_interp_spline(x, values, k=min(3, x.size - 1))
    dense_x = np.linspace(x[0], x[-1], max(801, 8 * x.size))
    derivative = spline.derivative()(dense_x)
    second = spline.derivative(2)
    roots: list[float] = []
    for index in range(dense_x.size - 1):
        left_d = float(derivative[index])
        right_d = float(derivative[index + 1])
        if left_d * right_d < 0.0:
            roots.append(0.5 * (dense_x[index] + dense_x[index + 1]))
    for index in range(1, dense_x.size - 1):
        if derivative[index] == 0.0 and derivative[index - 1] * derivative[index + 1] < 0.0:
            roots.append(float(dense_x[index]))
    roots = sorted({round(value, 15) for value in roots})
    minima = maxima = 0
    threshold = max(1e-10, 1e-4 * float(np.ptp(values)))
    for root_x in roots:
        center = float(spline(root_x))
        left = float(spline(max(x[0], root_x - 0.03 * (x[-1] - x[0]))))
        right = float(spline(min(x[-1], root_x + 0.03 * (x[-1] - x[0]))))
        prominence = min(abs(center - left), abs(center - right))
        if prominence < threshold:
            continue
        if float(second(root_x)) > 0.0:
            minima += 1
        else:
            maxima += 1
    return minima, maxima


def trace_rf_minimum_bspline(
    field: BEMField,
    x_m: np.ndarray,
    initial_y_m: float,
    initial_z_m: float,
    residual_tolerance_v_m: float,
    max_transverse_shift_m: float,
    anchor_count: int,
    max_nfev: int,
) -> tuple[RFNullTrace, dict[str, int | bool]]:
    """Sparse exact anchors, cubic prediction, selective exact correction."""
    x = np.asarray(x_m, dtype=float)
    anchor_count = int(np.clip(anchor_count, 5, len(x)))
    anchor_indices = np.unique(np.linspace(0, len(x) - 1, anchor_count).round().astype(int))
    anchor_trace = trace_rf_transverse_minimum(
        field,
        x[anchor_indices],
        initial_y_m=initial_y_m,
        initial_z_m=initial_z_m,
        residual_tolerance_v_m=residual_tolerance_v_m,
        max_transverse_shift_m=max_transverse_shift_m,
    )
    if not anchor_trace.valid:
        exact = trace_rf_transverse_minimum(
            field, x, initial_y_m=initial_y_m, initial_z_m=initial_z_m,
            residual_tolerance_v_m=residual_tolerance_v_m,
            max_transverse_shift_m=max_transverse_shift_m,
        )
        return exact, {"used": False, "anchors": len(anchor_indices), "corrections": len(x)}

    order = min(3, len(anchor_indices) - 1)
    y_predictor = make_interp_spline(anchor_trace.x_m, anchor_trace.y_m, k=order)
    z_predictor = make_interp_spline(anchor_trace.x_m, anchor_trace.z_m, k=order)
    y = np.asarray(y_predictor(x), dtype=float)
    z = np.asarray(z_predictor(x), dtype=float)
    z_lower = float(RF_NULL_MIN_HEIGHT_M)
    z_upper = float(RF_NULL_MAX_HEIGHT_M)

    values = field.bem.field_batch(
        x[None, :], y[None, :], z[None, :], field.charge, candidate_chunk=1,
    )
    fields = np.asarray(field.bem.asnumpy(values)[0], dtype=float)
    transverse = np.linalg.norm(fields[:, 1:3], axis=1)
    full_norm = np.linalg.norm(fields, axis=1)
    converged = np.isfinite(transverse) & (transverse <= residual_tolerance_v_m)
    corrections = 0
    messages = ["B-spline prediction accepted" for _ in x]

    anchor_lookup = {int(index): position for position, index in enumerate(anchor_indices)}
    for index in range(len(x)):
        if index in anchor_lookup:
            position = anchor_lookup[index]
            y[index] = anchor_trace.y_m[position]
            z[index] = anchor_trace.z_m[position]
            transverse[index] = anchor_trace.transverse_residual_v_m[position]
            full_norm[index] = anchor_trace.full_field_norm_v_m[position]
            converged[index] = anchor_trace.converged[position]
            messages[index] = "exact B-spline anchor"
            continue
        if converged[index]:
            continue

        x_value = float(x[index])
        prediction = np.array([y[index], z[index]], dtype=float)
        lower = np.array([prediction[0] - max_transverse_shift_m, z_lower])
        upper = np.array([prediction[0] + max_transverse_shift_m, z_upper])

        def residual(yz: np.ndarray) -> np.ndarray:
            vector = np.asarray(
                field.electric_field(x_value, float(yz[0]), float(yz[1])),
                dtype=float,
            )
            return vector[1:3]

        correction = least_squares(
            residual,
            x0=np.clip(prediction, lower, upper),
            bounds=(lower, upper),
            xtol=1e-11, ftol=1e-11, gtol=1e-11,
            max_nfev=max_nfev,
        )
        corrections += 1
        y[index], z[index] = correction.x
        vector = np.asarray(field.electric_field(x_value, y[index], z[index]), dtype=float)
        transverse[index] = float(np.linalg.norm(vector[1:3]))
        full_norm[index] = float(np.linalg.norm(vector))
        converged[index] = bool(
            correction.success
            and np.isfinite(transverse[index])
            and transverse[index] <= residual_tolerance_v_m
        )
        messages[index] = f"B-spline correction: {correction.message}"

    trace = RFNullTrace(
        x_m=x,
        y_m=y,
        z_m=z,
        transverse_residual_v_m=transverse,
        full_field_norm_v_m=full_norm,
        converged=converged,
        messages=tuple(messages),
    )
    if not trace.valid:
        exact = trace_rf_transverse_minimum(
            field, x, initial_y_m=initial_y_m, initial_z_m=initial_z_m,
            residual_tolerance_v_m=residual_tolerance_v_m,
            max_transverse_shift_m=max_transverse_shift_m,
        )
        return exact, {"used": False, "anchors": len(anchor_indices), "corrections": len(x)}
    return trace, {"used": True, "anchors": len(anchor_indices), "corrections": corrections}


def geometry_precheck(genome: np.ndarray) -> tuple[bool, str]:
    try:
        p = parameters(repair(genome))
        report = check_x_junction_manufacturability(p)
        if not report.valid:
            return False, "geometry: " + "; ".join(map(str, report.messages))
        s = np.linspace(0.0, p.arm_length_m, 1001)
        inner, outer = p.rail_boundaries_m(s)
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
            return False, "non-finite rail boundary"
        if float(np.min(inner)) <= MIN_INNER_CLEARANCE_M:
            return False, "inner clearance"
        if float(np.min(outer - inner)) < MIN_RAIL_WIDTH_M:
            return False, "rail width"
        return True, "ok"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def conventional_baseline() -> np.ndarray:
    genome = np.zeros(N_GENES, dtype=float)
    genome[LOCK] = FIXED_CORE_KNOTS_M[-1]
    genome[GAPS] = fixed_knots_to_logits(FIXED_CORE_KNOTS_M, min_gap_m=MIN_GAP_M)
    genome[CENTER_IN:] = [-25e-6, 20e-6, 30e-6, 150e-6, 2.0, 0.0, 70e-6, 25e-6]
    return repair(genome)


def discover_valid_baseline(rng: np.random.Generator, attempts: int = 3000) -> np.ndarray:
    baseline = conventional_baseline()
    if geometry_precheck(baseline)[0]:
        return baseline
    for _ in range(attempts):
        candidate = baseline.copy()
        candidate[INNER] = rng.uniform(-12e-6, 12e-6, N_KNOTS - 2)
        candidate[OUTER] = rng.uniform(-12e-6, 18e-6, N_KNOTS - 2)
        candidate[CENTER_IN] = rng.uniform(-48e-6, -3e-6)
        candidate[CENTER_OUT] = rng.uniform(10e-6, 60e-6)
        candidate[START] = rng.uniform(12e-6, 54e-6)
        candidate = repair(candidate)
        if geometry_precheck(candidate)[0]:
            return candidate
    raise RuntimeError("Could not find a manufacturable baseline within genome bounds")


def project_to_valid(baseline: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float]:
    baseline, candidate = repair(baseline), repair(candidate)
    if geometry_precheck(candidate)[0]:
        return candidate, 1.0
    if not geometry_precheck(baseline)[0]:
        raise ValueError("projection baseline is invalid")
    low, high = 0.0, 1.0
    for _ in range(48):
        alpha = 0.5 * (low + high)
        trial = repair(baseline + alpha * (candidate - baseline))
        if geometry_precheck(trial)[0]:
            low = alpha
        else:
            high = alpha
    alpha = 0.97 * low
    return repair(baseline + alpha * (candidate - baseline)), alpha


def unique_valid_candidate(
    candidate_factory: Callable[[], np.ndarray],
    baseline: np.ndarray,
    forbidden: list[np.ndarray],
    rng: np.random.Generator,
    attempts: int = 40,
) -> np.ndarray:
    for _ in range(attempts):
        candidate = repair(candidate_factory())
        if not geometry_precheck(candidate)[0]:
            candidate, _ = project_to_valid(baseline, candidate)
        if geometry_precheck(candidate)[0] and all(
            genome_distance(candidate, old) >= MIN_GENOME_DISTANCE for old in forbidden
        ):
            return candidate
    # Last-resort broad valid immigrant.
    for _ in range(500):
        normalized = normalized_genome(baseline)
        normalized += rng.normal(0.0, 0.04, N_GENES)
        candidate = denormalized_genome(normalized)
        candidate, _ = project_to_valid(baseline, candidate)
        if all(genome_distance(candidate, old) >= 0.25 * MIN_GENOME_DISTANCE for old in forbidden):
            return candidate
    return baseline.copy()


def memetic_candidates(
    elite: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
    amount: int,
) -> list[np.ndarray]:
    """Derivative-free trust-region coordinate/spline proposals."""
    center = normalized_genome(elite)
    radius = MEMETIC_RADIUS_START * (1.0 - fraction) + MEMETIC_RADIUS_END * fraction
    proposals: list[np.ndarray] = []
    for index in range(amount):
        trial = center.copy()
        if index < 4:
            block = (INNER, OUTER, GAPS, slice(LOCK, N_GENES))[index]
            direction = rng.normal(0.0, 1.0, trial[block].shape)
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                trial[block] += radius * direction / norm
        else:
            active = rng.random(N_GENES) < 0.22
            trial[active] += rng.normal(0.0, radius, np.sum(active))
        proposal = denormalized_genome(trial)
        if index % 2 == 0:
            proposal = bspline_genome_operator(proposal, rng, max(0.25, 1.0 - fraction), 0.65)
        proposals.append(proposal)
    return proposals

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
    "spline_minima": 0,
    "spline_maxima": 0,

    "dz_global_ratio_final": np.inf,
    "excursion_global_ratio_final": np.inf,
    "lateral_ratio_final": np.inf,

    "panels": 0,
    "points": 0,
    "converged": 0,
}
def build_mesh(
    parameter_object: Any,
    fast: bool,
) -> Any:
    if fast:
        return build_geometry_aware_quadtree_x_junction_bem(
            parameter_object,
            central_half_extent_m=130e-6,
            central_max_cell_m=70e-6,
            boundary_max_cell_m=28e-6,
            outer_max_cell_m=220e-6,
            min_cell_m=14e-6,
        )

    return build_geometry_aware_quadtree_x_junction_bem(
        parameter_object,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
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

        model = build_mesh(
            parameter_object,
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

        trace_x_m = np.linspace(-350e-6, 350e-6, points)
        if stage == "coarse" and cfg.spline_trace and points >= 9:
            trace, spline_trace_info = trace_rf_minimum_bspline(
                field,
                trace_x_m,
                initial_y_m=0.0,
                initial_z_m=cfg.target_z_m,
                residual_tolerance_v_m=1e-3,
                max_transverse_shift_m=25e-6,
                anchor_count=cfg.spline_trace_anchors,
                max_nfev=cfg.spline_trace_max_nfev,
            )
        else:
            trace = trace_rf_transverse_minimum(
                field,
                trace_x_m,
                initial_y_m=0.0,
                initial_z_m=cfg.target_z_m,
                residual_tolerance_v_m=1e-3,
                max_transverse_shift_m=25e-6,
            )
            spline_trace_info = {"used": False, "anchors": points, "corrections": points}

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

        spline_minima, spline_maxima = spline_profile_extrema(
            x_m,
            u_relative_ev,
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
        )

        search_feasible = bool(
            np.all(dz_zone_ratios_search <= 1.0)
            and np.all(
                excursion_zone_ratios_search <= 1.0
            )
            and lateral_ratio_search <= 1.0
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

        final_feasible = bool(
            np.all(dz_zone_ratios_final <= 1.0)
            and np.all(
                excursion_zone_ratios_final <= 1.0
            )
            and lateral_ratio_final <= 1.0
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

        # Computed constraint state must be returned to NSGA-II.
        "search_feasible": search_feasible,
        "final_feasible": final_feasible,
        "verified": bool(stage == "fine" and final_feasible),
        "search_constraint_violation": search_constraint_violation,
        "dz_global_ratio_search": float(np.max(dz_zone_ratios_search)),
        "excursion_global_ratio_search": float(np.max(excursion_zone_ratios_search)),
        "lateral_ratio_search": float(lateral_ratio_search),
        "dz_global_ratio_final": float(np.max(dz_zone_ratios_final)),
        "excursion_global_ratio_final": float(np.max(excursion_zone_ratios_final)),
        "lateral_ratio_final": float(lateral_ratio_final),
        "spline_minima": int(spline_minima),
        "spline_maxima": int(spline_maxima),
        "spline_trace_used": bool(spline_trace_info["used"]),
        "spline_trace_anchors": int(spline_trace_info["anchors"]),
        "spline_trace_corrections": int(spline_trace_info["corrections"]),

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

        for axis in range(2):
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

    bspline_count = max(1, int(round(len(styles) * BSPLINE_ISLAND_FRACTION)))
    memetic_count = max(1, int(round(len(styles) * MEMETIC_ISLAND_FRACTION)))
    for index, style in enumerate(styles):
        operator = (
            "bspline" if index < bspline_count
            else "memetic" if index < bspline_count + memetic_count
            else "full"
        )
        styles[index] = IslandStyle(
            name=f"{style.name}_{operator}",
            seed_kind=style.seed_kind,
            emphasis=style.emphasis,
            explore_scale=style.explore_scale,
            final_scale=style.final_scale,
            operator=operator,
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
        3,
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

    axes[2].plot(
        x_um,
        np.asarray(trace.y_m) * 1e6,
        color="#2ca02c",
    )

    axes[2].axhline(
        cfg.search_lateral_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[2].axhline(
        -cfg.search_lateral_limit_m * 1e6,
        color="darkorange",
        linestyle=":",
    )

    axes[2].axhline(
        cfg.final_lateral_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[2].axhline(
        -cfg.final_lateral_limit_m * 1e6,
        color="crimson",
        linestyle="--",
    )

    axes[2].set(
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

    command.add_argument("--memetic-interval", type=int, default=MEMETIC_INTERVAL)
    command.add_argument("--memetic-steps", type=int, default=MEMETIC_STEPS)
    command.add_argument("--memetic-candidates", type=int, default=MEMETIC_CANDIDATES)
    command.add_argument("--spline-trace-anchors", type=int, default=13)
    command.add_argument("--no-spline-trace", action="store_true")
    command.add_argument("--self-test", action="store_true")
    command.add_argument(
        "--smoke",
        action="store_true",
    )

    return command


def run_self_test(seed: int) -> None:
    rng = np.random.default_rng(seed)
    source = np.array([0.0, 0.7, -0.4, 0.8, -0.2, 0.3, 0.0])
    smooth = bspline_smooth_values(source)
    assert smooth.shape == source.shape and np.all(np.isfinite(smooth))
    x = np.linspace(-1.0, 1.0, 101)
    valley = (x - 0.137)**2
    hill = -((x + 0.173)**2)
    valley_minima, _ = spline_profile_extrema(x, valley)
    _, hill_maxima = spline_profile_extrema(x, hill)
    if valley_minima < 1 or hill_maxima < 1:
        raise RuntimeError(
            "B-spline extrema self-test failed: "
            f"valley_minima={valley_minima}, hill_maxima={hill_maxima}"
        )
    baseline = discover_valid_baseline(rng)
    child = bspline_genome_operator(baseline, rng, 0.5)
    child, _ = project_to_valid(baseline, child)
    assert geometry_precheck(baseline)[0] and geometry_precheck(child)[0]
    proposals = memetic_candidates(baseline, rng, 0.2, 8)
    assert len(proposals) == 8 and all(item.shape == (N_GENES,) for item in proposals)
    print("SELF-TEST OK: B-spline, extrema, projection, memetic proposals")


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
        memetic_interval=arguments.memetic_interval,
        memetic_steps=arguments.memetic_steps,
        memetic_candidates=arguments.memetic_candidates,
        spline_trace=not arguments.no_spline_trace,
        spline_trace_anchors=arguments.spline_trace_anchors,
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
        f"Workflow 14+ v2.2 | backend={backend.selected}"
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
    seeds: dict[str, np.ndarray] = {}
    for name, raw_seed in raw_seeds.items():
        seeds[name], alpha = project_to_valid(baseline, raw_seed)
        print(f"Seed {name}: feasible projection alpha={alpha:.6f}", flush=True)

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

        key = tuple(np.round(values, 13))

        if key not in cache:
            valid_geometry, reason = geometry_precheck(values)
            if valid_geometry:
                cache[key], _ = evaluate(values, cfg, stage="coarse", retain=False)
            else:
                cache[key] = rejected(reason, "coarse", cfg)

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

            genome, _ = project_to_valid(baseline, genome)

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

                forbidden = [item.genome for item in population]
                if style.operator == "bspline":
                    factory = lambda: bspline_genome_operator(base, rng, style.explore_scale, 0.80)
                else:
                    factory = lambda: local_mutation(
                        base, style.emphasis, rng,
                        scale=style.explore_scale, probability=0.75,
                    )
                genome = unique_valid_candidate(factory, baseline, forbidden, rng)

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
            "bspline_members": sum(
                island.style.operator == "bspline" for island in islands
                for _item in island.population
            ),
            "memetic_members": sum(
                island.style.operator == "memetic" for island in islands
                for _item in island.population
            ),
            "mean_genome_distance": float(np.mean([
                genome_distance(all_items[i].genome, all_items[j].genome)
                for i in range(len(all_items))
                for j in range(i + 1, len(all_items))
            ])) if len(all_items) > 1 else 0.0,
            "best_spline_minima": min(
                (item.result.get("spline_minima", 0) for item in valid),
                default=0,
            ),

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

        memetic_active = (
            cfg.memetic_interval > 0
            and generation % cfg.memetic_interval == 0
        )
        memetic_extra = sum(
            cfg.memetic_steps * cfg.memetic_candidates
            for island in islands
            if memetic_active and island.style.operator == "memetic"
        )

        with tqdm(
            total=(
                sum(
                    cfg.offspring
                    for island in islands
                    if not (memetic_active and island.style.operator == "memetic")
                )
                + memetic_extra
            ),
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

                if memetic_active and island.style.operator == "memetic":
                    memetic_center = elite[0].genome.copy()
                    for memetic_step in range(cfg.memetic_steps):
                        proposals = memetic_candidates(
                            memetic_center,
                            island.rng,
                            min(1.0, fraction + memetic_step / max(1, cfg.memetic_steps)),
                            cfg.memetic_candidates,
                        )
                        step_children: list[Individual] = []
                        for proposal_number, proposal in enumerate(proposals):
                            forbidden = [
                                item.genome
                                for item in island.population + children + step_children
                            ]
                            proposal = unique_valid_candidate(
                                lambda proposal=proposal: proposal,
                                baseline,
                                forbidden,
                                island.rng,
                            )
                            candidate = Individual(
                                genome=proposal,
                                result=coarse(proposal, progress),
                                island=island_id,
                                generation=generation,
                                origin=(
                                    f"memetic_step_{memetic_step:02d}_"
                                    f"proposal_{proposal_number:02d}"
                                ),
                                style_name=island.style.name,
                                seed_kind=island.style.seed_kind,
                                emphasis=island.style.emphasis,
                            )
                            step_children.append(candidate)
                            progress.update()
                        valid_step = [item for item in step_children if item.valid]
                        if valid_step:
                            winner = min(
                                valid_step,
                                key=lambda item: (
                                    not item.search_feasible,
                                    item.violation,
                                    zonal_score(item.result, island.style.emphasis),
                                ),
                            )
                            center_result = coarse(memetic_center)
                            center_key = (
                                not bool(center_result.get("search_feasible", False)),
                                float(center_result.get("search_constraint_violation", np.inf)),
                                zonal_score(center_result, island.style.emphasis),
                            )
                            winner_key = (
                                not winner.search_feasible,
                                winner.violation,
                                zonal_score(winner.result, island.style.emphasis),
                            )
                            if winner_key < center_key:
                                memetic_center = winner.genome.copy()
                        children.extend(step_children)

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

                    forbidden = [
                        item.genome for item in island.population + children
                    ]
                    if island.style.operator == "bspline":
                        factory = lambda: bspline_genome_operator(
                            genome,
                            island.rng,
                            mutation_scale,
                            mutation_probability,
                        )
                        origin = "bspline_" + origin
                    else:
                        factory = lambda: local_mutation(
                            genome,
                            island.style.emphasis,
                            island.rng,
                            scale=mutation_scale,
                            probability=mutation_probability,
                        )
                    genome = unique_valid_candidate(
                        factory,
                        baseline,
                        forbidden,
                        island.rng,
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