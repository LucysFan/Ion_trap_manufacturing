"""Workflow 09e (full): two-stage RF-only GA with adaptive mesh, islands,
migration, burst escape, multi-minimum penalty, GIF animations, and
interactive 2D/3D pseudopotential reports.

Layout: reports/09e_two_stage_rf_ga/

Stage 1: N independent quick GA launches (each is one island with its own
         style, its own per-generation history, its own top-5 GIF).
Stage 2: one deep GA starting from the diverse elite subset of stage 1
         plus random restarts. Same island infrastructure, plus final GIF.

Adaptive mesh: an x-junction mesh ladder in which the fine region always
extends past `lock_m` (the deformable zone) so small piecewise-linear
contour features are resolved, while the far field uses progressively
larger cells when the budget is tight.

Two required patches in core/geometry/junction_templates.py:
    max_amplitude_m = 30e-6  ->  60e-6
    max_slope       = 0.80   ->  3.00

Optional dependency for interactive 3D: `pip install plotly`.
If plotly is missing, a static PNG 3D surface is produced instead.
"""

from __future__ import annotations
from collections import Counter

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm.auto import tqdm

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config.targets import RF_ANGULAR_FREQUENCY_RAD_S
from core.analysis.barrier import compute_barrier_metrics, pseudopotential_profile_ev
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem
from core.geometry.movable_knot_xjunction import (
    MovableKnotContour, build_movable_knot_parameters,
    fixed_knots_to_logits, softmax_gap_knots,
)


# =============================================================================
# Extended genome layout
# =============================================================================

N_INTERVALS = 12
N_KNOTS = N_INTERVALS + 1
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

Z_ION_M = 90e-6
MIN_FEATURE_M = 50e-6
MIN_GAP_M = 3.0e-6
LOCK_MIN_M = 300e-6
LOCK_MAX_M = 500e-6

AXIS_CLEARANCE_M = 10e-6

FIXED_CORE_KNOTS_M = np.array(
    [0, 8, 18, 32, 50, 72, 100, 135, 180, 240, 310, 390, 500],
    dtype=float,
) * 1e-6

LOW = np.array(
    [-4.0] * N_INTERVALS
    + [-60e-6] * (N_KNOTS - 2)
    + [-50e-6] * (N_KNOTS - 2)
    + [LOCK_MIN_M, -50e-6, -8e-6, 8e-6, 60e-6, 0.70, -20e-6, 10e-6, 8e-6],
    dtype=float,
)
HIGH = np.array(
    [4.0] * N_INTERVALS
    + [60e-6] * (N_KNOTS - 2)
    + [50e-6] * (N_KNOTS - 2)
    + [LOCK_MAX_M, -3e-6, 65e-6, 55e-6, 310e-6, 5.25, 20e-6, 140e-6, 80e-6],
    dtype=float,
)


# =============================================================================
# Settings
# =============================================================================

@dataclass(frozen=True)
class Settings:
    backend: str

    stage1_launches: int
    stage1_population: int
    stage1_generations: int
    stage1_offspring: int
    stage1_top_per_launch: int
    stage1_migration_interval: int
    stage1_migrants: int
    stage1_stall_generations: int
    stage1_burst_generations: int
    stage1_burst_multiplier: float

    stage2_population: int
    stage2_generations: int
    stage2_offspring: int
    stage2_random_fraction: float
    stage2_migration_interval: int
    stage2_migrants: int
    stage2_stall_generations: int
    stage2_burst_generations: int
    stage2_burst_multiplier: float

    coarse_points: int
    fine_points: int
    fine_top: int
    max_panels: int
    fast_coarse: bool

    seed: int
    diversity_threshold: float

    animation_stride: int
    surface_points_2d: int
    surface_points_3d: int

    output_dir: Path

    rf_peak_v: float = 100.0
    target_z_m: float = Z_ION_M
    min_feature_m: float = MIN_FEATURE_M
    dz_limit_m: float = 3e-6
    barrier_limit_ev: float = 1e-3
    lateral_limit_m: float = 5e-6


@dataclass(frozen=True)
class IslandStyle:
    name: str
    weights: tuple[float, float]
    explore: float
    final: float
    emphasis: str


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    generation: int
    origin: str
    launch_id: int = 0
    island_id: int = 0
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result.get("valid", False))

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf])
        return np.array([
            self.result["dz_peak_m"] / 3e-6,
            self.result["barrier_ev"] / 1e-3,
        ])

    def copy(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(), result=dict(self.result),
            generation=self.generation, origin=self.origin,
            launch_id=self.launch_id, island_id=self.island_id,
            rank=self.rank, crowding=self.crowding,
        )


class BEMField:
    def __init__(self, bem: FixedMeshBEM, charge: Any) -> None:
        self.bem, self.charge = bem, charge

    def electric_field(self, x_m, y_m, z_m):
        v = self.bem.field_batch(
            np.array([x_m]), np.array([y_m]), np.array([z_m]), self.charge,
        )
        f = self.bem.asnumpy(v)[0, 0]
        return float(f[0]), float(f[1]), float(f[2])


# =============================================================================
# Genome operations
# =============================================================================

def repair(x):
    x = np.asarray(x, dtype=float).copy()
    if x.shape != (N_GENES,):
        raise ValueError(f"Expected shape {(N_GENES,)}, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite genome")
    return np.clip(x, LOW, HIGH)


def decode(x):
    x = repair(x)
    lock_m = float(x[LOCK])
    knots = softmax_gap_knots(x[GAPS], lock_m=lock_m, min_gap_m=MIN_GAP_M)
    return knots, np.r_[0.0, x[INNER], 0.0], np.r_[0.0, x[OUTER], 0.0]

def baseline():
    x = np.zeros(N_GENES, dtype=float)
    x[LOCK] = FIXED_CORE_KNOTS_M[-1]
    x[GAPS] = fixed_knots_to_logits(FIXED_CORE_KNOTS_M, min_gap_m=MIN_GAP_M)
    x[CENTER_IN:] = [-25e-6, 20e-6, 30e-6, 150e-6, 2.0, 0.0, 70e-6, 25e-6]
    return repair(x)


def parameters_from_genome(x):
    x = repair(x)
    knots, inner, outer = decode(x)
    contour = MovableKnotContour(
        knots_m=knots, inner_offsets_m=inner, outer_offsets_m=outer,
        lock_m=float(x[LOCK]), min_gap_m=MIN_GAP_M,
    )
    return build_movable_knot_parameters(
        contour=contour,
        inner_edge_shift_at_centre_m=float(x[CENTER_IN]),
        outer_edge_shift_at_centre_m=float(x[CENTER_OUT]),
        rf_start_radius_override_m=float(x[START]),
        taper_length_m=float(x[LENGTH]),
        taper_power=float(x[POWER]),
        outer_bulge_amplitude_m=float(x[BULGE]),
        outer_bulge_center_m=float(x[BULGE_CENTER]),
        outer_bulge_sigma_m=float(x[BULGE_WIDTH]),
        ion_height_m=Z_ION_M,
    )


# =============================================================================
# Adaptive mesh ladder
# =============================================================================

def build_mesh_adaptive(parameters, lock_m, cfg, *, stage):
    """Try finer-to-coarser mesh levels; return first that fits the budget.

    The fine-cell region always extends past `lock_m` so all movable-knot
    contour features are resolved. Only cell sizes in the far field are
    relaxed at lower levels. If no level fits, returns (None, last_level).
    """
    central_half = max(float(lock_m) + 30e-6, 200e-6)

    if stage == "fine":
        ladder = [
            dict(central_half_extent_m=central_half, central_max_cell_m=20e-6,
                 boundary_max_cell_m=8e-6, outer_max_cell_m=260e-6, min_cell_m=4e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=28e-6,
                 boundary_max_cell_m=10e-6, outer_max_cell_m=260e-6, min_cell_m=5e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=42e-6,
                 boundary_max_cell_m=16e-6, outer_max_cell_m=300e-6, min_cell_m=8e-6),
        ]
    elif cfg.fast_coarse:
        ladder = [
            dict(central_half_extent_m=central_half, central_max_cell_m=42e-6,
                 boundary_max_cell_m=16e-6, outer_max_cell_m=240e-6, min_cell_m=8e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=55e-6,
                 boundary_max_cell_m=22e-6, outer_max_cell_m=280e-6, min_cell_m=10e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=75e-6,
                 boundary_max_cell_m=30e-6, outer_max_cell_m=320e-6, min_cell_m=13e-6),
        ]
    else:
        ladder = [
            dict(central_half_extent_m=central_half, central_max_cell_m=24e-6,
                 boundary_max_cell_m=9e-6, outer_max_cell_m=260e-6, min_cell_m=4e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=32e-6,
                 boundary_max_cell_m=12e-6, outer_max_cell_m=280e-6, min_cell_m=6e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=45e-6,
                 boundary_max_cell_m=18e-6, outer_max_cell_m=300e-6, min_cell_m=8e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=60e-6,
                 boundary_max_cell_m=25e-6, outer_max_cell_m=340e-6, min_cell_m=10e-6),
        ]

    last_panels = 0
    last_level = -1
    for level, params in enumerate(ladder):
        model = build_geometry_aware_quadtree_x_junction_bem(parameters, **params)
        last_panels = int(model.n_panels)
        last_level = level
        if model.n_panels <= cfg.max_panels:
            return model, level
    return None, last_level


# =============================================================================
# Evaluation
# =============================================================================

def rejected(reason, stage):
    return dict(
        valid=False, verified=False, reason=reason, stage=stage,
        dz_peak_m=np.inf, dz_rms_m=np.inf, barrier_ev=np.inf,
        lateral_peak_m=np.inf, barrier_reference_ev=np.nan,
        panels=0, points=0, converged=0, n_minima=0, mesh_level=-1,
    )

def _count_effective_minima(u_ps, prominence_frac=0.10):
    u = np.asarray(u_ps, dtype=float)
    span = float(u.max() - u.min())
    if span <= 0.0:
        return 0
    prom = prominence_frac * span
    n = len(u); cnt = 0
    for i in range(1, n - 1):
        if u[i] <= u[i - 1] and u[i] <= u[i + 1]:
            w = max(5, n // 20)
            lo, hi = max(0, i - w), min(n, i + w + 1)
            if (u[lo:hi].max() - u[i]) >= prom:
                cnt += 1
    return cnt

def evaluate(x, cfg, *, stage="coarse", retain=False):
    x = repair(x)
    try:
        p = parameters_from_genome(x)
        report = check_x_junction_manufacturability(
            p, min_feature_size_m=cfg.min_feature_m,
        )
        if not report.valid:
            return rejected("geometry: " + "; ".join(map(str, report.messages)), stage), None

        s = np.linspace(0.0, p.arm_length_m, 1001)
        rin, rout = p.rail_boundaries_m(s)

        if np.min(rin) <= AXIS_CLEARANCE_M:
            return rejected(
                f"inner clearance (< {AXIS_CLEARANCE_M * 1e6:.1f} um)",
                stage,
            ), None

        if np.min(rout - rin) < cfg.min_feature_m:
            return rejected("rail width", stage), None

        # Это именно литографическое ограничение: ширина RF-рельса.
        if np.min(rout - rin) < cfg.min_feature_m:
            return rejected("rail width", stage), None
        model, level = build_mesh_adaptive(p, float(x[LOCK]), cfg, stage=stage)
        if model is None:
            return rejected(f"panel limit at level {level}", stage), None

        bem = FixedMeshBEM(model.bem.panels_m, backend=cfg.backend)
        if cfg.backend == "cuda" and bem.backend_name != "cuda":
            return rejected(f"cuda->{bem.backend_name}", stage), None
        bem.assemble(block_rows=64)
        if cfg.backend == "cuda" and bem.backend_name != "cuda":
            return rejected("cuda downgraded", stage), None
        bem.factorize()
        bem.release_matrix()
        charge = bem.solve_masks(model.bem.electrode_voltages_v)
        field = BEMField(bem, charge)

        n = cfg.fine_points if stage == "fine" else cfg.coarse_points
        trace = trace_rf_transverse_minimum(
            field, np.linspace(-350e-6, 350e-6, n),
            initial_y_m=0.0, initial_z_m=Z_ION_M,
            residual_tolerance_v_m=1e-3, max_transverse_shift_m=25e-6,
        )
        conv = np.asarray(trace.converged, dtype=bool)
        z_m = np.asarray(trace.z_m, dtype=float)
        y_m = np.asarray(trace.y_m, dtype=float)
        if not trace.valid or conv.shape != (n,) or not np.all(conv):
            return rejected("incomplete trace", stage), None
        if not np.all(np.isfinite(z_m)) or not np.all(np.isfinite(y_m)):
            return rejected("nonfinite trace", stage), None

        pseudo_ev = np.asarray(pseudopotential_profile_ev(
            field, trace, rf_voltage_peak_v=cfg.rf_peak_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        ), dtype=float)
        bar = compute_barrier_metrics(pseudo_ev, trace)
        if not bar.valid or pseudo_ev.shape != (n,) or not np.all(np.isfinite(pseudo_ev)):
            return rejected("invalid pseudopotential", stage), None

        n_min = _count_effective_minima(pseudo_ev, prominence_frac=0.10)
        if n_min > 1:
            return rejected(f"multi-minimum ({n_min})", stage), None

        dz = z_m - Z_ION_M
        peak = float(np.max(np.abs(dz)))
        rms = float(np.sqrt(np.mean(dz ** 2)))
        lat = float(np.max(np.abs(y_m)))
        U = float(bar.barrier_height_ev)
        if not np.all(np.isfinite([peak, rms, lat, U])):
            return rejected("nonfinite metrics", stage), None

        result = dict(
            valid=True,
            verified=bool(stage == "fine" and peak <= cfg.dz_limit_m
                          and U <= cfg.barrier_limit_ev
                          and lat <= cfg.lateral_limit_m),
            reason="ok", stage=stage, backend=bem.backend_name,
            dz_peak_m=peak, dz_rms_m=rms, barrier_ev=U, lateral_peak_m=lat,
            barrier_reference_ev=float(bar.reference_energy_ev),
            panels=int(model.n_panels), points=n,
            converged=int(np.sum(conv)), n_minima=int(n_min),
            mesh_level=int(level),
        )
        data = (model, trace, pseudo_ev, field) if retain else None
        return result, data
    except Exception as err:
        if cfg.backend == "cuda" and "out of memory" in str(err).lower():
            raise RuntimeError("CUDA OOM: reduce --max-panels") from err
        return rejected(
            f"error {type(err).__name__}: "
            f"{str(err).replace(chr(10), ' ').strip()}", stage,
        ), None


# =============================================================================
# NSGA-II
# =============================================================================

def dominates(a, b):
    if a.valid != b.valid:
        return a.valid
    return bool(a.valid and np.all(a.objectives <= b.objectives)
                and np.any(a.objectives < b.objectives))

def rank_and_crowding(pop):
    n = len(pop)
    defeated = [[] for _ in pop]
    loss = np.zeros(n, int)
    fronts = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(pop[i], pop[j]):
                defeated[i].append(j); loss[j] += 1
            elif dominates(pop[j], pop[i]):
                defeated[j].append(i); loss[i] += 1
    fronts[0] = [i for i in range(n) if loss[i] == 0]
    while fronts[-1]:
        nxt = []
        for i in fronts[-1]:
            for j in defeated[i]:
                loss[j] -= 1
                if loss[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
    fronts.pop()
    for lvl, fr in enumerate(fronts):
        for i in fr:
            pop[i].rank = lvl; pop[i].crowding = 0.0
        valid = [i for i in fr if pop[i].valid]
        if len(valid) <= 2:
            for i in valid:
                pop[i].crowding = np.inf
            continue
        for ax in range(2):
            order = sorted(valid, key=lambda i: pop[i].objectives[ax])
            lo, hi = pop[order[0]].objectives[ax], pop[order[-1]].objectives[ax]
            if hi <= lo:
                continue
            pop[order[0]].crowding = pop[order[-1]].crowding = np.inf
            for k in range(1, len(order) - 1):
                if np.isfinite(pop[order[k]].crowding):
                    pop[order[k]].crowding += (
                        pop[order[k+1]].objectives[ax]
                        - pop[order[k-1]].objectives[ax]
                    ) / (hi - lo)

def survivors(pool, size):
    rank_and_crowding(pool)
    pool.sort(key=lambda i: (i.rank, -i.crowding))
    return [i.copy() for i in pool[:size]]


# =============================================================================
# Operators and island styles
# =============================================================================

def anneal(style, frac):
    if frac <= 0.75:
        return style.explore
    q = (frac - 0.75) / 0.25
    return style.final + 0.5 * (style.explore - style.final) * (1.0 + math.cos(math.pi * q))

def sigmas(style, frac):
    s = anneal(style, frac)
    v = np.zeros(N_GENES)
    v[GAPS] = 0.34 * s
    v[INNER] = 10e-6 * s; v[OUTER] = 10e-6 * s
    v[LOCK] = 24e-6 * s
    v[CENTER_IN:CENTER_OUT + 1] = 9e-6 * s
    v[START] = 10e-6 * s; v[LENGTH] = 24e-6 * s; v[POWER] = 0.35 * s
    v[BULGE] = 6e-6 * s; v[BULGE_CENTER] = 18e-6 * s; v[BULGE_WIDTH] = 12e-6 * s

    if style.emphasis == "inner":
        v[INNER] *= 1.8
    elif style.emphasis == "outer":
        v[OUTER] *= 1.8
    elif style.emphasis == "x":
        v[GAPS] *= 2.3; v[LOCK] *= 1.7
    elif style.emphasis == "inner_x":
        v[INNER] *= 1.7; v[GAPS] *= 1.5
    elif style.emphasis == "outer_x":
        v[OUTER] *= 1.7; v[GAPS] *= 1.5
    elif style.emphasis == "central":
        v[CENTER_IN:START + 1] *= 2.0
    elif style.emphasis == "wide":
        v *= 1.35
    elif style.emphasis == "local":
        v *= 0.55
    elif style.emphasis == "feature":
        v[INNER] *= 2.4; v[OUTER] *= 2.4; v[GAPS] *= 1.5; v[LOCK] *= 1.4
    elif style.emphasis == "feature_inner":
        v[INNER] *= 3.0; v[OUTER] *= 1.4; v[GAPS] *= 1.7
    elif style.emphasis == "feature_outer":
        v[INNER] *= 1.4; v[OUTER] *= 3.0; v[GAPS] *= 1.7
    return v

def mutate(parent, style, rng, frac, *, burst=False, burst_mul=1.0):
    x = repair(parent)
    sig = sigmas(style, frac) * (burst_mul if burst else 1.0)
    chance = np.full(N_GENES, 0.55)

    if style.emphasis in ("x", "inner_x", "outer_x", "feature", "feature_inner", "feature_outer"):
        chance[GAPS] = 0.88
    if style.emphasis in ("inner", "inner_x", "feature_inner"):
        chance[INNER] = 0.88
    if style.emphasis in ("outer", "outer_x", "feature_outer"):
        chance[OUTER] = 0.88
    if burst:
        chance = np.maximum(chance, 0.82)
        chance[GAPS] = 0.92; chance[INNER] = 0.88; chance[OUTER] = 0.88
        chance[LOCK] = 0.92; chance[CENTER_IN:CENTER_OUT + 1] = 0.90
        chance[START:LENGTH + 1] = 0.85

    active = rng.random(N_GENES) < chance
    x[active] += rng.normal(0.0, sig[active])
    return repair(x)

def block_crossover(a, b, rng):
    out = np.empty(N_GENES)
    for blk in (GAPS, INNER, OUTER, slice(LOCK, N_GENES)):
        alpha = float(rng.uniform(-0.10, 1.10))
        out[blk] = alpha * a[blk] + (1.0 - alpha) * b[blk]
    return repair(out)

def style_score(i, style):
    if not i.valid:
        return np.inf
    if style.emphasis == "joint":
        return max(i.result["dz_peak_m"]/10e-6, i.result["barrier_ev"]/10e-3)
    return float(np.dot(np.asarray(style.weights), i.objectives))

def tournament(pop, style, rng):
    a, b = (pop[int(k)] for k in rng.integers(0, len(pop), 2))
    if a.valid != b.valid:
        return a if a.valid else b
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.crowding != b.crowding:
        return a if a.crowding > b.crowding else b
    return a if style_score(a, style) <= style_score(b, style) else b


ORDINARY_STYLES = (
    IslandStyle("height", (7, 1), 1.00, 0.10, "inner"),
    IslandStyle("barrier", (1, 7), 1.00, 0.10, "outer"),
    IslandStyle("balanced", (2, 2), 0.90, 0.09, "both"),
    IslandStyle("movable_x", (2, 2), 1.05, 0.10, "x"),
    IslandStyle("central", (3, 2), 1.05, 0.10, "central"),
    IslandStyle("inner_x", (4, 2), 1.10, 0.11, "inner_x"),
    IslandStyle("outer_x", (2, 4), 1.10, 0.11, "outer_x"),
    IslandStyle("wide", (2, 2), 1.25, 0.14, "wide"),
    IslandStyle("joint", (2, 2), 0.95, 0.08, "joint"),
    IslandStyle("local", (2, 2), 0.42, 0.03, "local"),
    IslandStyle("feature", (2, 2), 1.05, 0.10, "feature"),
    IslandStyle("feature_inner", (4, 1), 1.10, 0.11, "feature_inner"),
    IslandStyle("feature_outer", (1, 4), 1.10, 0.11, "feature_outer"),
)


# =============================================================================
# Seeds and diversity
# =============================================================================

def seed_list():
    base = baseline()
    i1 = base.copy(); i1[INNER] = np.array([-5, -10, -14, -16, -14, -10, -6, -3, -2, -1, -1], dtype=float)*1e-6
    o1 = base.copy(); o1[OUTER] = np.array([2, 4, 6, 8, 9, 8, 6, 4, 3, 2, 1], dtype=float)*1e-6
    c1 = base.copy()
    c1[INNER] = np.array([-4, -8, -12, -15, -14, -11, -7, -4, -2, -1, -1], dtype=float)*1e-6
    c1[OUTER] = np.array([2, 5, 8, 10, 10, 8, 6, 4, 3, 1, 1], dtype=float)*1e-6
    x1 = base.copy(); x1[GAPS] = np.linspace(1.5, -1.5, N_INTERVALS); x1[LOCK] = 380e-6
    return [
        ("baseline", base),
        ("inner", i1),
        ("outer", o1),
        ("coupled", c1),
        ("packed", x1),
    ]

def diversity_filter(cands, threshold):
    cands = sorted(cands, key=lambda i: max(i.objectives))
    kept = []
    span = np.where((HIGH - LOW) > 0, HIGH - LOW, 1.0)
    for c in cands:
        if all(
            float(np.sqrt(np.sum(((c.genome - k.genome) / span) ** 2))) >= threshold
            for k in kept
        ):
            kept.append(c)
    return kept


# =============================================================================
# Serialization
# =============================================================================

def serialize(i):
    knots, inner, outer = decode(i.genome)
    return dict(
        launch=i.launch_id, island=i.island_id,
        generation=i.generation, origin=i.origin,
        rank=i.rank, crowding=i.crowding,
        genome=i.genome.tolist(),
        movable_knots_m=knots.tolist(),
        inner_offsets_m=inner.tolist(),
        outer_offsets_m=outer.tolist(),
        lock_m=float(i.genome[LOCK]),
        **i.result,
    )

def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            })


# =============================================================================
# Geometry drawing
# =============================================================================

def draw_geometry(ax, row, label):
    raw = row["genome"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    genome = repair(np.asarray(raw, dtype=float))
    p = parameters_from_genome(genome)

    s = np.linspace(0.0, 260e-6, 700)
    rin, rout = p.rail_boundaries_m(s)
    for theta in (0.0, np.pi/2, np.pi, 3*np.pi/2):
        c, sn = np.cos(theta), np.sin(theta)
        x1, y1 = s*c - rin*sn, s*sn + rin*c
        x2, y2 = s*c - rout*sn, s*sn + rout*c
        ax.fill(np.r_[x1, x2[::-1]]*1e6, np.r_[y1, y2[::-1]]*1e6,
                color="#db4f4f", ec="black", lw=0.35, alpha=0.90)

    knots, _, _ = decode(genome)
    ki, ko = p.rail_boundaries_m(knots)
    ax.plot(knots*1e6, ki*1e6, "ko", ms=2.5)
    ax.plot(knots*1e6, ko*1e6, "wo", mec="black", ms=2.5)
    ax.axhline(0, color="0.7", lw=0.35); ax.axvline(0, color="0.7", lw=0.35)
    ax.set(xlim=(-260, 260), ylim=(-260, 260), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    dz = float(row.get("dz_peak_m", np.inf)) * 1e6
    U = float(row.get("barrier_ev", np.inf)) * 1e3
    lock = float(row.get("lock_m", 0.0)) * 1e6
    ax.set_title(f"{label}\ndz={dz:.2f}um U={U:.2f}meV\nlock={lock:.0f}um",
                 fontsize=7)


# =============================================================================
# Surfaces
# =============================================================================

def pseudopotential_surface_ev(field, *, z_m, extent_m, points, rf_peak_v):
    coords = np.linspace(-extent_m, extent_m, points)
    xg, yg = np.meshgrid(coords, coords, indexing="xy")
    xf = xg.reshape(-1); yf = yg.reshape(-1)
    zf = np.full_like(xf, z_m)
    v = field.bem.field_batch(xf[None, :], yf[None, :], zf[None, :], field.charge)
    E = field.bem.asnumpy(v)[0]
    E2 = np.sum(E**2, axis=1)
    q = 1.602176634e-19
    u = 1.66053906660e-27
    m_ca40 = 39.96259098 * u
    pseudo = q * rf_peak_v**2 * E2 / (4.0 * m_ca40 * RF_ANGULAR_FREQUENCY_RAD_S**2)
    return xg, yg, pseudo.reshape(xg.shape)

def plot_surface_2d(xg, yg, surf_mev, trace, path, title, cfg):
    finite = surf_mev[np.isfinite(surf_mev)]
    upper = float(np.percentile(finite, 97.0)) if finite.size else 1.0
    upper = max(upper, 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(xg*1e6, yg*1e6, np.clip(surf_mev, 0, upper),
                     levels=60, cmap="magma")
    ax.contour(xg*1e6, yg*1e6, np.clip(surf_mev, 0, upper),
               levels=12, colors="white", lw=0.35, alpha=0.45)
    ax.plot(np.asarray(trace.x_m)*1e6, np.asarray(trace.y_m)*1e6,
            color="cyan", lw=1.2, label="RF-null trace")
    ax.set(xlim=(-230, 230), ylim=(-230, 230), aspect="equal",
           xlabel="x [um]", ylabel="y [um]",
           title=f"{title}\nU_ps(x,y,z={cfg.target_z_m*1e6:.0f}um)")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="RF pseudopotential [meV]")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def plot_surface_3d_plotly(xg, yg, surf_mev, trace, html_path):
    if not _HAS_PLOTLY:
        return False
    fig = go.Figure(data=[go.Surface(
        x=xg*1e6, y=yg*1e6, z=surf_mev, colorscale="Magma",
        colorbar=dict(title="meV"),
        hovertemplate="x=%{x:.0f}um<br>y=%{y:.0f}um<br>U=%{z:.3f}meV",
    )])
    fig.update_layout(
        title="RF pseudopotential (interactive)",
        scene=dict(xaxis_title="x [um]", yaxis_title="y [um]",
                   zaxis_title="U_ps [meV]",
                   aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=0.5)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return True

def plot_surface_3d_static(xg, yg, surf_mev, png_path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xg*1e6, yg*1e6, surf_mev, cmap="magma",
                           linewidth=0, antialiased=True,
                           rstride=4, cstride=4, alpha=0.95)
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    ax.set_zlabel("U_ps [meV]")
    ax.set_title("RF pseudopotential (static)")
    ax.view_init(elev=35, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="meV")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_layout(model, trace, path, title):
    panels = np.asarray(model.bem.panels_m)
    rf = np.asarray(model.bem.electrode_voltages_v) > 0.5
    rects = [Rectangle((p[0]*1e6, p[2]*1e6), (p[1]-p[0])*1e6,
                       (p[3]-p[2])*1e6) for p in panels]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    coll = PatchCollection(rects, array=rf.astype(float), cmap="RdYlBu_r",
                           edgecolor="none")
    ax.add_collection(coll)
    ax.plot(np.asarray(trace.x_m)*1e6, np.asarray(trace.y_m)*1e6,
            "k-", lw=1.2, label="RF-null")
    ax.set(xlim=(-230, 230), ylim=(-230, 230), aspect="equal",
           xlabel="x [um]", ylabel="y [um]", title=title)
    ax.legend(loc="upper right")
    fig.colorbar(coll, ax=ax, label="RF mask")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def plot_profiles(trace, pseudo_ev, result, path, title, cfg):
    xx = np.asarray(trace.x_m)*1e6
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    z = np.asarray(trace.z_m) - Z_ION_M
    axs[0].plot(xx, z*1e6, color="tab:blue")
    axs[0].axhline(3, color="red", ls="--"); axs[0].axhline(-3, color="red", ls="--")
    axs[0].set_ylabel("z-target [um]"); axs[0].grid(alpha=0.25)
    ref = result.get("barrier_reference_ev", np.nan)
    axs[1].plot(xx, (pseudo_ev - ref)*1e3, color="tab:purple")
    axs[1].axhline(1, color="red", ls="--")
    axs[1].set_ylabel("U_ps - ref [meV]"); axs[1].grid(alpha=0.25)
    axs[2].plot(xx, np.asarray(trace.y_m)*1e6, color="tab:green")
    axs[2].set(xlabel="x [um]", ylabel="y [um]"); axs[2].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# GIF animation
# =============================================================================

def animate_top5(history, path, *, title, stride=1):
    if not history:
        return
    frames = []
    for gen, pop in enumerate(history):
        if gen % stride != 0:
            continue
        valid = [i for i in pop if i.valid]
        valid.sort(key=lambda i: (i.rank, -i.crowding, i.objectives.sum()))
        frames.append((gen, [serialize(i) for i in valid[:5]]))
    if not frames:
        return

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.8), constrained_layout=True)

    def update(k):
        gen, rows = frames[k]
        for ax in axes:
            ax.clear()
        for pos, ax in enumerate(axes):
            if pos < len(rows):
                try:
                    draw_geometry(ax, rows[pos], f"Top {pos+1}")
                except Exception as err:
                    ax.text(0.5, 0.5, f"err: {err}",
                            transform=ax.transAxes, ha="center", va="center")
                    ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, "no valid elite",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
        fig.suptitle(f"{title} | generation {gen}", fontsize=12)
        return axes

    try:
        anim = FuncAnimation(fig, update, frames=len(frames),
                             interval=600, blit=False, repeat=True)
        anim.save(path, writer=PillowWriter(fps=2), dpi=100)
    except Exception as err:
        print(f"  GIF skipped: {type(err).__name__}: {err}")
    plt.close(fig)


# =============================================================================
# Island GA (one launch with multiple islands)
# =============================================================================

def run_island_ga(
    cfg: Settings, *,
    n_islands: int,
    population: int,
    generations: int,
    offspring: int,
    migration_interval: int,
    migrants: int,
    stall_generations: int,
    burst_generations: int,
    burst_multiplier: float,
    seed: int,
    initial: list[np.ndarray] | None = None,
    desc: str = "GA",
    launch_id: int = 0,
    track_history: bool = True,
) -> tuple[list[list[Individual]], list[list[Individual]]]:
    """Run one multi-island GA. Returns (final_islands, per_gen_populations)."""
    master = np.random.default_rng(seed)
    evaluated: dict[tuple[float, ...], dict[str, Any]] = {}
    styles = list(ORDINARY_STYLES[:n_islands])
    islands: list[list[Individual]] = []
    rngs = [np.random.default_rng(int(master.integers(0, 2**32 - 1)))
            for _ in range(n_islands)]

    seeds = seed_list()

    def ev(x, bar=None):
        key = tuple(np.round(repair(x), 12))
        if key not in evaluated:
            evaluated[key], _ = evaluate(x, cfg, stage="coarse")
        if bar is not None:
            valid = sum(1 for r in evaluated.values() if r["valid"])
            bar.set_postfix(valid=valid, uniq=len(evaluated), refresh=False)
        return dict(evaluated[key])

    total_init = n_islands * population
    with tqdm(total=total_init, desc=f"{desc} init", unit="cand") as bar:
        for isl_id, style in enumerate(styles):
            rng = rngs[isl_id]
            pop: list[Individual] = []
            if initial and isl_id < len(initial):
                x0 = initial[isl_id]
                pop.append(Individual(repair(x0), ev(x0, bar), 0,
                                      "stage_seed", launch_id, isl_id))
                bar.update()
            # fill with rotated seeds + noise
            while len(pop) < population:
                idx = (isl_id + len(pop)) % len(seeds)
                name, sx = seeds[idx]
                x = repair(sx + rng.normal(0, (HIGH - LOW) * 0.03 * style.explore))
                pop.append(Individual(x, ev(x, bar), 0,
                                      f"seed_{name}", launch_id, isl_id))
                bar.update()
            rank_and_crowding(pop)
            islands.append(pop)

    initial_population = [ind for island in islands for ind in island]

    reasons = Counter(
        ind.result.get("reason", "unknown")
        for ind in initial_population
        if not ind.valid
    )

    print(
        "Initial rejection reasons: "
        f"{dict(reasons)}"
    )

    history: list[list[Individual]] = []
    if track_history:
        history.append([ind.copy() for isl in islands for ind in isl])

    best_joint_seen = np.inf
    stall_count = 0
    burst_remaining = 0

    def joint(i):
        if not i.valid:
            return np.inf
        return max(i.result["dz_peak_m"]/10e-6, i.result["barrier_ev"]/10e-3)

    def best_joint():
        cands = [i for isl in islands for i in isl if i.valid]
        return min((joint(i) for i in cands), default=np.inf)

    for gen in range(1, generations + 1):
        burst_active = burst_remaining > 0
        mode = "BURST" if burst_active else "NORMAL"

        with tqdm(total=n_islands * offspring,
                  desc=f"{desc} g{gen:03d} [{mode}]",
                  unit="cand") as bar:
            next_islands: list[list[Individual]] = []
            for isl_id, (pop, style, rng) in enumerate(zip(islands, styles, rngs)):
                rank_and_crowding(pop)
                elite = sorted(pop, key=lambda i: (i.rank, -i.crowding,
                                                   style_score(i, style)))[:5]
                children: list[Individual] = []
                while len(children) < offspring:
                    u = rng.random()
                    if not burst_active:
                        if u < 0.60:
                            parent = elite[0] if rng.random() < 0.8 else elite[min(1, len(elite)-1)]
                            child = parent.genome.copy()
                            origin = "top_mut"
                        elif u < 0.88:
                            l = tournament(pop, style, rng); r = tournament(pop, style, rng)
                            child = block_crossover(l.genome, r.genome, rng)
                            origin = "pareto_x"
                        else:
                            l = elite[int(rng.integers(len(elite)))]
                            r = pop[int(rng.integers(len(pop)))]
                            child = block_crossover(l.genome, r.genome, rng)
                            origin = "elite_rand_x"
                    else:
                        if u < 0.25:
                            parent = elite[int(rng.integers(min(2, len(elite))))]
                            child = parent.genome.copy()
                            origin = "burst_elite"
                        elif u < 0.60:
                            l = tournament(pop, style, rng); r = tournament(pop, style, rng)
                            child = block_crossover(l.genome, r.genome, rng)
                            origin = "burst_pareto_x"
                        elif u < 0.80:
                            l = elite[int(rng.integers(len(elite)))]
                            r = pop[int(rng.integers(len(pop)))]
                            child = block_crossover(l.genome, r.genome, rng)
                            origin = "burst_elite_rand"
                        else:
                            # cross-island mixing
                            donor_island = islands[int(rng.integers(len(islands)))]
                            donor = donor_island[int(rng.integers(len(donor_island)))]
                            l = elite[int(rng.integers(len(elite)))]
                            child = block_crossover(l.genome, donor.genome, rng)
                            origin = "burst_cross_isl"

                    child = mutate(child, style, rng, gen/max(1, generations),
                                   burst=burst_active,
                                   burst_mul=burst_multiplier)
                    children.append(Individual(child, ev(child, bar), gen,
                                               origin, launch_id, isl_id))
                    bar.update()
                next_islands.append(survivors(pop + children, population))
            islands = next_islands

        # migration ring
        migration_now = (
            len(islands) > 1
            and (
                burst_active
                or (migration_interval > 0 and gen % migration_interval == 0)
            )
        )

        if migration_now:
            migrant_count = (
                min(12, population)
                if burst_active
                else min(migrants, population)
            )

            outgoing: list[list[Individual]] = []
            for pop in islands:
                rank_and_crowding(pop)
                outgoing.append([
                    i.copy()
                    for i in sorted(
                        pop,
                        key=lambda x: (
                            x.rank,
                            -x.crowding,
                            style_score(x, styles[0]),
                        ),
                    )[:migrant_count]
                ])

            for src, incoming_migrants in enumerate(outgoing):
                dst = (src + 1) % len(islands)
                islands[dst] = survivors(
                    islands[dst] + incoming_migrants,
                    population,
                )

        # stagnation / burst
        now = best_joint()
        improved = now < best_joint_seen * (1.0 - 0.01)
        if improved:
            best_joint_seen = now
            stall_count = 0
            if burst_active:
                burst_remaining = 0
        else:
            stall_count += 1

        if burst_active and burst_remaining > 0:
            burst_remaining -= 1

        if (not burst_active and burst_remaining == 0
                and stall_count >= stall_generations):
            burst_remaining = burst_generations
            stall_count = 0
            print(f"  {desc}: burst scheduled at g{gen+1:03d}")

        if track_history:
            history.append([i.copy() for isl in islands for i in isl])

        save_csv([serialize(i) for isl in islands for i in isl],
                 cfg.output_dir / "tmp_last_population.csv")

    return islands, history


# =============================================================================
# CLI
# =============================================================================

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--stage1-launches", type=int, default=15)
    p.add_argument("--stage1-islands", type=int, default=3)
    p.add_argument("--stage1-population", type=int, default=12)
    p.add_argument("--stage1-generations", type=int, default=10)
    p.add_argument("--stage1-offspring", type=int, default=6)
    p.add_argument("--stage1-top-per-launch", type=int, default=6)
    p.add_argument("--stage1-migration-interval", type=int, default=3)
    p.add_argument("--stage1-migrants", type=int, default=3)
    p.add_argument("--stage1-stall-generations", type=int, default=4)
    p.add_argument("--stage1-burst-generations", type=int, default=2)
    p.add_argument("--stage1-burst-multiplier", type=float, default=3.0)

    p.add_argument("--stage2-islands", type=int, default=4)
    p.add_argument("--stage2-population", type=int, default=20)
    p.add_argument("--stage2-generations", type=int, default=60)
    p.add_argument("--stage2-offspring", type=int, default=10)
    p.add_argument("--stage2-random-fraction", type=float, default=0.40)
    p.add_argument("--stage2-migration-interval", type=int, default=3)
    p.add_argument("--stage2-migrants", type=int, default=3)
    p.add_argument("--stage2-stall-generations", type=int, default=5)
    p.add_argument("--stage2-burst-generations", type=int, default=3)
    p.add_argument("--stage2-burst-multiplier", type=float, default=3.0)

    p.add_argument("--coarse-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=181)
    p.add_argument("--fine-top", type=int, default=25)
    p.add_argument("--max-panels", type=int, default=6000)
    p.add_argument("--no-fast-coarse", action="store_true")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--diversity-threshold", type=float, default=0.10)

    p.add_argument("--animation-stride", type=int, default=1)
    p.add_argument("--surface-points-2d", type=int, default=121)
    p.add_argument("--surface-points-3d", type=int, default=81)

    p.add_argument("--output-dir", type=Path,
                   default=Path("reports/09e_two_stage_rf_ga"))
    return p


# =============================================================================
# Main
# =============================================================================

def main():
    args = parser().parse_args()
    if args.smoke:
        args.stage1_launches = 2
        args.stage1_islands = 2
        args.stage1_population = 6
        args.stage1_generations = 3
        args.stage1_offspring = 3
        args.stage1_top_per_launch = 3
        args.stage1_migration_interval = 1
        args.stage1_migrants = 1
        args.stage1_stall_generations = 1
        args.stage1_burst_generations = 1
        args.stage2_islands = 2
        args.stage2_population = 8
        args.stage2_generations = 4
        args.stage2_offspring = 4
        args.stage2_migration_interval = 1
        args.stage2_migrants = 1
        args.stage2_stall_generations = 1
        args.stage2_burst_generations = 1
        args.coarse_points = 31
        args.fine_points = 61
        args.fine_top = 3
        args.max_panels = 4000
        args.surface_points_2d = 61
        args.surface_points_3d = 41

    backend = available_backend(args.backend)
    if not backend.available:
        raise RuntimeError(backend.reason)

    cfg = Settings(
        backend=backend.selected,
        stage1_launches=args.stage1_launches,
        stage1_population=args.stage1_population,
        stage1_generations=args.stage1_generations,
        stage1_offspring=args.stage1_offspring,
        stage1_top_per_launch=args.stage1_top_per_launch,
        stage1_migration_interval=args.stage1_migration_interval,
        stage1_migrants=args.stage1_migrants,
        stage1_stall_generations=args.stage1_stall_generations,
        stage1_burst_generations=args.stage1_burst_generations,
        stage1_burst_multiplier=args.stage1_burst_multiplier,
        stage2_population=args.stage2_population,
        stage2_generations=args.stage2_generations,
        stage2_offspring=args.stage2_offspring,
        stage2_random_fraction=args.stage2_random_fraction,
        stage2_migration_interval=args.stage2_migration_interval,
        stage2_migrants=args.stage2_migrants,
        stage2_stall_generations=args.stage2_stall_generations,
        stage2_burst_generations=args.stage2_burst_generations,
        stage2_burst_multiplier=args.stage2_burst_multiplier,
        coarse_points=args.coarse_points,
        fine_points=args.fine_points,
        fine_top=args.fine_top,
        max_panels=args.max_panels,
        fast_coarse=not args.no_fast_coarse,
        seed=args.seed,
        diversity_threshold=args.diversity_threshold,
        animation_stride=args.animation_stride,
        surface_points_2d=args.surface_points_2d,
        surface_points_3d=args.surface_points_3d,
        output_dir=args.output_dir,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "settings.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8",
    )

    print(f"Backend: {backend.selected} ({backend.device_name})")
    print(f"z_ion = {Z_ION_M*1e6:.0f} um, min_feature = {MIN_FEATURE_M*1e6:.0f} um")
    print(f"Genome: {N_GENES} genes; adaptive mesh: fine region extends past lock_m")
    print(f"Output: {cfg.output_dir}\n")

    # Диагностика базовой геометрии до запуска GA.
    base = baseline()
    params = parameters_from_genome(base)

    s = np.linspace(0.0, params.arm_length_m, 1001)
    rin, rout = params.rail_boundaries_m(s)

    print("=== Baseline rail diagnostic ===")
    print(f"min inner radius: {np.min(rin) * 1e6:.3f} um")
    print(f"min rail width  : {np.min(rout - rin) * 1e6:.3f} um")

    base_result, _ = evaluate(base, cfg, stage="coarse")

    print("\n=== Baseline diagnostic ===")
    print(f"valid  : {base_result['valid']}")
    print(f"reason : {base_result['reason']}")

    print("\n=== Baseline diagnostic ===")
    print(f"valid  : {base_result['valid']}")
    print(f"reason : {base_result['reason']}")

    if base_result["valid"]:
        print(f"dz_peak : {base_result['dz_peak_m'] * 1e6:.3f} um")
        print(f"barrier : {base_result['barrier_ev'] * 1e3:.3f} meV")
        print(f"panels  : {base_result['panels']}")

    # ------------------------- STAGE 1 -------------------------
    print(f"=== Stage 1: {args.stage1_launches} launches × "
          f"{args.stage1_islands} islands × {args.stage1_population} pop ===")
    stage1_dir = cfg.output_dir / "stage1"
    stage1_dir.mkdir(exist_ok=True)
    (stage1_dir / "animations").mkdir(exist_ok=True)

    t0 = time.time()
    stage1_all: list[Individual] = []

    for k in range(args.stage1_launches):
        seed_k = cfg.seed + k * 7919
        islands_k, hist_k = run_island_ga(
            cfg,
            n_islands=args.stage1_islands,
            population=args.stage1_population,
            generations=args.stage1_generations,
            offspring=args.stage1_offspring,
            migration_interval=args.stage1_migration_interval,
            migrants=args.stage1_migrants,
            stall_generations=args.stage1_stall_generations,
            burst_generations=args.stage1_burst_generations,
            burst_multiplier=args.stage1_burst_multiplier,
            seed=seed_k, desc=f"L{k+1:02d}",
            launch_id=k, track_history=True,
        )

        launch_dir = stage1_dir / f"launch_{k:02d}"
        launch_dir.mkdir(exist_ok=True)
        all_inds = [i for isl in islands_k for i in isl]
        save_csv([serialize(i) for i in all_inds],
                 launch_dir / "final_population.csv")
        # collect elites
        valid = [i for i in all_inds if i.valid]
        valid.sort(key=lambda i: (i.rank, -i.crowding, i.objectives.sum()))
        top_k = valid[:args.stage1_top_per_launch]
        for i in top_k:
            i.origin = f"{i.origin}_L{k+1:02d}"
        stage1_all.extend(top_k)

        # history CSV and GIF
        history_rows = []
        for gen, pop in enumerate(hist_k):
            for isl_id in range(args.stage1_islands):
                isl_pop = [i for i in pop if i.island_id == isl_id]
                if not isl_pop:
                    continue
                best = min(isl_pop, key=lambda i: i.objectives.sum())
                history_rows.append(dict(
                    generation=gen, island=isl_id,
                    best_dz_um=best.result.get("dz_peak_m", np.inf)*1e6,
                    best_U_mev=best.result.get("barrier_ev", np.inf)*1e3,
                    best_valid=int(best.valid),
                ))
        save_csv(history_rows, launch_dir / "history.csv")
        animate_top5(
            hist_k,
            stage1_dir / "animations" / f"launch_{k:02d}_top5.gif",
            title=f"Stage 1 Launch {k:02d}",
            stride=args.animation_stride,
        )
        print(f"  L{k+1:02d}: {len(valid)}/{len(all_inds)} valid, "
              f"top-{len(top_k)} collected, "
              f"{len(hist_k)} generations tracked")

    print(f"  stage-1 time: {(time.time()-t0)/60:.1f} min")
    print(f"  Stage-1 raw: {len(stage1_all)} candidates")
    diverse = diversity_filter(stage1_all, args.diversity_threshold)
    print(f"  Stage-1 diverse (threshold {args.diversity_threshold}): {len(diverse)}")

    save_csv([serialize(i) for i in diverse],
             cfg.output_dir / "stage1_diverse.csv")
    (cfg.output_dir / "stage1_diverse.json").write_text(
        json.dumps([serialize(i) for i in diverse], indent=2, default=float),
        encoding="utf-8",
    )

    # ------------------------- STAGE 2 -------------------------
    print(f"\n=== Stage 2: refine ({len(diverse)} seeds + random) ===")
    n_total_seeds = args.stage2_islands * args.stage2_population
    n_seed = int(round(n_total_seeds * (1.0 - args.stage2_random_fraction)))
    rng = np.random.default_rng(cfg.seed + 999)
    seed_genomes: list[np.ndarray] = []
    for i in range(n_seed):
        if i < len(diverse):
            seed_genomes.append(diverse[i].genome.copy())
        else:
            seed_genomes.append(repair(rng.uniform(LOW, HIGH)))
    while len(seed_genomes) < n_total_seeds:
        seed_genomes.append(repair(rng.uniform(LOW, HIGH)))

    stage2_dir = cfg.output_dir / "stage2"
    stage2_dir.mkdir(exist_ok=True)
    (stage2_dir / "animations").mkdir(exist_ok=True)

    t0 = time.time()
    islands_final, hist_final = run_island_ga(
        cfg,
        n_islands=args.stage2_islands,
        population=args.stage2_population,
        generations=args.stage2_generations,
        offspring=args.stage2_offspring,
        migration_interval=args.stage2_migration_interval,
        migrants=args.stage2_migrants,
        stall_generations=args.stage2_stall_generations,
        burst_generations=args.stage2_burst_generations,
        burst_multiplier=args.stage2_burst_multiplier,
        seed=cfg.seed + 424242,
        initial=seed_genomes,
        desc="Stage 2", launch_id=0, track_history=True,
    )
    print(f"  stage-2 time: {(time.time()-t0)/60:.1f} min")

    # Stage-2 history / rejection / populations
    history_rows = []
    rejection_rows = []
    elites_dir = stage2_dir / "island_elites"
    elites_dir.mkdir(exist_ok=True)
    for gen, pop in enumerate(hist_final):
        save_csv([serialize(i) for i in pop],
                 stage2_dir / f"population_gen_{gen:04d}.csv")
        for isl_id in range(args.stage2_islands):
            isl_pop = [i for i in pop if i.island_id == isl_id]
            valid_isl = [i for i in isl_pop if i.valid]
            valid_isl.sort(key=lambda i: (i.rank, -i.crowding))
            save_csv([serialize(i) for i in valid_isl[:5]],
                     elites_dir / f"elites_gen_{gen:04d}_isl_{isl_id:02d}.csv")

        valid = [i for i in pop if i.valid]
        rejected_count = Counter(
            i.result["reason"] for i in pop if not i.valid
        )
        for reason, count in rejected_count.items():
            rejection_rows.append(dict(generation=gen, reason=reason, count=count))

        row = dict(generation=gen, valid=len(valid),
                   invalid=len(pop)-len(valid), islands=args.stage2_islands)
        if valid:
            bz = min(valid, key=lambda i: i.result["dz_peak_m"])
            bu = min(valid, key=lambda i: i.result["barrier_ev"])
            bj = min(valid, key=lambda i: max(i.objectives))
            row.update(
                best_dz_um=bz.result["dz_peak_m"]*1e6,
                U_at_best_dz_mev=bz.result["barrier_ev"]*1e3,
                best_U_mev=bu.result["barrier_ev"]*1e3,
                dz_at_best_U_um=bu.result["dz_peak_m"]*1e6,
                best_joint_dz_um=bj.result["dz_peak_m"]*1e6,
                best_joint_U_mev=bj.result["barrier_ev"]*1e3,
            )
        else:
            row.update(best_dz_um=np.inf, best_U_mev=np.inf)
        history_rows.append(row)

    save_csv(history_rows, stage2_dir / "history.csv")
    save_csv(rejection_rows, stage2_dir / "rejection_summary.csv")
    animate_top5(
        hist_final,
        stage2_dir / "animations" / "stage2_top5.gif",
        title="Stage 2",
        stride=args.animation_stride,
    )
    print()

    # ------------------------- FINE VALIDATION -------------------------
    all_final = [i for isl in islands_final for i in isl]
    valid_final = [i for i in all_final if i.valid]
    valid_final.sort(key=lambda i: (
        i.result["dz_peak_m"]/3e-6 + i.result["barrier_ev"]/1e-3
    ))
    selected = valid_final[:args.fine_top]
    if len(selected) < args.fine_top:
        extra = [i for i in diverse if i not in selected]
        selected.extend(extra[:args.fine_top - len(selected)])

    print(f"=== Fine validation: {len(selected)} candidates ===")
    fine_rows = []
    for num, item in enumerate(tqdm(selected, desc="Fine", unit="cand"), 1):
        result, data = evaluate(item.genome, cfg, stage="fine", retain=True)
        row = dict(candidate=num, **serialize(item),
                   **{f"fine_{k}": v for k, v in result.items()})
        fine_rows.append(row)
        if data is None:
            continue
        model, trace, pseudo_ev, field = data
        cdir = cfg.output_dir / "fine" / f"candidate_{num:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        title = (f"Fine candidate {num} | origin={item.origin} | "
                 f"mesh_level={result.get('mesh_level', -1)}")
        for name, fn in (
            ("layout", lambda: plot_layout(model, trace, cdir/"layout.png", title)),
            ("profiles", lambda: plot_profiles(trace, pseudo_ev, result,
                                                cdir/"profiles.png", title, cfg)),
        ):
            try:
                fn()
            except Exception as err:
                print(f"  cand {num} {name} failed: {err}")
        try:
            xg, yg, s2 = pseudopotential_surface_ev(
                field, z_m=cfg.target_z_m, extent_m=230e-6,
                points=cfg.surface_points_2d, rf_peak_v=cfg.rf_peak_v)
            plot_surface_2d(xg, yg, s2*1e3, trace,
                            cdir/"pseudopotential_2d.png", title, cfg)
        except Exception as err:
            print(f"  cand {num} 2D surface failed: {err}")
        try:
            xg3, yg3, s3 = pseudopotential_surface_ev(
                field, z_m=cfg.target_z_m, extent_m=230e-6,
                points=cfg.surface_points_3d, rf_peak_v=cfg.rf_peak_v)
            plot_surface_3d_plotly(xg3, yg3, s3*1e3, trace,
                                   cdir/"pseudopotential_3d.html")
            plot_surface_3d_static(xg3, yg3, s3*1e3,
                                   cdir/"pseudopotential_3d.png")
        except Exception as err:
            print(f"  cand {num} 3D surface failed: {err}")
        (cdir / "full_candidate.json").write_text(
            json.dumps(row, indent=2, default=float), encoding="utf-8")

    save_csv(fine_rows, cfg.output_dir / "fine_validation.csv")

    fine_valid = [r for r in fine_rows if r.get("fine_valid")]
    fine_valid.sort(key=lambda r: max(
        r["fine_dz_peak_m"]/3e-6, r["fine_barrier_ev"]/1e-3))
    top5 = fine_valid[:5]
    (cfg.output_dir / "top5_candidates.json").write_text(
        json.dumps(top5, indent=2, default=float), encoding="utf-8")
    save_csv(top5, cfg.output_dir / "top5_candidates.csv")
    (cfg.output_dir / "verified_hits.json").write_text(
        json.dumps([r for r in fine_valid if r.get("fine_verified")],
                   indent=2, default=float), encoding="utf-8")

    print(f"\nDone. Output: {cfg.output_dir}")
    if top5:
        for r in top5:
            print(f"  cand {r['candidate']:>3}  "
                  f"U = {r['fine_barrier_ev']*1e3:6.3f} meV  "
                  f"dz = {r['fine_dz_peak_m']*1e6:6.3f} um  "
                  f"origin={r['origin']}")
    else:
        print("  No valid fine candidates this run. All reports saved.")


if __name__ == "__main__":
    main()
