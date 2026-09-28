"""Workflow 14: seeded diverse local-island GA for a planar C4v X-junction.

This workflow starts from three known good ordinary full-genome candidates A, B
and C. It preserves the established barrier metric from core.analysis.barrier
as the primary optimisation metric and adds diagnostics for absolute energy
excursion relative to exactly the same reference energy.

The islands explore local diversity around A/B/C with mutations targeted at:
- centre: inner contour, central shifts, RF start;
- transition: inner/outer contour, start, taper;
- shoulder: outer contour, lock, taper and outer bulge;
- global: small perturbation of the full genome.

All candidates use the original project pipeline:
genome -> XJunctionParameters -> quadtree BEM -> RF-null trace -> barrier.

Run from repository root:
    python workflows/14_seeded_diverse_zonal_ga.py --smoke --fast --backend cpu

First practical run:
    python workflows/14_seeded_diverse_zonal_ga.py --backend auto --population 10 --offspring 3 --generations 20 --coarse-points 61 --fine-points 181 --fine-top 20 --max-panels 6500 --fast --output-dir reports/14_seeded_probe
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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from core.analysis.barrier import compute_barrier_metrics, pseudopotential_profile_ev
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.junction_templates import baseline_x_junction_rf_mask
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem
from core.geometry.movable_knot_xjunction import (
    MovableKnotContour,
    build_movable_knot_parameters,
    softmax_gap_knots,
)

Mode = Literal["centre", "transition", "shoulder", "global", "height", "barrier"]
SeedKind = Literal["A", "B", "C", "AB", "AC", "BC", "ABC"]

N_INTERVALS = 8
N_KNOTS = N_INTERVALS + 1
MIN_GAP_M = 3.0e-6
LOCK_MIN_M = 100e-6
LOCK_MAX_M = 180e-6

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

CONTOUR_BLOCK = slice(0, LOCK + 1)
GLOBAL_BLOCK = slice(LOCK + 1, N_GENES)

LOW = np.array(
    [-4.0] * N_INTERVALS
    + [-28e-6] * (N_KNOTS - 2)
    + [-28e-6] * (N_KNOTS - 2)
    + [LOCK_MIN_M, -50e-6, -8e-6, 8e-6, 60e-6, 0.70, -30e-6, 10e-6, 8e-6],
    dtype=float,
)
HIGH = np.array(
    [4.0] * N_INTERVALS
    + [28e-6] * (N_KNOTS - 2)
    + [28e-6] * (N_KNOTS - 2)
    + [LOCK_MAX_M, -3e-6, 65e-6, 55e-6, 310e-6, 5.25, 30e-6, 140e-6, 80e-6],
    dtype=float,
)

CENTRE_LIMIT_M = 45e-6
TRANSITION_LIMIT_M = 160e-6
ARM_LIMIT_M = 320e-6
REFERENCE_MIN_M = 260e-6
REFERENCE_MAX_M = 320e-6

SEARCH_DZ_M = 3.0e-6
SEARCH_BARRIER_EV = 1.0e-3
FINAL_DZ_M = 1.0e-6
FINAL_BARRIER_EV = 0.10e-3

INITIAL_BARRIER_CEILING_EV = 20e-3
INITIAL_DZ_CEILING_M = 18e-6
MAX_INITIAL_ATTEMPTS = 16


@dataclass(frozen=True)
class Settings:
    backend: str
    population: int
    offspring: int
    generations: int
    migrants: int
    migration_interval: int
    retain_per_island: int
    coarse_points: int
    fine_points: int
    fine_top: int
    max_panels: int
    fast: bool
    seed: int
    output_dir: Path
    rf_peak_v: float = 100.0
    target_z_m: float = TARGET_ION_HEIGHT_M


@dataclass(frozen=True)
class IslandStyle:
    name: str
    seed_kind: SeedKind
    mode: Mode
    scale: float


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    style: IslandStyle
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result.get("valid", False))

    @property
    def score(self) -> float:
        return float(self.result.get("score", np.inf))

    def copy(self) -> "Individual":
        return Individual(
            genome=self.genome.copy(),
            result=dict(self.result),
            island=self.island,
            generation=self.generation,
            origin=self.origin,
            style=self.style,
            rank=self.rank,
            crowding=self.crowding,
        )


@dataclass
class Island:
    style: IslandStyle
    rng: np.random.Generator
    population: list[Individual]


class BEMField:
    def __init__(self, bem: FixedMeshBEM, charge: Any) -> None:
        self.bem = bem
        self.charge = charge

    def electric_field(self, x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
        values = self.bem.field_batch(
            np.array([x_m], dtype=float),
            np.array([y_m], dtype=float),
            np.array([z_m], dtype=float),
            self.charge,
        )
        field = self.bem.asnumpy(values)[0, 0]
        return float(field[0]), float(field[1]), float(field[2])


def repair(genome: np.ndarray) -> np.ndarray:
    values = np.asarray(genome, dtype=float).copy()
    if values.shape != (N_GENES,):
        raise ValueError(f"Expected genome shape {(N_GENES,)}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Genome contains non-finite values.")
    return np.clip(values, LOW, HIGH)


def decode(genome: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = repair(genome)
    knots_m = softmax_gap_knots(
        values[GAPS],
        lock_m=float(values[LOCK]),
        min_gap_m=MIN_GAP_M,
    )
    return knots_m, np.r_[0.0, values[INNER], 0.0], np.r_[0.0, values[OUTER], 0.0]


def parameters(genome: np.ndarray) -> Any:
    values = repair(genome)
    knots_m, inner_m, outer_m = decode(values)
    contour = MovableKnotContour(
        knots_m=knots_m,
        inner_offsets_m=inner_m,
        outer_offsets_m=outer_m,
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


def elite_seeds() -> dict[str, np.ndarray]:
    A = np.array([
        -2.693919933820525, -3.8791883884875884, -2.3107549237131555, -2.6635360680776348, -2.604364450724748, -1.0188113709792743, -2.295002976432831, -1.58956654563482,
        -4.671047050058288e-06, -6.628886993159777e-06, -1.4230855848703506e-05, -9.938723998480415e-06, -6.968225587669311e-07, -8.992668411738875e-06, -4.4706657882594386e-07,
        8.96957498891864e-06, 1.2337918405266222e-05, 1.5843821583578765e-05, 6.160598233461362e-06, -3.0745376990661804e-06, 6.330512570923556e-07, 8.470178316320093e-07,
        1.6992351073699268e-04, -3.017299702847008e-05, 1.9768896889529104e-05, 2.477974346604927e-05, 1.5294601025453086e-04, 2.058947594763603, 1.3531585219616183e-07, 2.053237855986011e-05, 2.1517821592870727e-05,
    ], dtype=float)
    B = np.array([
        -2.709933932853829, -3.8337678030869418, -2.3338229158144213, -2.6934269345805704, -2.5421057953892996, -0.997926753075712, -2.295002976432831, -1.6903460656946008,
        -6.331026783805033e-06, -5.890457246891707e-06, -1.5281547606642362e-05, -9.999924830276548e-06, -1.26739540498944e-06, -9.912303077128267e-06, -1.3597748541155676e-06,
        7.274234673725851e-06, 9.817866241204519e-06, 1.4897250517529313e-05, 6.478090564381574e-06, -3.81292082739843e-06, -1.9943656647684993e-08, 2.796813400808457e-06,
        1.6687669509666937e-04, -2.9448708514443475e-05, 2.135454064244456e-05, 2.5211293026688254e-05, 1.5024507179705118e-04, 2.001091036207649, -9.751371789496804e-07, 2.263872675032877e-05, 2.191668209459116e-05,
    ], dtype=float)
    C = np.array([
        -3.5878694899167303, -3.153762938162855, -2.713733453435762, -2.3309751634792555, -1.942648043149124, -1.4907367463890762, -1.4675270878437812, -1.761484342471865,
        -3.5052820174793393e-06, -9.070629157996265e-06, -1.1218134594172648e-05, -1.5222896079720358e-05, -2.757105287824218e-06, -5.559592001521011e-06, -5.054037406570247e-06,
        -1.3584388230842257e-06, -6.336830747342028e-08, 7.49031343173437e-06, 1.1462476127269313e-05, 6.7139106982945445e-06, 5.139106510379214e-06, 1.7667972654596989e-06,
        1.5279141515234126e-04, -2.9765839678324105e-05, 1.952631811890536e-05, 2.627562347274212e-05, 1.4496027635234118e-04, 1.9605311344426817, -1.4640821799977548e-06, 7.499354408876493e-05, 2.569149473853044e-05,
    ], dtype=float)
    return {"A": repair(A), "B": repair(B), "C": repair(C)}


def base_sigma(mode: Mode, scale: float) -> np.ndarray:
    sigma = np.zeros(N_GENES, dtype=float)
    sigma[GAPS] = 0.015
    sigma[INNER] = 0.30e-6
    sigma[OUTER] = 0.30e-6
    sigma[LOCK] = 0.75e-6
    sigma[CENTER_IN] = 0.25e-6
    sigma[CENTER_OUT] = 0.25e-6
    sigma[START] = 0.50e-6
    sigma[LENGTH] = 2.00e-6
    sigma[POWER] = 0.020
    sigma[BULGE] = 0.20e-6
    sigma[BULGE_CENTER] = 1.00e-6
    sigma[BULGE_WIDTH] = 0.75e-6
    if mode == "centre":
        sigma[INNER] *= 2.0
        sigma[CENTER_IN:CENTER_OUT + 1] *= 2.0
        sigma[START] *= 1.5
    elif mode == "transition":
        sigma[INNER] *= 1.5
        sigma[OUTER] *= 1.5
        sigma[START:LENGTH + 1] *= 2.0
    elif mode == "shoulder":
        sigma[OUTER] *= 2.0
        sigma[LOCK] *= 1.5
        sigma[LENGTH] *= 2.0
        sigma[POWER] *= 1.5
        sigma[BULGE:BULGE_WIDTH + 1] *= 2.0
    elif mode == "height":
        sigma[INNER] *= 2.0
        sigma[CENTER_IN:CENTER_OUT + 1] *= 2.0
    elif mode == "barrier":
        sigma[OUTER] *= 2.0
        sigma[LENGTH] *= 1.5
        sigma[POWER] *= 1.5
        sigma[BULGE:BULGE_WIDTH + 1] *= 1.5
    elif mode == "global":
        sigma *= 1.35
    return sigma * scale


def mutate(genome: np.ndarray, mode: Mode, rng: np.random.Generator, scale: float, probability: float = 0.55) -> np.ndarray:
    child = repair(genome)
    sigma = base_sigma(mode, scale)
    active = rng.random(N_GENES) < probability
    child[active] += rng.normal(0.0, sigma[active])
    return repair(child)


def safe_mix(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = np.empty(N_GENES, dtype=float)
    contour_parent = left if rng.random() < 0.5 else right
    global_parent = left if rng.random() < 0.5 else right
    child[CONTOUR_BLOCK] = contour_parent[CONTOUR_BLOCK]
    child[GLOBAL_BLOCK] = global_parent[GLOBAL_BLOCK]
    return repair(child)


def local_blend(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.80, 0.95))
    if rng.random() < 0.5:
        alpha = 1.0 - alpha
    return repair(alpha * left + (1.0 - alpha) * right)


def source_genome(kind: SeedKind, seeds: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[str, np.ndarray]:
    A, B, C = seeds["A"], seeds["B"], seeds["C"]
    if kind == "A":
        return "seed_A", A.copy()
    if kind == "B":
        return "seed_B", B.copy()
    if kind == "C":
        return "seed_C", C.copy()
    if kind == "AB":
        return "blend_AB", local_blend(A, B, rng)
    if kind == "AC":
        return "mix_AC", safe_mix(A, C, rng)
    if kind == "BC":
        return "mix_BC", safe_mix(B, C, rng)
    mid = safe_mix(A, B, rng)
    return "mix_ABC", local_blend(mid, C, rng)


def rejected(reason: str, stage: str) -> dict[str, Any]:
    return {
        "valid": False,
        "verified": False,
        "reason": reason,
        "stage": stage,
        "barrier_ev": np.inf,
        "barrier_reference_ev": np.nan,
        "excursion_ev": np.inf,
        "peak_ev": np.inf,
        "dip_ev": np.inf,
        "dz_peak_m": np.inf,
        "dz_rms_m": np.inf,
        "lateral_peak_m": np.inf,
        "dz_centre_peak_m": np.inf,
        "dz_transition_peak_m": np.inf,
        "dz_arm_peak_m": np.inf,
        "excursion_centre_ev": np.inf,
        "excursion_transition_ev": np.inf,
        "excursion_arm_ev": np.inf,
        "score": np.inf,
        "panels": 0,
        "points": 0,
        "converged": 0,
    }


def build_mesh(parameter_object: Any, fast: bool) -> Any:
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


def masks(x_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.abs(np.asarray(x_m, dtype=float))
    centre = s <= CENTRE_LIMIT_M
    transition = (s > CENTRE_LIMIT_M) & (s <= TRANSITION_LIMIT_M)
    arm = (s > TRANSITION_LIMIT_M) & (s <= ARM_LIMIT_M)
    reference = (s >= REFERENCE_MIN_M) & (s <= REFERENCE_MAX_M)
    return centre, transition, arm, reference


def evaluate(genome: np.ndarray, cfg: Settings, stage: str = "coarse", retain: bool = False) -> tuple[dict[str, Any], Any | None]:
    try:
        values = repair(genome)
        parameter_object = parameters(values)
        report = check_x_junction_manufacturability(parameter_object)
        if not report.valid:
            return rejected("geometry: " + "; ".join(map(str, report.messages)), stage), None
        s_m = np.linspace(0.0, parameter_object.arm_length_m, 2001)
        inner_m, outer_m = parameter_object.rail_boundaries_m(s_m)
        if np.min(inner_m) <= 0.5e-6:
            return rejected("inner clearance", stage), None
        if np.min(outer_m - inner_m) < 18e-6:
            return rejected("rail width", stage), None
        model = build_mesh(parameter_object, cfg.fast and stage == "coarse")
        if model.n_panels > cfg.max_panels:
            return rejected("panel limit", stage), None
        bem = FixedMeshBEM(model.bem.panels_m, backend=cfg.backend)
        bem.assemble(block_rows=64)
        bem.factorize()
        bem.release_matrix()
        charge = bem.solve_masks(model.bem.electrode_voltages_v)
        field = BEMField(bem, charge)
        points = cfg.fine_points if stage == "fine" else cfg.coarse_points
        trace = trace_rf_transverse_minimum(
            field,
            np.linspace(-350e-6, 350e-6, points),
            initial_y_m=0.0,
            initial_z_m=cfg.target_z_m,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )
        converged = np.asarray(trace.converged, dtype=bool)
        if not trace.valid or converged.shape != (points,) or not np.all(converged):
            return rejected("incomplete trace", stage), None
        x_m = np.asarray(trace.x_m, dtype=float)
        y_m = np.asarray(trace.y_m, dtype=float)
        z_m = np.asarray(trace.z_m, dtype=float)
        if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)) or not np.all(np.isfinite(z_m)):
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
        barrier = compute_barrier_metrics(pseudo_ev, trace)
        if not barrier.valid or pseudo_ev.shape != (points,) or not np.all(np.isfinite(pseudo_ev)):
            return rejected("invalid pseudopotential", stage), None
        centre, transition, arm, reference = masks(x_m)
        if not (np.any(centre) and np.any(transition) and np.any(arm) and np.any(reference)):
            return rejected("incomplete zone coverage", stage), None
        z_ref_m = float(np.median(z_m[reference]))
        dz_profile_m = z_m - z_ref_m
        dz_centre = float(np.max(np.abs(dz_profile_m[centre])))
        dz_transition = float(np.max(np.abs(dz_profile_m[transition])))
        dz_arm = float(np.max(np.abs(dz_profile_m[arm])))
        dz_peak = float(np.max(np.abs(dz_profile_m)))
        dz_rms = float(np.sqrt(np.mean(dz_profile_m**2)))
        u_ref_ev = float(barrier.reference_energy_ev)
        relative_ev = pseudo_ev - u_ref_ev
        peak_ev = float(np.max(relative_ev[converged]))
        dip_ev = float(-np.min(relative_ev[converged]))
        excursion_ev = float(np.max(np.abs(relative_ev[converged])))
        if not np.isclose(barrier.barrier_height_ev, max(0.0, peak_ev), rtol=1e-8, atol=1e-13):
            return rejected("barrier/reference inconsistency", stage), None
        def excursion(mask: np.ndarray) -> float:
            return float(np.max(np.abs(relative_ev[mask & converged])))
        ex_centre = excursion(centre)
        ex_transition = excursion(transition)
        ex_arm = excursion(arm)
        lateral = float(np.max(np.abs(y_m)))
        score = float(max(dz_peak / SEARCH_DZ_M, barrier.barrier_height_ev / SEARCH_BARRIER_EV))
        result = {
            "valid": True,
            "verified": bool(stage == "fine" and dz_peak <= FINAL_DZ_M and barrier.barrier_height_ev <= FINAL_BARRIER_EV),
            "reason": "ok",
            "stage": stage,
            "backend": bem.backend_name,
            "barrier_ev": float(barrier.barrier_height_ev),
            "barrier_reference_ev": u_ref_ev,
            "excursion_ev": excursion_ev,
            "peak_ev": peak_ev,
            "dip_ev": dip_ev,
            "dz_peak_m": dz_peak,
            "dz_rms_m": dz_rms,
            "lateral_peak_m": lateral,
            "dz_centre_peak_m": dz_centre,
            "dz_transition_peak_m": dz_transition,
            "dz_arm_peak_m": dz_arm,
            "excursion_centre_ev": ex_centre,
            "excursion_transition_ev": ex_transition,
            "excursion_arm_ev": ex_arm,
            "score": score,
            "panels": int(model.n_panels),
            "points": int(points),
            "converged": int(np.sum(converged)),
        }
        data = (model, trace, pseudo_ev, field, parameter_object, z_ref_m, u_ref_ev) if retain else None
        return result, data
    except Exception as error:
        return rejected(f"geometry/BEM error {type(error).__name__}: {str(error).replace(chr(10), ' ').strip()}", stage), None


def quality_ok(result: dict[str, Any]) -> bool:
    return bool(result["valid"] and result["barrier_ev"] <= INITIAL_BARRIER_CEILING_EV and result["dz_peak_m"] <= INITIAL_DZ_CEILING_M)


def dominates(left: Individual, right: Individual) -> bool:
    if left.valid != right.valid:
        return left.valid
    if not left.valid:
        return False
    l = np.array([left.result["dz_peak_m"], left.result["barrier_ev"]])
    r = np.array([right.result["dz_peak_m"], right.result["barrier_ev"]])
    return bool(np.all(l <= r) and np.any(l < r))


def rank_population(population: list[Individual]) -> None:
    for item in population:
        item.rank = 0
        item.crowding = 0.0
    for i, left in enumerate(population):
        for j, right in enumerate(population):
            if i != j and dominates(right, left):
                left.rank += 1
    valid = [item for item in population if item.valid]
    if valid:
        for objective in ("dz_peak_m", "barrier_ev"):
            ordered = sorted(valid, key=lambda item: item.result[objective])
            ordered[0].crowding = np.inf
            ordered[-1].crowding = np.inf


def choose_best(population: list[Individual], size: int) -> list[Individual]:
    rank_population(population)
    ordered = sorted(population, key=lambda item: (not item.valid, item.rank, item.score, -item.crowding))
    return [item.copy() for item in ordered[:size]]


def tournament(population: list[Individual], rng: np.random.Generator) -> Individual:
    first, second = (population[int(i)] for i in rng.integers(0, len(population), 2))
    return min((first, second), key=lambda item: (not item.valid, item.rank, item.score, -item.crowding))


def island_styles() -> list[IslandStyle]:
    layout: list[tuple[SeedKind, int]] = [("A", 5), ("B", 5), ("C", 4), ("AB", 5), ("AC", 4), ("BC", 4), ("ABC", 3)]
    modes: tuple[Mode, ...] = ("centre", "transition", "shoulder", "height", "barrier", "global")
    styles: list[IslandStyle] = []
    for kind, count in layout:
        for index in range(count):
            scale = 1.0 if kind in ("A", "B", "C") else 1.2
            styles.append(IslandStyle(f"{kind.lower()}_{modes[len(styles) % len(modes)]}_{index:02d}", kind, modes[len(styles) % len(modes)], scale))
    return styles


def serialize(item: Individual) -> dict[str, Any]:
    knots, inner, outer = decode(item.genome)
    return {
        "island": item.island,
        "generation": item.generation,
        "origin": item.origin,
        "style": item.style.name,
        "seed_kind": item.style.seed_kind,
        "mode": item.style.mode,
        "rank": item.rank,
        "crowding": item.crowding,
        "genome": item.genome.tolist(),
        "movable_knots_m": knots.tolist(),
        "inner_offsets_m": inner.tolist(),
        "outer_offsets_m": outer.tolist(),
        "lock_m": float(item.genome[LOCK]),
        **item.result,
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def draw_geometry(axis: Any, item: Individual) -> None:
    parameter_object = parameters(item.genome)
    s_m = np.linspace(0.0, 260e-6, 900)
    inner_m, outer_m = parameter_object.rail_boundaries_m(s_m)
    for theta in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
        c, s = np.cos(theta), np.sin(theta)
        xi, yi = s_m * c - inner_m * s, s_m * s + inner_m * c
        xo, yo = s_m * c - outer_m * s, s_m * s + outer_m * c
        axis.fill(np.r_[xi, xo[::-1]] * 1e6, np.r_[yi, yo[::-1]] * 1e6, color="#db4f4f", edgecolor="black", linewidth=0.35)
    axis.set(xlim=(-260, 260), ylim=(-260, 260), aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{item.style.name}\nU={item.result['barrier_ev'] * 1e3:.2f} meV, dz={item.result['dz_peak_m'] * 1e6:.2f} um", fontsize=8)


def plot_history(history: list[dict[str, Any]], output: Path) -> None:
    if not history:
        return
    generation = [row["generation"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(generation, [row.get("best_U_meV", np.nan) for row in history], marker="o")
    axes[0, 0].axhline(1.0, color="darkorange", linestyle=":")
    axes[0, 0].axhline(0.1, color="crimson", linestyle="--")
    axes[0, 0].set(title="Best legacy barrier", xlabel="generation", ylabel="barrier [meV]", yscale="log")
    axes[0, 1].plot(generation, [row.get("best_dz_um", np.nan) for row in history], marker="o")
    axes[0, 1].axhline(3.0, color="darkorange", linestyle=":")
    axes[0, 1].axhline(1.0, color="crimson", linestyle="--")
    axes[0, 1].set(title="Best height deviation", xlabel="generation", ylabel="dz [um]", yscale="log")
    axes[1, 0].plot(generation, [row.get("best_excursion_meV", np.nan) for row in history], marker="o")
    axes[1, 0].set(title="Diagnostic excursion", xlabel="generation", ylabel="U excursion [meV]", yscale="log")
    axes[1, 1].plot(generation, [row.get("valid", 0) for row in history], marker="o", label="valid")
    axes[1, 1].plot(generation, [row.get("invalid", 0) for row in history], marker="o", label="invalid")
    axes[1, 1].legend()
    axes[1, 1].set(title="Population validity", xlabel="generation", ylabel="individuals")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    fig.savefig(output / "convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--offspring", type=int, default=3)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--migrants", type=int, default=2)
    p.add_argument("--migration-interval", type=int, default=3)
    p.add_argument("--retain-per-island", type=int, default=5)
    p.add_argument("--coarse-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=181)
    p.add_argument("--fine-top", type=int, default=20)
    p.add_argument("--max-panels", type=int, default=6500)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--output-dir", type=Path, default=Path("reports/14_seeded_diverse_zonal_ga"))
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.smoke:
        args.population = 8
        args.offspring = 2
        args.generations = 2
        args.coarse_points = 31
        args.fine_points = 61
        args.fine_top = 6
        args.max_panels = 5000
        args.fast = True
    backend = available_backend(args.backend)
    if not backend.available:
        raise RuntimeError(backend.reason)
    cfg = Settings(
        backend=backend.selected,
        population=args.population,
        offspring=args.offspring,
        generations=args.generations,
        migrants=args.migrants,
        migration_interval=args.migration_interval,
        retain_per_island=args.retain_per_island,
        coarse_points=args.coarse_points,
        fine_points=args.fine_points,
        fine_top=args.fine_top,
        max_panels=args.max_panels,
        fast=args.fast,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    styles = island_styles()
    seeds = elite_seeds()
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def cached_eval(genome: np.ndarray, stage: str = "coarse", retain: bool = False) -> tuple[dict[str, Any], Any | None]:
        if retain or stage == "fine":
            return evaluate(genome, cfg, stage, retain)
        key = tuple(np.round(repair(genome), 12))
        if key not in cache:
            cache[key], _ = evaluate(genome, cfg, stage, False)
        return dict(cache[key]), None

    print(f"Workflow 14 | backend={cfg.backend} | islands={len(styles)} | population={cfg.population}")
    print("Exact seed validation using legacy barrier_ev:")
    for name, genome in seeds.items():
        result, _ = cached_eval(genome)
        print(f"  {name}: valid={result['valid']} U={result['barrier_ev'] * 1e3:.6f} meV dz={result['dz_peak_m'] * 1e6:.6f} um excursion={result['excursion_ev'] * 1e3:.6f} meV")
    (cfg.output_dir / "settings.json").write_text(json.dumps({"settings": asdict(cfg), "styles": [asdict(style) for style in styles]}, indent=2, default=str), encoding="utf-8")

    master = np.random.default_rng(cfg.seed)
    islands: list[Island] = []
    with tqdm(total=len(styles) * cfg.population, desc="Initial seeded BEM", unit="candidate") as progress:
        for island_id, style in enumerate(styles):
            rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
            population: list[Individual] = []
            label, base = source_genome(style.seed_kind, seeds, rng)
            exact_result, _ = cached_eval(base)
            population.append(Individual(base.copy(), exact_result, island_id, 0, label, style))
            progress.update()
            while len(population) < cfg.population:
                accepted = False
                best_genome, best_result = base.copy(), exact_result
                for attempt in range(MAX_INITIAL_ATTEMPTS):
                    proposal = mutate(base, style.mode, rng, style.scale, probability=0.55 if attempt < 8 else 0.35)
                    result, _ = cached_eval(proposal)
                    if result["valid"] and result["score"] < best_result["score"]:
                        best_genome, best_result = proposal, result
                    if quality_ok(result):
                        best_genome, best_result = proposal, result
                        accepted = True
                        break
                population.append(Individual(best_genome, best_result, island_id, 0, f"initial_{label}_{'accepted' if accepted else 'fallback'}", style))
                progress.update()
            rank_population(population)
            islands.append(Island(style, rng, population))

    archive: list[Individual] = []
    history: list[dict[str, Any]] = []

    def add_archive(items: list[Individual]) -> None:
        nonlocal archive
        unique = {tuple(np.round(item.genome, 12)): item.copy() for item in archive + items if item.valid}
        archive = sorted(unique.values(), key=lambda item: item.score)[:2000]

    def write_state(generation: int) -> None:
        all_items = [item for island in islands for item in island.population]
        valid = [item for item in all_items if item.valid]
        invalid = [item for item in all_items if not item.valid]
        if valid:
            best_u = min(valid, key=lambda item: item.result["barrier_ev"])
            best_z = min(valid, key=lambda item: item.result["dz_peak_m"])
            best_joint = min(valid, key=lambda item: item.score)
            row = {
                "generation": generation,
                "valid": len(valid),
                "invalid": len(invalid),
                "best_U_meV": best_u.result["barrier_ev"] * 1e3,
                "best_dz_um": best_z.result["dz_peak_m"] * 1e6,
                "best_excursion_meV": best_joint.result["excursion_ev"] * 1e3,
                "joint_U_meV": best_joint.result["barrier_ev"] * 1e3,
                "joint_dz_um": best_joint.result["dz_peak_m"] * 1e6,
                "joint_score": best_joint.score,
                "joint_origin": best_joint.origin,
                "joint_style": best_joint.style.name,
                "joint_dz_centre_um": best_joint.result["dz_centre_peak_m"] * 1e6,
                "joint_dz_transition_um": best_joint.result["dz_transition_peak_m"] * 1e6,
                "joint_dz_arm_um": best_joint.result["dz_arm_peak_m"] * 1e6,
                "joint_U_centre_meV": best_joint.result["excursion_centre_ev"] * 1e3,
                "joint_U_transition_meV": best_joint.result["excursion_transition_ev"] * 1e3,
                "joint_U_arm_meV": best_joint.result["excursion_arm_ev"] * 1e3,
                "unique_evaluations": len(cache),
            }
        else:
            row = {"generation": generation, "valid": 0, "invalid": len(invalid), "unique_evaluations": len(cache)}
        history.append(row)
        save_csv(history, cfg.output_dir / "history.csv")
        save_csv([serialize(item) for island in islands for item in island.population], cfg.output_dir / f"population_gen_{generation:04d}.csv")
        save_csv([serialize(item) for island in islands for item in island.population for item in choose_best(island.population, min(cfg.retain_per_island, len(island.population)))], cfg.output_dir / "island_elites" / f"elites_gen_{generation:04d}.csv")
        print(f"G{generation:03d}", row, flush=True)

    add_archive([item for island in islands for item in island.population])
    write_state(0)

    for generation in range(1, cfg.generations + 1):
        next_islands: list[Island] = []
        with tqdm(total=len(islands) * cfg.offspring, desc=f"Generation {generation}", unit="candidate") as progress:
            for island_id, island in enumerate(islands):
                rank_population(island.population)
                elite = choose_best(island.population, max(2, min(5, len(island.population))))
                children: list[Individual] = []
                while len(children) < cfg.offspring:
                    r = island.rng.random()
                    if r < 0.55:
                        parent = elite[int(island.rng.integers(min(2, len(elite))))]
                        genome = parent.genome.copy()
                        origin = "elite_mutation"
                    elif r < 0.78:
                        left = tournament(island.population, island.rng)
                        right = tournament(island.population, island.rng)
                        genome = local_blend(left.genome, right.genome, island.rng)
                        origin = "local_blend"
                    elif r < 0.90:
                        left = tournament(island.population, island.rng)
                        right = tournament(island.population, island.rng)
                        genome = safe_mix(left.genome, right.genome, island.rng)
                        origin = "safe_mix"
                    else:
                        label, genome = source_genome(island.style.seed_kind, seeds, island.rng)
                        origin = f"restart_{label}"
                    genome = mutate(genome, island.style.mode, island.rng, island.style.scale, probability=0.55)
                    result, _ = cached_eval(genome)
                    children.append(Individual(genome, result, island_id, generation, origin, island.style))
                    progress.update()
                next_islands.append(Island(island.style, island.rng, choose_best(island.population + children, cfg.population)))
        islands = next_islands
        if cfg.migration_interval > 0 and generation % cfg.migration_interval == 0 and len(islands) > 1:
            outgoing = [choose_best(island.population, min(cfg.migrants, cfg.population)) for island in islands]
            for source, migrants in enumerate(outgoing):
                destination = (source + 1) % len(islands)
                target = islands[destination]
                incoming = [Individual(m.genome.copy(), dict(m.result), destination, generation, f"migrant_{source:02d}", target.style, m.rank, m.crowding) for m in migrants]
                islands[destination] = Island(target.style, target.rng, choose_best(target.population + incoming, cfg.population))
        add_archive([item for island in islands for item in island.population])
        write_state(generation)

    plot_history(history, cfg.output_dir)
    selected = sorted(archive, key=lambda item: item.score)[:cfg.fine_top]
    fine_rows: list[dict[str, Any]] = []
    for number, item in enumerate(tqdm(selected, desc="Fine validation", unit="candidate"), start=1):
        result, data = cached_eval(item.genome, stage="fine", retain=True)
        row = {"candidate": number, **serialize(item), **{f"fine_{key}": value for key, value in result.items()}}
        fine_rows.append(row)
        if data is not None:
            model, trace, pseudo_ev, field, parameter_object, z_ref, u_ref = data
            directory = cfg.output_dir / "fine" / f"candidate_{number:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            plot_mask_vs_bem(parameter_object, model, directory / "mask_vs_bem.png", f"Candidate {number}")
            x_um = np.asarray(trace.x_m) * 1e6
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
            axes[0].plot(x_um, (np.asarray(trace.z_m) - z_ref) * 1e6)
            axes[0].set_ylabel("z - arm ref [um]")
            axes[1].plot(x_um, (pseudo_ev - u_ref) * 1e3)
            axes[1].set_ylabel("U - old ref [meV]")
            axes[2].plot(x_um, np.asarray(trace.y_m) * 1e6)
            axes[2].set(xlabel="x [um]", ylabel="y [um]")
            for axis in axes:
                axis.grid(alpha=0.3)
                axis.axvspan(-320, -160, color="#DDEBFF", alpha=0.25)
                axis.axvspan(-160, -45, color="#FFF2CC", alpha=0.25)
                axis.axvspan(-45, 45, color="#FCE4D6", alpha=0.30)
                axis.axvspan(45, 160, color="#FFF2CC", alpha=0.25)
                axis.axvspan(160, 320, color="#DDEBFF", alpha=0.25)
            fig.suptitle(f"Candidate {number}: old U={result['barrier_ev'] * 1e3:.4f} meV, excursion={result['excursion_ev'] * 1e3:.4f} meV")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(directory / "profiles_zonal.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            (directory / "candidate.json").write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
    save_csv(fine_rows, cfg.output_dir / "fine_validation.csv")
    top5 = sorted(fine_rows, key=lambda row: (row.get("fine_barrier_ev", np.inf), row.get("fine_dz_peak_m", np.inf)))[:5]
    (cfg.output_dir / "top5_candidates.json").write_text(json.dumps(top5, indent=2, default=float), encoding="utf-8")
    save_csv(top5, cfg.output_dir / "top5_candidates.csv")
    print(f"Completed. Top candidates: {cfg.output_dir / 'top5_candidates.json'}")


if __name__ == "__main__":
    main()
