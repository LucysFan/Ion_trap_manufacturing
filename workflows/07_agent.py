"""Workflow 07 agent: seeded island GA for the existing connected-C4v X-junction.

Geometry
--------
This workflow searches only the established workflow-07 house-style junction:
  * a shared inner RF rail contour;
  * a shared outer RF rail contour;
  * C4v symmetry imposed by make_house_style_x_junction;
  * no disks, rings, central RF cross, moat, or local RF feature genes.

Corrected height metric
-----------------------
The historic 07 scripts used compute_path_metrics(trace), whose height-reference
semantics are not used here. The optimization target is calculated directly:

    dz_peak = max_i |z_RF_null(i) - TARGET_ION_HEIGHT_M|

only over converged trace points. The RMS metric uses the same fixed target.
This makes the target dz_peak < 3 um unambiguous.

The file uses local plotting helpers and therefore does not import plotting
functions from core.optimization.junction_evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from core.analysis.barrier import compute_barrier_metrics, pseudopotential_profile_ev
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem


CONTOUR_KNOTS_M = np.asarray((30e-6, 60e-6, 90e-6, 120e-6, 150e-6), dtype=float)
N_GENES = 11
OBJECTIVE_NAMES = (
    "fixed_target_height_peak_over_3um",
    "barrier_over_50mev",
    "lateral_peak_over_5um",
    "fixed_target_height_rms_over_1um",
    "contour_roughness",
)


@dataclass(frozen=True)
class Bounds:
    inner_offset_low_m: float = -35e-6
    inner_offset_high_m: float = 15e-6
    outer_offset_low_m: float = -25e-6
    outer_offset_high_m: float = 30e-6
    central_inner_shift_low_m: float = -42e-6
    central_inner_shift_high_m: float = -14e-6
    central_outer_shift_low_m: float = 8e-6
    central_outer_shift_high_m: float = 38e-6
    taper_length_low_m: float = 105e-6
    taper_length_high_m: float = 215e-6
    taper_power_low: float = 1.15
    taper_power_high: float = 3.10
    start_radius_low_m: float = 22e-6
    start_radius_high_m: float = 46e-6
    max_adjacent_inner_difference_m: float = 22e-6
    max_adjacent_outer_difference_m: float = 24e-6

    def low(self) -> np.ndarray:
        return np.asarray([
            *([self.inner_offset_low_m] * 3),
            *([self.outer_offset_low_m] * 3),
            self.central_inner_shift_low_m,
            self.central_outer_shift_low_m,
            self.taper_length_low_m,
            self.taper_power_low,
            self.start_radius_low_m,
        ], dtype=float)

    def high(self) -> np.ndarray:
        return np.asarray([
            *([self.inner_offset_high_m] * 3),
            *([self.outer_offset_high_m] * 3),
            self.central_inner_shift_high_m,
            self.central_outer_shift_high_m,
            self.taper_length_high_m,
            self.taper_power_high,
            self.start_radius_high_m,
        ], dtype=float)


@dataclass(frozen=True)
class IslandSpec:
    island_id: int
    name: str
    init_fraction: float
    mutation_fraction: float
    crossover_probability: float
    mutation_probability: float
    gene_weights: tuple[float, ...]
    objective_weights: tuple[float, ...]
    coherent_outer: bool = False
    coherent_inner: bool = False


ISLAND_SPECS: tuple[IslandSpec, ...] = (
    IslandSpec(0, "height_local", 0.07, 0.025, 0.92, 0.55,
               (1, 1, 1, 1, 1, 1, 0.7, 0.7, 0.6, 0.5, 0.5),
               (5.0, 1.0, 0.8, 1.2, 0.15), coherent_inner=True),
    IslandSpec(1, "inner_height", 0.17, 0.080, 0.88, 0.90,
               (1.9, 1.9, 1.9, 0.5, 0.5, 0.5, 1.4, 0.5, 0.8, 0.6, 0.8),
               (5.0, 1.2, 0.8, 1.2, 0.15), coherent_inner=True),
    IslandSpec(2, "outer_barrier", 0.18, 0.090, 0.88, 0.93,
               (0.5, 0.5, 0.5, 2.4, 2.4, 2.4, 0.5, 1.8, 1.2, 0.8, 1.0),
               (2.2, 4.8, 0.8, 0.8, 0.10), coherent_outer=True),
    IslandSpec(3, "taper_barrier", 0.17, 0.085, 0.92, 0.88,
               (0.7, 0.7, 0.7, 1.5, 1.5, 1.5, 1.7, 2.1, 2.3, 2.3, 2.3),
               (2.5, 4.2, 0.8, 0.8, 0.10), coherent_outer=True),
    IslandSpec(4, "balanced_pareto", 0.11, 0.045, 0.95, 0.72,
               (1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.8, 0.8, 1.0, 0.8, 0.8),
               (3.5, 2.7, 0.8, 1.0, 0.15), coherent_outer=True, coherent_inner=True),
    IslandSpec(5, "wide_basin", 0.23, 0.115, 0.84, 0.95,
               (1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4),
               (3.0, 3.0, 0.8, 1.0, 0.12), coherent_outer=True, coherent_inner=True),
)


@dataclass(frozen=True)
class Config:
    islands: int
    population: int
    offspring: int
    generations: int
    migration_interval: int
    migrants: int
    checkpoint_interval: int
    seed: int
    path_points: int
    fast: bool
    output_dir: Path
    render_top: int
    render_maps: bool
    bounds: Bounds = field(default_factory=Bounds)


@dataclass
class Candidate:
    vector: np.ndarray
    result: dict[str, Any]
    objectives: np.ndarray
    violation: float
    feasible: bool
    island_id: int
    generation: int
    origin: str
    rank: int = 0
    crowding: float = 0.0

    def copy(self) -> "Candidate":
        return Candidate(
            vector=self.vector.copy(),
            result=dict(self.result),
            objectives=self.objectives.copy(),
            violation=float(self.violation),
            feasible=bool(self.feasible),
            island_id=int(self.island_id),
            generation=int(self.generation),
            origin=str(self.origin),
            rank=int(self.rank),
            crowding=float(self.crowding),
        )


@dataclass
class Island:
    spec: IslandSpec
    rng: np.random.Generator
    population: list[Candidate]
    best_quality: float = np.inf
    stagnation: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="07 agent: corrected-height connected-C4v island GA")
    parser.add_argument("--islands", type=int, default=6)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--offspring", type=int, default=None)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--migration-interval", type=int, default=5)
    parser.add_argument("--migrants", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--path-points", type=int, default=41)
    parser.add_argument("--render-top", type=int, default=4)
    parser.add_argument("--render-maps", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output-dir", default="reports/07_agent")
    return parser


def baseline_vector() -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, -25e-6, 20e-6, 150e-6, 2.0, 30e-6], dtype=float)


def seed_vectors() -> list[tuple[str, np.ndarray]]:
    baseline = baseline_vector()
    height = baseline.copy()
    height[:3] = (-12e-6, -12e-6, -9e-6)
    barrier = baseline.copy()
    barrier[:3] = (-12e-6, -8e-6, -9e-6)
    seeds: list[tuple[str, np.ndarray]] = [
        ("seed_baseline_07", baseline),
        ("seed_height_07f_legacy", height),
        ("seed_barrier_07f_legacy", barrier),
    ]
    for base_name, base_vector in (("seed_height", height), ("seed_barrier", barrier)):
        for amplitude_um in (-10.0, -5.0, 5.0, 10.0):
            child = base_vector.copy()
            child[3:6] = amplitude_um * 1e-6
            seeds.append((f"{base_name}_outer_{amplitude_um:+.0f}um", child))
    return seeds


def clip_vector(vector: np.ndarray, bounds: Bounds) -> np.ndarray:
    value = np.clip(np.asarray(vector, dtype=float), bounds.low(), bounds.high())
    for start, maximum in ((0, bounds.max_adjacent_inner_difference_m), (3, bounds.max_adjacent_outer_difference_m)):
        controls = value[start:start + 3].copy()
        for index in range(1, len(controls)):
            controls[index] = np.clip(controls[index], controls[index - 1] - maximum, controls[index - 1] + maximum)
        for index in range(len(controls) - 2, -1, -1):
            controls[index] = np.clip(controls[index], controls[index + 1] - maximum, controls[index + 1] + maximum)
        value[start:start + 3] = controls
    return np.clip(value, bounds.low(), bounds.high())


def profiles(vector: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (0.0, float(vector[0]), float(vector[1]), float(vector[2]), 0.0),
        (0.0, float(vector[3]), float(vector[4]), float(vector[5]), 0.0),
    )


def make_parameters(vector: np.ndarray):
    inner, outer = profiles(vector)
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=float(vector[8]),
        inner_edge_shift_at_centre_m=float(vector[6]),
        outer_edge_shift_at_centre_m=float(vector[7]),
        taper_power=float(vector[9]),
        rf_start_radius_override_m=float(vector[10]),
        inner_contour_knots_m=tuple(CONTOUR_KNOTS_M),
        inner_contour_offsets_m=inner,
        outer_contour_knots_m=tuple(CONTOUR_KNOTS_M),
        outer_contour_offsets_m=outer,
    )


def empty_result(vector: np.ndarray, reason: str) -> dict[str, Any]:
    inner, outer = profiles(vector)
    return {
        "valid": False,
        "reason": reason,
        "height_peak_target_m": np.inf,
        "height_rms_target_m": np.inf,
        "height_peak_arm_referenced_m": np.nan,
        "lateral_peak_m": np.inf,
        "barrier_ev": np.inf,
        "n_converged": 0,
        "n_points": 0,
        "convergence_fraction": 0.0,
        "panel_count": 0,
        "inner_offsets_m": list(inner),
        "outer_offsets_m": list(outer),
        "central_inner_shift_m": float(vector[6]),
        "central_outer_shift_m": float(vector[7]),
        "taper_length_m": float(vector[8]),
        "taper_power": float(vector[9]),
        "start_radius_m": float(vector[10]),
    }


def mesh_and_route(parameters: Any, fast: bool):
    if fast:
        model = build_geometry_aware_quadtree_x_junction_bem(
            parameters,
            central_half_extent_m=150e-6,
            central_max_cell_m=40e-6,
            boundary_max_cell_m=15e-6,
            outer_max_cell_m=180e-6,
            min_cell_m=7.5e-6,
        )
        return model, 170e-6
    model = build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )
    return model, 350e-6


def solve_candidate(vector: np.ndarray, cfg: Config, *, diagnostics: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    vector = clip_vector(vector, cfg.bounds)
    started = time.perf_counter()
    try:
        parameters = make_parameters(vector)
        manufacturing = check_x_junction_manufacturability(parameters)
        if not manufacturing.valid:
            return empty_result(vector, "manufacturability:" + ";".join(manufacturing.messages)), None
        model, route_extent_m = mesh_and_route(parameters, cfg.fast)
        model.bem.assemble(show_progress=False)
        model.bem.solve(show_progress=False)
    except Exception as error:
        return empty_result(vector, f"bem_failure:{type(error).__name__}:{error}"), None
    try:
        point_count = max(cfg.path_points, 81) if diagnostics else cfg.path_points
        x_m = np.linspace(-route_extent_m, route_extent_m, point_count)
        trace = trace_rf_transverse_minimum(
            model.bem,
            x_m,
            initial_y_m=0.0,
            initial_z_m=TARGET_ION_HEIGHT_M,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6,
        )
        potential_ev = pseudopotential_profile_ev(
            model.bem,
            trace,
            rf_voltage_peak_v=100.0,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        )
        barrier = compute_barrier_metrics(potential_ev, trace)
    except Exception as error:
        return empty_result(vector, f"path_failure:{type(error).__name__}:{error}"), None

    converged = np.asarray(trace.converged, dtype=bool)
    z_m = np.asarray(trace.z_m, dtype=float)
    y_m = np.asarray(trace.y_m, dtype=float)
    residual_v_m = np.asarray(trace.transverse_residual_v_m, dtype=float)
    x_m = np.asarray(trace.x_m, dtype=float)
    potential_ev = np.asarray(potential_ev, dtype=float)
    finite = converged & np.isfinite(z_m) & np.isfinite(y_m) & np.isfinite(potential_ev)
    n_converged = int(np.count_nonzero(finite))
    n_points = int(len(x_m))
    fraction = n_converged / max(n_points, 1)
    if n_converged:
        dz_target_m = z_m[finite] - TARGET_ION_HEIGHT_M
        height_peak_target_m = float(np.max(np.abs(dz_target_m)))
        height_rms_target_m = float(np.sqrt(np.mean(dz_target_m ** 2)))
        lateral_peak_m = float(np.max(np.abs(y_m[finite])))
        arm_count = max(1, min(5, n_converged // 3))
        arm_reference_m = float(np.mean(np.r_[z_m[finite][:arm_count], z_m[finite][-arm_count:]]))
        height_peak_arm_referenced_m = float(np.max(np.abs(z_m[finite] - arm_reference_m)))
    else:
        height_peak_target_m = np.inf
        height_rms_target_m = np.inf
        lateral_peak_m = np.inf
        height_peak_arm_referenced_m = np.nan
    scalar_values = np.asarray([height_peak_target_m, height_rms_target_m, lateral_peak_m, barrier.barrier_height_ev], dtype=float)
    usable = bool(n_converged >= max(5, int(0.60 * n_points)) and np.all(np.isfinite(scalar_values)) and barrier.valid)
    inner, outer = profiles(vector)
    result = {
        "valid": usable,
        "reason": "ok" if usable else f"partial_trace:converged={n_converged}/{n_points};barrier_valid={barrier.valid}",
        "trace_valid": bool(trace.valid),
        "barrier_valid": bool(barrier.valid),
        "height_peak_target_m": height_peak_target_m,
        "height_rms_target_m": height_rms_target_m,
        "height_peak_arm_referenced_m": height_peak_arm_referenced_m,
        "lateral_peak_m": lateral_peak_m,
        "barrier_ev": float(barrier.barrier_height_ev),
        "barrier_reference_ev": float(barrier.reference_energy_ev),
        "barrier_maximum_ev": float(barrier.maximum_energy_ev),
        "n_converged": n_converged,
        "n_points": n_points,
        "convergence_fraction": float(fraction),
        "panel_count": int(model.n_panels),
        "runtime_s": time.perf_counter() - started,
        "inner_offsets_m": list(inner),
        "outer_offsets_m": list(outer),
        "central_inner_shift_m": float(vector[6]),
        "central_outer_shift_m": float(vector[7]),
        "taper_length_m": float(vector[8]),
        "taper_power": float(vector[9]),
        "start_radius_m": float(vector[10]),
    }
    artifacts = None
    if diagnostics:
        artifacts = {
            "model": model,
            "trace": trace,
            "potential_ev": potential_ev,
            "barrier": barrier,
            "finite": finite,
        }
    return result, artifacts


def roughness(vector: np.ndarray) -> float:
    inner, outer = profiles(vector)
    inner_second = np.diff(np.asarray(inner), n=2) / 20e-6
    outer_second = np.diff(np.asarray(outer), n=2) / 25e-6
    return float(np.sqrt(np.mean(inner_second ** 2) + np.mean(outer_second ** 2)))


def objective_vector(result: dict[str, Any], vector: np.ndarray) -> np.ndarray:
    values = np.asarray([
        result["height_peak_target_m"], result["barrier_ev"], result["lateral_peak_m"], result["height_rms_target_m"],
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        return np.full(len(OBJECTIVE_NAMES), np.inf)
    return np.asarray([
        values[0] / 3e-6,
        values[1] / 0.050,
        values[2] / 5e-6,
        values[3] / 1e-6,
        roughness(vector),
    ], dtype=float)


def constraint_violation(result: dict[str, Any]) -> tuple[bool, float]:
    h = float(result.get("height_peak_target_m", np.inf))
    y = float(result.get("lateral_peak_m", np.inf))
    b = float(result.get("barrier_ev", np.inf))
    fraction = float(result.get("convergence_fraction", 0.0))
    if not np.all(np.isfinite([h, y, b, fraction])):
        return False, 1e6
    violation = (
        max(0.0, h - 25e-6) / 25e-6
        + max(0.0, y - 10e-6) / 10e-6
        + max(0.0, b - 0.30) / 0.30
        + max(0.0, 0.60 - fraction) / 0.60
    )
    return bool(violation <= 0.0), float(violation)


def make_candidate(vector: np.ndarray, island_id: int, generation: int, origin: str, cfg: Config) -> Candidate:
    vector = clip_vector(vector, cfg.bounds)
    result, _ = solve_candidate(vector, cfg)
    feasible, violation = constraint_violation(result)
    return Candidate(vector, result, objective_vector(result, vector), violation, feasible, island_id, generation, origin)


def quality(candidate: Candidate, weights: tuple[float, ...] | None = None) -> float:
    if not np.all(np.isfinite(candidate.objectives)):
        return np.inf
    if weights is None:
        weights = (3.5, 2.5, 0.8, 1.0, 0.12)
    convergence_penalty = 20.0 * max(0.0, 0.60 - float(candidate.result.get("convergence_fraction", 0.0)))
    return float(np.dot(np.asarray(weights, dtype=float), candidate.objectives) + convergence_penalty)


def dominates(first: Candidate, second: Candidate) -> bool:
    if first.feasible and not second.feasible:
        return True
    if second.feasible and not first.feasible:
        return False
    if not first.feasible and not second.feasible and not np.isclose(first.violation, second.violation, rtol=1e-12, atol=1e-12):
        return bool(first.violation < second.violation)
    return bool(np.all(first.objectives <= second.objectives) and np.any(first.objectives < second.objectives))


def rank_crowding(population: list[Candidate]) -> list[list[int]]:
    n = len(population)
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    dominated_by = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i], population[j]):
                dominates_set[i].append(j)
                dominated_by[j] += 1
            elif dominates(population[j], population[i]):
                dominates_set[j].append(i)
                dominated_by[i] += 1
    for i in range(n):
        if dominated_by[i] == 0:
            fronts[0].append(i)
    level = 0
    while fronts[level]:
        next_front: list[int] = []
        for i in fronts[level]:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        level += 1
        fronts.append(next_front)
    fronts = fronts[:-1]
    for rank, front in enumerate(fronts):
        for index in front:
            population[index].rank = rank
            population[index].crowding = 0.0
        if len(front) <= 2:
            for index in front:
                population[index].crowding = np.inf
            continue
        matrix = np.asarray([population[index].objectives for index in front], dtype=float)
        finite_rows = np.all(np.isfinite(matrix), axis=1)
        positions = np.flatnonzero(finite_rows)
        if len(positions) <= 2:
            for position in positions:
                population[front[int(position)]].crowding = np.inf
            continue
        finite = matrix[positions]
        for obj in range(finite.shape[1]):
            order = np.argsort(finite[:, obj])
            population[front[int(positions[int(order[0])])]].crowding = np.inf
            population[front[int(positions[int(order[-1])])]].crowding = np.inf
            span = finite[order[-1], obj] - finite[order[0], obj]
            if not np.isfinite(span) or span <= 1e-15:
                continue
            for k in range(1, len(order) - 1):
                item = population[front[int(positions[int(order[k])])]]
                if np.isfinite(item.crowding):
                    increment = (finite[order[k + 1], obj] - finite[order[k - 1], obj]) / span
                    if np.isfinite(increment):
                        item.crowding += float(increment)
    return fronts


def select_survivors(population: list[Candidate], size: int) -> list[Candidate]:
    fronts = rank_crowding(population)
    selected: list[Candidate] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(population[index].copy() for index in front)
            continue
        ordered = sorted(front, key=lambda index: (population[index].crowding, -quality(population[index])), reverse=True)
        selected.extend(population[index].copy() for index in ordered[:size - len(selected)])
        break
    return selected


def tournament(population: list[Candidate], rng: np.random.Generator, weights: tuple[float, ...]) -> Candidate:
    a, b = rng.integers(0, len(population), size=2)
    first, second = population[int(a)], population[int(b)]
    if first.rank != second.rank:
        return first if first.rank < second.rank else second
    if first.crowding != second.crowding:
        return first if first.crowding > second.crowding else second
    return first if quality(first, weights) <= quality(second, weights) else second


def crossover(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    alpha = rng.uniform(-0.15, 1.15, N_GENES)
    return clip_vector(alpha * first + (1.0 - alpha) * second, cfg.bounds)


def mutate(vector: np.ndarray, spec: IslandSpec, rng: np.random.Generator, progress: float, cfg: Config) -> np.ndarray:
    low, high = cfg.bounds.low(), cfg.bounds.high()
    sigma = (high - low) * spec.mutation_fraction * (1.0 - 0.60 * progress) * np.asarray(spec.gene_weights)
    child = vector.copy()
    changed = rng.random(N_GENES) < spec.mutation_probability
    child[changed] += rng.normal(0.0, sigma[changed])
    anneal = 1.0 - 0.60 * progress
    if spec.coherent_outer and rng.random() < 0.75:
        outer_mode = float(rng.normal(0.0, 5e-6 * anneal))
        child[3:6] += outer_mode
    if spec.coherent_inner and rng.random() < 0.75:
        inner_mode = float(rng.normal(0.0, 4e-6 * anneal))
        child[0:3] += inner_mode * np.asarray((1.0, 1.0, 0.75))
    return clip_vector(child, cfg.bounds)


def initial_vector(spec: IslandSpec, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    low, high = cfg.bounds.low(), cfg.bounds.high()
    sigma = (high - low) * spec.init_fraction * np.asarray(spec.gene_weights)
    return clip_vector(baseline_vector() + rng.normal(0.0, sigma), cfg.bounds)


def record(candidate: Candidate, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "origin": candidate.origin,
        "island_id": candidate.island_id,
        "generation": candidate.generation,
        "feasible": candidate.feasible,
        "violation": candidate.violation,
        "quality_default": quality(candidate),
        "rank": candidate.rank,
        "crowding": candidate.crowding,
        "vector": candidate.vector.tolist(),
        **{name: float(value) for name, value in zip(OBJECTIVE_NAMES, candidate.objectives)},
        **candidate.result,
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=True) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def update_archive(archive: list[Candidate], additions: list[Candidate]) -> list[Candidate]:
    all_items = archive + additions
    feasible = [item.copy() for item in all_items if item.feasible]
    finite = [item.copy() for item in all_items if np.all(np.isfinite(item.objectives))]
    pool = feasible if feasible else finite
    if not pool:
        return []
    fronts = rank_crowding(pool)
    chosen = [pool[index].copy() for index in fronts[0]] if fronts else []
    if len(chosen) > 512:
        chosen.sort(key=lambda item: (item.violation, quality(item), -item.crowding))
        chosen = chosen[:512]
    return chosen


def plot_evolution(history: list[dict[str, Any]], output: Path) -> None:
    generation = np.asarray([row["generation"] for row in history], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
    axes[0].plot(generation, [row["feasible_fraction"] for row in history], marker="o")
    axes[0].set_ylabel("feasible fraction")
    axes[1].plot(generation, [row["best_height_target_um"] for row in history], marker="o", color="tab:blue")
    axes[1].axhline(3.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("best fixed-target dz [um]")
    axes[2].plot(generation, [row["lowest_barrier_mev"] for row in history], marker="o", color="tab:purple", label="lowest barrier")
    axes[2].plot(generation, [row["best_height_barrier_mev"] for row in history], marker="s", color="tab:green", label="barrier at best height")
    axes[2].set_xlabel("generation")
    axes[2].set_ylabel("barrier [meV]")
    axes[2].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(archive: list[Candidate], output: Path) -> None:
    if not archive:
        return
    matrix = np.asarray([item.objectives for item in archive], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (left, right) in zip(axes, ((0, 1), (0, 2), (1, 4))):
        axis.scatter(matrix[:, left], matrix[:, right], s=32, alpha=0.82, color="crimson")
        axis.set_xlabel(OBJECTIVE_NAMES[left])
        axis.set_ylabel(OBJECTIVE_NAMES[right])
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_layout(model: Any, trace: Any, output: Path, title: str) -> None:
    panels = np.asarray(model.bem.panels_m, dtype=float)
    rf = np.asarray(model.bem.electrode_voltages_v > 0.5, dtype=float)
    patches = [Rectangle((panel[0] * 1e6, panel[2] * 1e6), (panel[1] - panel[0]) * 1e6, (panel[3] - panel[2]) * 1e6) for panel in panels]
    figure, axis = plt.subplots(figsize=(8.2, 8.0))
    collection = PatchCollection(patches, array=rf, cmap="RdYlBu_r", edgecolor=(0.2, 0.2, 0.2, 0.12), linewidth=0.15)
    collection.set_clim(0.0, 1.0)
    axis.add_collection(collection)
    converged = np.asarray(trace.converged, dtype=bool)
    x_m = np.asarray(trace.x_m, dtype=float)
    y_m = np.asarray(trace.y_m, dtype=float)
    if np.any(converged):
        axis.plot(x_m[converged] * 1e6, y_m[converged] * 1e6, color="black", linewidth=1.8, label="RF-null path")
    if np.any(~converged):
        axis.scatter(x_m[~converged] * 1e6, y_m[~converged] * 1e6, color="crimson", marker="x", label="trace failure")
    axis.set_aspect("equal")
    axis.set_xlim(-220, 220)
    axis.set_ylim(-220, 220)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_profiles(trace: Any, potential_ev: np.ndarray, barrier: Any, output: Path, title: str) -> None:
    x_um = np.asarray(trace.x_m, dtype=float) * 1e6
    converged = np.asarray(trace.converged, dtype=bool)
    z_error_um = (np.asarray(trace.z_m, dtype=float) - TARGET_ION_HEIGHT_M) * 1e6
    y_um = np.asarray(trace.y_m, dtype=float) * 1e6
    potential_mev = np.asarray(potential_ev, dtype=float) * 1e3
    residual = np.asarray(trace.transverse_residual_v_m, dtype=float)
    figure, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)
    axes[0].plot(x_um[converged], z_error_um[converged], color="tab:blue", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("z - target [um]")
    axes[0].set_title(title)
    axes[1].plot(x_um[converged], y_um[converged], color="tab:orange", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("y [um]")
    axes[2].plot(x_um[converged], potential_mev[converged], color="tab:green", linewidth=1.8, label="pseudopotential")
    axes[2].axhline(float(barrier.reference_energy_ev) * 1e3, color="black", linestyle="--", linewidth=0.8, label="reference")
    axes[2].axhline(float(barrier.maximum_energy_ev) * 1e3, color="crimson", linestyle=":", linewidth=1.0, label="maximum")
    axes[2].set_ylabel("Psi_ps [meV]")
    axes[2].legend(loc="best")
    axes[3].plot(x_um[converged], residual[converged], color="tab:purple", linewidth=1.5)
    axes[3].set_xlabel("x [um]")
    axes[3].set_ylabel("residual [V/m]")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_map_xy(model: Any, output: Path, title: str) -> None:
    extent_m = 150e-6
    count = 61
    coordinate = np.linspace(-extent_m, extent_m, count)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="xy")
    values = np.empty_like(x_grid)
    for row in range(count):
        for column in range(count):
            field = np.asarray(model.bem.electric_field(float(x_grid[row, column]), float(y_grid[row, column]), TARGET_ION_HEIGHT_M), dtype=float)
            values[row, column] = 0.5 * np.dot(field, field)
    finite = values[np.isfinite(values)]
    levels = np.linspace(np.percentile(finite, 5), np.percentile(finite, 95), 32)
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    contour = axis.contourf(x_grid * 1e6, y_grid * 1e6, values, levels=levels, cmap="viridis")
    figure.colorbar(contour, ax=axis, label="0.5 |E|^2 [arb.]")
    axis.set_aspect("equal")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title + "\nXY pseudopotential surrogate at target height")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def render_candidates(archive: list[Candidate], all_candidates: list[Candidate], cfg: Config) -> None:
    source = archive if archive else all_candidates
    if not source or cfg.render_top <= 0:
        return
    selected = sorted(source, key=lambda item: (item.violation, quality(item), item.result.get("height_peak_target_m", np.inf)))[:cfg.render_top]
    root = cfg.output_dir / "best_candidates"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(tqdm(selected, desc="Rendering traps", unit="candidate", dynamic_ncols=True), start=1):
        directory = root / f"rank_{rank:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        result, artifacts = solve_candidate(candidate.vector, cfg, diagnostics=True)
        payload = record(candidate, f"rank_{rank:02d}")
        payload["diagnostic_result"] = result
        if artifacts is None:
            payload["render_error"] = result["reason"]
        else:
            title = f"rank {rank:02d}: fixed-target dz={result['height_peak_target_m'] * 1e6:.3f} um; barrier={result['barrier_ev'] * 1e3:.3f} meV"
            plot_layout(artifacts["model"], artifacts["trace"], directory / "layout_and_trace.png", title)
            plot_profiles(artifacts["trace"], artifacts["potential_ev"], artifacts["barrier"], directory / "path_diagnostics.png", title)
            if cfg.render_maps:
                try:
                    plot_map_xy(artifacts["model"], directory / "pseudopotential_map_xy.png", title)
                except Exception as error:
                    payload["map_render_error"] = f"{type(error).__name__}: {error}"
        (directory / "candidate.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        rows.append(payload)
    write_csv(rows, root / "diagnostic_summary.csv")


def save_checkpoint(output_dir: Path, generation: int, islands: list[Island], archive: list[Candidate], history: list[dict[str, Any]], migrations: list[dict[str, Any]]) -> None:
    directory = output_dir / "checkpoints" / f"generation_{generation:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": generation,
        "history": history,
        "migrations": migrations,
        "archive": [record(item, "archive") for item in archive],
        "islands": [{"name": island.spec.name, "best_quality": island.best_quality, "stagnation": island.stagnation, "population": [record(item, island.spec.name) for item in island.population]} for island in islands],
    }
    (directory / "checkpoint.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.islands <= len(ISLAND_SPECS):
        raise ValueError(f"--islands must be in [1, {len(ISLAND_SPECS)}]")
    cfg = Config(
        islands=args.islands,
        population=args.population,
        offspring=args.offspring or args.population,
        generations=args.generations,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        path_points=args.path_points,
        fast=args.fast,
        output_dir=Path(args.output_dir),
        render_top=args.render_top,
        render_maps=args.render_maps,
    )
    seeds = seed_vectors()
    if cfg.population < len(seeds):
        raise ValueError(f"--population must be at least {len(seeds)} because all deterministic seeds are retained")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    specs = ISLAND_SPECS[:cfg.islands]
    master = np.random.default_rng(cfg.seed)

    print("Workflow 07 agent: corrected fixed-target-height connected C4v island GA")
    print("=" * 88)
    print(f"Target ion height : {TARGET_ION_HEIGHT_M * 1e6:.3f} um")
    print("Height metric     : max |z_RF_null - TARGET_ION_HEIGHT_M| over converged points")
    print(f"Islands           : {cfg.islands}")
    print(f"Population/island : {cfg.population}")
    print(f"Offspring/island  : {cfg.offspring}")
    print(f"Generations       : {cfg.generations}")
    print(f"Seeds/island      : {len(seeds)}")

    probe = make_candidate(baseline_vector(), -1, 0, "baseline_probe", cfg)
    print()
    print(
        "Baseline fixed-target probe: "
        f"dz={probe.result['height_peak_target_m'] * 1e6:.4f} um; "
        f"arm-ref dz={probe.result['height_peak_arm_referenced_m'] * 1e6:.4f} um; "
        f"barrier={probe.result['barrier_ev'] * 1e3:.4f} meV; "
        f"converged={probe.result['n_converged']}/{probe.result['n_points']}; "
        f"reason={probe.result['reason']}"
    )

    islands: list[Island] = []
    with tqdm(total=cfg.islands * cfg.population, desc="Initial population", unit="candidate", dynamic_ncols=True) as progress:
        for spec in specs:
            rng = np.random.default_rng(int(master.integers(0, np.iinfo(np.uint32).max)))
            population: list[Candidate] = []
            for label, vector in seeds:
                population.append(make_candidate(vector, spec.island_id, 0, label, cfg))
                progress.update(1)
            while len(population) < cfg.population:
                population.append(make_candidate(initial_vector(spec, rng, cfg), spec.island_id, 0, "random_initial", cfg))
                progress.update(1)
            rank_crowding(population)
            islands.append(Island(spec, rng, population, min(quality(item, spec.objective_weights) for item in population)))

    archive = update_archive([], [item for island in islands for item in island.population])
    history: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []

    def snapshot(generation: int) -> dict[str, Any]:
        population = [item for island in islands for item in island.population]
        feasible = [item for item in population if item.feasible]
        finite = [item for item in population if np.all(np.isfinite(item.objectives))]
        best_height = min(finite, key=lambda item: item.result["height_peak_target_m"]) if finite else None
        lowest_barrier = min(finite, key=lambda item: item.result["barrier_ev"]) if finite else None
        return {
            "generation": generation,
            "population": len(population),
            "feasible": len(feasible),
            "feasible_fraction": len(feasible) / len(population) if population else 0.0,
            "archive_size": len(archive),
            "best_height_target_um": float(best_height.result["height_peak_target_m"] * 1e6) if best_height else np.inf,
            "best_height_barrier_mev": float(best_height.result["barrier_ev"] * 1e3) if best_height else np.inf,
            "best_height_origin": best_height.origin if best_height else "none",
            "lowest_barrier_mev": float(lowest_barrier.result["barrier_ev"] * 1e3) if lowest_barrier else np.inf,
            "lowest_barrier_height_um": float(lowest_barrier.result["height_peak_target_m"] * 1e6) if lowest_barrier else np.inf,
            "best_violation": min((item.violation for item in population), default=np.inf),
        }

    history.append(snapshot(0))
    state = history[-1]
    print(f"Generation 0000 | feasible={state['feasible']}/{state['population']} | best_dz={state['best_height_target_um']:.3f} um ({state['best_height_origin']}) | lowest_barrier={state['lowest_barrier_mev']:.2f} meV")

    for generation in range(1, cfg.generations + 1):
        with tqdm(total=cfg.islands * cfg.offspring, desc=f"Generation {generation:04d}", unit="candidate", dynamic_ncols=True) as progress:
            for island in islands:
                rank_crowding(island.population)
                children: list[Candidate] = []
                while len(children) < cfg.offspring:
                    first = tournament(island.population, island.rng, island.spec.objective_weights)
                    if island.rng.random() < island.spec.crossover_probability:
                        second = tournament(island.population, island.rng, island.spec.objective_weights)
                        vector = crossover(first.vector, second.vector, island.rng, cfg)
                        origin = f"cross:{first.origin}+{second.origin}"
                    else:
                        vector = first.vector.copy()
                        origin = f"clone:{first.origin}"
                    vector = mutate(vector, island.spec, island.rng, generation / max(1, cfg.generations), cfg)
                    children.append(make_candidate(vector, island.spec.island_id, generation, origin, cfg))
                    progress.update(1)
                island.population = select_survivors(island.population + children, cfg.population)
                current = min(quality(item, island.spec.objective_weights) for item in island.population)
                if current + 1e-12 < island.best_quality:
                    island.best_quality = current
                    island.stagnation = 0
                else:
                    island.stagnation += 1

        if cfg.migration_interval > 0 and generation % cfg.migration_interval == 0 and len(islands) > 1:
            outgoing: list[list[Candidate]] = []
            for island in islands:
                rank_crowding(island.population)
                ordered = sorted(island.population, key=lambda item: (item.rank, -item.crowding, quality(item, island.spec.objective_weights)))
                outgoing.append([item.copy() for item in ordered[:cfg.migrants]])
            for source, migrants in enumerate(outgoing):
                destination = (source + 1) % len(islands)
                islands[destination].population = select_survivors(islands[destination].population + migrants, cfg.population)
                migrations.append({"generation": generation, "source": islands[source].spec.name, "destination": islands[destination].spec.name, "migrant_count": len(migrants)})

        archive = update_archive(archive, [item for island in islands for item in island.population])
        history.append(snapshot(generation))
        state = history[-1]
        print(f"Generation {generation:04d} | feasible={state['feasible']}/{state['population']} | best_dz={state['best_height_target_um']:.3f} um | barrier@bestdz={state['best_height_barrier_mev']:.2f} meV | lowest_barrier={state['lowest_barrier_mev']:.2f} meV @ dz={state['lowest_barrier_height_um']:.3f} um")
        if generation % cfg.checkpoint_interval == 0 or generation == cfg.generations:
            save_checkpoint(cfg.output_dir, generation, islands, archive, history, migrations)

    all_candidates = [item for island in islands for item in island.population]
    write_csv([record(item, island.spec.name) for island in islands for item in island.population], cfg.output_dir / "final_population.csv")
    write_csv([record(item, "archive") for item in archive], cfg.output_dir / "pareto_archive.csv")
    write_csv(history, cfg.output_dir / "history.csv")
    write_csv(migrations, cfg.output_dir / "migration_history.csv")
    metadata = {
        "config": asdict(cfg),
        "target_ion_height_m": TARGET_ION_HEIGHT_M,
        "height_definition": "max(abs(z_m[converged] - TARGET_ION_HEIGHT_M))",
        "objective_names": OBJECTIVE_NAMES,
        "contour_knots_m": CONTOUR_KNOTS_M.tolist(),
        "seeds": [{"label": label, "vector": vector.tolist()} for label, vector in seeds],
        "architecture": "existing workflow-07 house-style connected C4v contours; 11 genes; island NSGA-II; barrier-oriented islands",
    }
    (cfg.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    plot_evolution(history, cfg.output_dir / "evolution.png")
    plot_pareto(archive, cfg.output_dir / "pareto_projection.png")
    render_candidates(archive, all_candidates, cfg)

    print()
    print(f"Results written to: {cfg.output_dir}")
    print(f"Archive candidates: {len(archive)}")
    if archive:
        best = min(archive, key=lambda item: (item.violation, quality(item)))
        print(json.dumps(record(best, "best"), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
