"""Full-genome CUDA/CPU island NSGA-II for a C4v open-centre planar RF X-junction.

The full fixed-length genome includes both inner and outer Chebyshev-polynomial
edge coefficients (orders 0..8), effective active degree of each polynomial,
and junction/global geometry parameters. It deliberately does NOT insert a
central RF cross. The four C4v copies are produced by the project's existing
junction template.

Coarse search is exploratory and uses Pareto objectives:
  1) max |z_null - z_target|
  2) RF pseudopotential barrier
Only a complete, fine-mesh validation with dz < 3 um AND U < 1 meV is a hit.

A fixed BEM mesh is not shared between different genomes: each geometry gets
its own mesh, BEM assembly and factorization. Hence population size affects
wall time primarily through the number of new candidates, not VRAM linearly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from core.analysis.barrier import compute_barrier_metrics, pseudopotential_profile_ev
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem

# Boundary knots on a single fundamental C4v arm. Boundary offsets are evaluated
# from two Chebyshev polynomials on t in [-1, 1], then passed to the existing
# project template as its contour samples.
KNOTS_M = np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210], dtype=float) * 1e-6
N_POLY = 9                         # Chebyshev orders T_0 ... T_8
MAX_DEGREE = N_POLY - 1

DEG_IN = 0
DEG_OUT = 1
INNER_COEFF = slice(2, 2 + N_POLY)
OUTER_COEFF = slice(2 + N_POLY, 2 + 2 * N_POLY)
CENTER_IN = 2 + 2 * N_POLY
CENTER_OUT = CENTER_IN + 1
START = CENTER_IN + 2
LENGTH = CENTER_IN + 3
POWER = CENTER_IN + 4
BULGE = CENTER_IN + 5
BULGE_CENTER = CENTER_IN + 6
BULGE_WIDTH = CENTER_IN + 7
N_GENES = CENTER_IN + 8

# Coefficients are metres. Bounds intentionally allow strongly non-smooth
# discovery shapes; post-search manufacturing/robustness is a separate stage.
LOW = np.array(
    [0, 0]
    + [-45e-6] * N_POLY
    + [-45e-6] * N_POLY
    + [-55e-6, -10e-6, 4e-6, 50e-6, 0.60, -40e-6, 6e-6, 5e-6],
    dtype=float,
)
HIGH = np.array(
    [MAX_DEGREE, MAX_DEGREE]
    + [45e-6] * N_POLY
    + [45e-6] * N_POLY
    + [-2e-6, 75e-6, 65e-6, 340e-6, 5.75, 40e-6, 150e-6, 90e-6],
    dtype=float,
)


@dataclass(frozen=True)
class Settings:
    backend: str
    islands: int
    population: int
    offspring: int
    generations: int
    migration_interval: int
    migrants: int
    checkpoint_interval: int
    retain_per_island: int
    seed: int
    coarse_points: int
    fine_points: int
    fine_top: int
    max_panels: int
    fast: bool
    animation_stride: int
    output_dir: Path
    rf_peak_v: float = 100.0
    target_z_m: float = TARGET_ION_HEIGHT_M
    dz_limit_m: float = 3e-6
    barrier_limit_ev: float = 1e-3
    lateral_limit_m: float = 5e-6


@dataclass(frozen=True)
class IslandStyle:
    name: str
    weights: tuple[float, float]
    explore_scale: float
    final_scale: float
    emphasis: str


# More islands are useful because they represent distinct search policies, not
# merely the two physical polynomials. They coexist through ring migration.
STYLES = (
    IslandStyle("height_inner", (7.0, 1.0), 1.10, 0.10, "inner"),
    IslandStyle("barrier_outer", (1.0, 7.0), 1.10, 0.10, "outer"),
    IslandStyle("balanced_global", (2.0, 2.0), 1.00, 0.09, "global"),
    IslandStyle("central_geometry", (3.0, 2.0), 1.20, 0.10, "central"),
    IslandStyle("high_order_inner", (4.0, 2.0), 1.30, 0.12, "high_inner"),
    IslandStyle("high_order_outer", (2.0, 4.0), 1.30, 0.12, "high_outer"),
    IslandStyle("degree_explorer", (2.0, 2.0), 1.45, 0.14, "degree"),
    IslandStyle("wide_restart", (2.0, 2.0), 1.65, 0.16, "wide"),
    IslandStyle("joint_10x10", (2.0, 2.0), 1.05, 0.08, "joint"),
    IslandStyle("pareto_diverse", (2.0, 2.0), 1.25, 0.11, "diverse"),
    IslandStyle("local_polish", (2.0, 2.0), 0.55, 0.035, "local"),
    IslandStyle("taper_bulge", (2.0, 3.0), 1.20, 0.10, "global"),
)


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result["valid"])

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf], dtype=float)
        return np.array([self.result["dz_peak_m"] / 3e-6,
                         self.result["barrier_ev"] / 1e-3], dtype=float)

    def copy(self) -> "Individual":
        return Individual(self.genome.copy(), dict(self.result), self.island,
                          self.generation, self.origin, self.rank, self.crowding)


@dataclass
class Island:
    style: IslandStyle
    rng: np.random.Generator
    population: list[Individual]


class BEMField:
    def __init__(self, bem: FixedMeshBEM, charge: Any):
        self.bem = bem
        self.charge = charge

    def electric_field(self, x_m: float, y_m: float, z_m: float):
        values = self.bem.field_batch(np.array([x_m]), np.array([y_m]),
                                      np.array([z_m]), self.charge)
        return tuple(map(float, self.bem.asnumpy(values)[0, 0]))


def degree(x: float) -> int:
    return int(np.clip(np.rint(x), 0, MAX_DEGREE))


def active_coefficients(genome: np.ndarray, part: slice, active_degree: int) -> np.ndarray:
    values = np.asarray(genome[part], dtype=float).copy()
    values[active_degree + 1:] = 0.0
    return values


def chebyshev_profile(coefficients: np.ndarray, s_m: np.ndarray) -> np.ndarray:
    """Evaluate coefficient vector in Chebyshev basis on KNOTS_M domain."""
    denominator = float(KNOTS_M[-1] - KNOTS_M[0])
    coordinate = 2.0 * (np.asarray(s_m, dtype=float) - KNOTS_M[0]) / denominator - 1.0
    coordinate = np.clip(coordinate, -1.0, 1.0)
    return np.polynomial.chebyshev.chebval(coordinate, coefficients)


def repair(genome: np.ndarray) -> np.ndarray:
    """Fixed-size genome repair. It does not smooth the produced contour.

    Coefficients remain free. The later geometric checks reject crossing rails,
    clearance violations and invalid template geometry.
    """
    x = np.asarray(genome, dtype=float).copy()
    if x.shape != (N_GENES,) or not np.all(np.isfinite(x)):
        raise ValueError(f"genome must contain exactly {N_GENES} finite values")
    x = np.clip(x, LOW, HIGH)
    x[DEG_IN] = degree(x[DEG_IN])
    x[DEG_OUT] = degree(x[DEG_OUT])
    return x


def baseline() -> np.ndarray:
    x = np.zeros(N_GENES, dtype=float)
    x[DEG_IN], x[DEG_OUT] = 2, 2
    x[CENTER_IN:] = [-25e-6, 20e-6, 30e-6, 150e-6, 2.0, 0.0, 70e-6, 25e-6]
    return x


def seed_genomes() -> list[tuple[str, np.ndarray]]:
    base = baseline()
    h = base.copy()
    h[INNER_COEFF][1:4] = [-8e-6, -13e-6, -8e-6]
    h[DEG_IN] = 3
    b = base.copy()
    b[OUTER_COEFF][1:4] = [7e-6, 12e-6, 8e-6]
    b[DEG_OUT] = 3
    coupled = base.copy()
    coupled[DEG_IN], coupled[DEG_OUT] = 5, 5
    coupled[INNER_COEFF][:6] = [-3e-6, -9e-6, -15e-6, -10e-6, -5e-6, -2e-6]
    coupled[OUTER_COEFF][:6] = [2e-6, 6e-6, 12e-6, 9e-6, 4e-6, 2e-6]
    near = base.copy()
    near[START], near[LENGTH], near[POWER] = 18e-6, 125e-6, 2.5
    return [("baseline", base), ("inner_seed", h), ("outer_seed", b),
            ("coupled_poly_seed", coupled), ("near_center_seed", near)]


def contour_offsets(genome: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    din, dout = degree(genome[DEG_IN]), degree(genome[DEG_OUT])
    inner = chebyshev_profile(active_coefficients(genome, INNER_COEFF, din), KNOTS_M)
    outer = chebyshev_profile(active_coefficients(genome, OUTER_COEFF, dout), KNOTS_M)
    # Template joins to its external arm at the final knot; fixed zero prevents
    # a discontinuity there. Central behavior is separately controlled by genes.
    inner[-1] = 0.0
    outer[-1] = 0.0
    return inner, outer, din, dout


def make_parameters(genome: np.ndarray):
    inner, outer, _, _ = contour_offsets(genome)
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        inner_edge_shift_at_centre_m=float(genome[CENTER_IN]),
        outer_edge_shift_at_centre_m=float(genome[CENTER_OUT]),
        rf_start_radius_override_m=float(genome[START]),
        taper_length_m=float(genome[LENGTH]),
        taper_power=float(genome[POWER]),
        outer_bulge_amplitude_m=float(genome[BULGE]),
        outer_bulge_center_m=float(genome[BULGE_CENTER]),
        outer_bulge_sigma_m=float(genome[BULGE_WIDTH]),
        inner_contour_knots_m=tuple(KNOTS_M),
        inner_contour_offsets_m=tuple(inner),
        outer_contour_knots_m=tuple(KNOTS_M),
        outer_contour_offsets_m=tuple(outer),
    )


def rejected(reason: str, stage: str) -> dict[str, Any]:
    return dict(valid=False, verified=False, reason=reason, stage=stage,
                dz_peak_m=np.inf, dz_rms_m=np.inf, barrier_ev=np.inf,
                lateral_peak_m=np.inf, panels=0, points=0, converged=0)


def build_mesh(parameters: Any, fast: bool):
    if fast:
        return build_geometry_aware_quadtree_x_junction_bem(
            parameters, central_half_extent_m=145e-6, central_max_cell_m=46e-6,
            boundary_max_cell_m=17e-6, outer_max_cell_m=180e-6, min_cell_m=8e-6)
    return build_geometry_aware_quadtree_x_junction_bem(
        parameters, central_half_extent_m=180e-6, central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6, outer_max_cell_m=180e-6, min_cell_m=5e-6)


def evaluate(genome: np.ndarray, cfg: Settings, stage="coarse", retain=False):
    x = repair(genome)
    try:
        p = make_parameters(x)
        report = check_x_junction_manufacturability(p)
        if not report.valid:
            return rejected("geometry: " + "; ".join(map(str, report.messages)), stage), None
        s = np.linspace(0, p.arm_length_m, 2001)
        r_inner, r_outer = p.rail_boundaries_m(s)
        if np.min(r_inner) <= 0.5e-6:
            return rejected("inner RF edge enters central clearance", stage), None
        if np.min(r_outer-r_inner) < 18e-6:
            return rejected("RF rail width < 18 um", stage), None
        model = build_mesh(p, cfg.fast and stage == "coarse")
        if model.n_panels > cfg.max_panels:
            return rejected(f"panel limit {model.n_panels}>{cfg.max_panels}", stage), None
        bem = FixedMeshBEM(model.bem.panels_m, backend=cfg.backend)
        bem.assemble(block_rows=64)
        bem.factorize()
        bem.release_matrix()
        charge = bem.solve_masks(model.bem.electrode_voltages_v)
        field = BEMField(bem, charge)
        points = cfg.fine_points if stage == "fine" else cfg.coarse_points
        xs = np.linspace(-350e-6, 350e-6, points)
        trace = trace_rf_transverse_minimum(
            field, xs, initial_y_m=0.0, initial_z_m=cfg.target_z_m,
            residual_tolerance_v_m=1e-3, max_transverse_shift_m=25e-6)
        converged = np.asarray(trace.converged, dtype=bool)
        z, y = np.asarray(trace.z_m, float), np.asarray(trace.y_m, float)
        if not trace.valid or converged.shape != (points,) or not np.all(converged):
            return rejected(f"incomplete trace {int(np.sum(converged))}/{points}", stage), None
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            return rejected("nonfinite RF-null trace", stage), None
        pseudo = np.asarray(pseudopotential_profile_ev(
            field, trace, rf_voltage_peak_v=cfg.rf_peak_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S), dtype=float)
        barrier = compute_barrier_metrics(pseudo, trace)
        if not barrier.valid or pseudo.shape != (points,) or not np.all(np.isfinite(pseudo)):
            return rejected("invalid RF pseudopotential profile", stage), None
        dz = z-cfg.target_z_m
        dz_peak = float(np.max(np.abs(dz)))
        dz_rms = float(np.sqrt(np.mean(dz**2)))
        lateral = float(np.max(np.abs(y)))
        barrier_ev = float(barrier.barrier_height_ev)
        if not np.all(np.isfinite([dz_peak, dz_rms, lateral, barrier_ev])):
            return rejected("nonfinite physical metrics", stage), None
        return dict(
            valid=True,
            verified=bool(stage == "fine" and dz_peak < cfg.dz_limit_m
                          and barrier_ev < cfg.barrier_limit_ev
                          and lateral < cfg.lateral_limit_m),
            reason="ok", stage=stage, backend=bem.backend_name,
            dz_peak_m=dz_peak, dz_rms_m=dz_rms, barrier_ev=barrier_ev,
            lateral_peak_m=lateral, barrier_reference_ev=float(barrier.reference_energy_ev),
            panels=int(model.n_panels), points=points, converged=int(np.sum(converged)),
        ), (model, trace, pseudo) if retain else None
    except Exception as error:
        if cfg.backend == "cuda" and "out of memory" in str(error).lower():
            raise RuntimeError("CUDA OOM: reduce --max-panels or use --fast") from error
        return rejected(f"{type(error).__name__}: {error}", stage), None


def dominates(a: Individual, b: Individual) -> bool:
    if a.valid != b.valid:
        return a.valid
    if not a.valid:
        return False
    return bool(np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives))


def rank_and_crowding(population: list[Individual]) -> list[list[int]]:
    n = len(population)
    dominates_list = [[] for _ in population]
    domination_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(i+1, n):
            if dominates(population[i], population[j]):
                dominates_list[i].append(j); domination_count[j] += 1
            elif dominates(population[j], population[i]):
                dominates_list[j].append(i); domination_count[i] += 1
    fronts[0] = [i for i in range(n) if domination_count[i] == 0]
    while fronts[-1]:
        following: list[int] = []
        for i in fronts[-1]:
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    following.append(j)
        fronts.append(following)
    fronts.pop()
    for level, front in enumerate(fronts):
        for i in front:
            population[i].rank, population[i].crowding = level, 0.0
        valid = [i for i in front if population[i].valid]
        if len(valid) <= 2:
            for i in valid:
                population[i].crowding = np.inf
            continue
        for axis in range(2):
            order = sorted(valid, key=lambda i: population[i].objectives[axis])
            lower = population[order[0]].objectives[axis]
            upper = population[order[-1]].objectives[axis]
            if upper <= lower:
                continue
            population[order[0]].crowding = np.inf
            population[order[-1]].crowding = np.inf
            for k in range(1, len(order)-1):
                value = population[order[k]]
                if np.isfinite(value.crowding):
                    value.crowding += ((population[order[k+1]].objectives[axis]
                                       - population[order[k-1]].objectives[axis]) / (upper-lower))
    return fronts


def survivors(pool: list[Individual], size: int) -> list[Individual]:
    result: list[Individual] = []
    for front in rank_and_crowding(pool):
        front.sort(key=lambda i: pool[i].crowding, reverse=True)
        result.extend(pool[i].copy() for i in front[:size-len(result)])
        if len(result) == size:
            break
    return result


def scalar_score(item: Individual, style: IslandStyle) -> float:
    if not item.valid:
        return np.inf
    if style.emphasis == "joint":
        return max(item.result["dz_peak_m"]/10e-6, item.result["barrier_ev"]/10e-3)
    return float(np.dot(np.asarray(style.weights), item.objectives))


def tournament(population: list[Individual], style: IslandStyle, rng: np.random.Generator) -> Individual:
    a, b = (population[int(i)] for i in rng.integers(0, len(population), 2))
    if rng.random() < .50:
        return a if scalar_score(a, style) < scalar_score(b, style) else b
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.crowding != b.crowding:
        return a if a.crowding > b.crowding else b
    return a if scalar_score(a, style) < scalar_score(b, style) else b


def anneal_scale(style: IslandStyle, fraction: float) -> float:
    """High mutation for first 75%, cosine cool-down in final 25%."""
    hold = .75
    if fraction <= hold:
        return style.explore_scale
    q = (fraction-hold)/(1.0-hold)
    return style.final_scale + .5*(style.explore_scale-style.final_scale)*(1.0+math.cos(math.pi*q))


def mutation_sigmas(style: IslandStyle, fraction: float) -> np.ndarray:
    scale = anneal_scale(style, fraction)
    sigma = np.zeros(N_GENES, dtype=float)
    sigma[DEG_IN:DEG_OUT+1] = .70 * scale
    # Chebyshev low order controls broad geometry; high order controls detail.
    base = np.array([20, 20, 16, 14, 11, 9, 7, 6, 5], dtype=float) * 1e-6 * scale
    sigma[INNER_COEFF] = base
    sigma[OUTER_COEFF] = base
    sigma[CENTER_IN:CENTER_OUT+1] = 14e-6 * scale
    sigma[START] = 16e-6 * scale
    sigma[LENGTH] = 42e-6 * scale
    sigma[POWER] = .55 * scale
    sigma[BULGE] = 16e-6 * scale
    sigma[BULGE_CENTER] = 28e-6 * scale
    sigma[BULGE_WIDTH] = 18e-6 * scale
    if style.emphasis == "inner":
        sigma[INNER_COEFF] *= 1.8
    elif style.emphasis == "outer":
        sigma[OUTER_COEFF] *= 1.8
    elif style.emphasis == "high_inner":
        sigma[INNER_COEFF.start+4:INNER_COEFF.stop] *= 2.5
    elif style.emphasis == "high_outer":
        sigma[OUTER_COEFF.start+4:OUTER_COEFF.stop] *= 2.5
    elif style.emphasis == "central":
        sigma[CENTER_IN:START+1] *= 2.2
    elif style.emphasis == "degree":
        sigma[DEG_IN:DEG_OUT+1] *= 3.0
    elif style.emphasis == "wide":
        sigma *= 1.7
    elif style.emphasis == "local":
        sigma *= .55
    return sigma


def mutate(parent: np.ndarray, style: IslandStyle, rng: np.random.Generator, fraction: float) -> np.ndarray:
    x = parent.copy()
    sigma = mutation_sigmas(style, fraction)
    active_probability = np.full(N_GENES, .58)
    active_probability[DEG_IN:DEG_OUT+1] = .16 if fraction < .75 else .035
    if style.emphasis == "degree":
        active_probability[DEG_IN:DEG_OUT+1] = .48 if fraction < .75 else .08
    if style.emphasis == "inner":
        active_probability[INNER_COEFF] = .90
    elif style.emphasis == "outer":
        active_probability[OUTER_COEFF] = .90
    elif style.emphasis == "high_inner":
        active_probability[INNER_COEFF.start+4:INNER_COEFF.stop] = .95
    elif style.emphasis == "high_outer":
        active_probability[OUTER_COEFF.start+4:OUTER_COEFF.stop] = .95
    active = rng.random(N_GENES) < active_probability
    x[active] += rng.normal(0.0, sigma[active])
    # Discrete degree jumps are meaningful and less sensitive than Gaussian noise.
    for index in (DEG_IN, DEG_OUT):
        if active[index] and rng.random() < .85:
            x[index] = degree(parent[index]) + int(rng.choice((-2, -1, 1, 2)))
    return repair(x)


def serialize(individual: Individual) -> dict[str, Any]:
    inner, outer, din, dout = contour_offsets(individual.genome)
    return dict(
        island=individual.island, generation=individual.generation, origin=individual.origin,
        rank=individual.rank, crowding=individual.crowding,
        genome=individual.genome.tolist(), degree_inner=din, degree_outer=dout,
        inner_offsets_at_knots_m=inner.tolist(), outer_offsets_at_knots_m=outer.tolist(),
        **individual.result,
    )


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def per_island_top(island: Island, count: int) -> list[Individual]:
    rank_and_crowding(island.population)
    return [i.copy() for i in sorted(island.population,
            key=lambda i: (i.rank, -i.crowding, scalar_score(i, island.style)))[:count]]


def write_generation_snapshot(islands: list[Island], generation: int, cfg: Settings) -> None:
    rows: list[dict[str, Any]] = []
    for island in islands:
        rows.extend(serialize(i) for i in per_island_top(island, cfg.retain_per_island))
    save_csv(rows, cfg.output_dir / "island_elites" / f"elites_gen_{generation:04d}.csv")


def profile_plot(data: tuple[Any, Any, np.ndarray], result: dict[str, Any], directory: Path, name: str) -> None:
    model, trace, potential = data
    panels = np.asarray(model.bem.panels_m)
    rf = np.asarray(model.bem.electrode_voltages_v) > .5
    rectangles = [Rectangle((p[0]*1e6, p[2]*1e6), (p[1]-p[0])*1e6,
                            (p[3]-p[2])*1e6) for p in panels]
    fig, axis = plt.subplots(figsize=(7, 7))
    collection = PatchCollection(rectangles, array=rf.astype(float), cmap="RdYlBu_r")
    axis.add_collection(collection)
    axis.plot(np.asarray(trace.x_m)*1e6, np.asarray(trace.y_m)*1e6, "k-", lw=1.3)
    axis.set(xlim=(-230, 230), ylim=(-230, 230), xlabel="x [um]", ylabel="y [um]",
             title=name)
    axis.set_aspect("equal")
    fig.colorbar(collection, ax=axis)
    fig.tight_layout(); fig.savefig(directory / "layout.png", dpi=175); plt.close(fig)
    xx = np.asarray(trace.x_m)*1e6
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(xx, (np.asarray(trace.z_m)-TARGET_ION_HEIGHT_M)*1e6)
    axes[0].axhline(3.0, color="r", ls="--"); axes[0].axhline(-3.0, color="r", ls="--")
    axes[0].set_ylabel("z-target [um]")
    axes[1].plot(xx, (potential-result["barrier_reference_ev"])*1e3)
    axes[1].axhline(1.0, color="r", ls="--"); axes[1].set_ylabel("RF pseudo - ref [meV]")
    axes[2].plot(xx, np.asarray(trace.y_m)*1e6)
    axes[2].set_xlabel("x [um]"); axes[2].set_ylabel("y [um]")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(directory / "profiles.png", dpi=175); plt.close(fig)


def animation_from_elites(cfg: Settings) -> None:
    """Create one GIF for each island from retained top-5 generation CSVs.

    This is intentionally post-run and cheap: no BEM reruns. It animates the
    recorded Pareto metrics of retained individuals, ending at each island's
    best retained state. Pillow is optional; missing Pillow skips GIFs safely.
    """
    files = sorted((cfg.output_dir / "island_elites").glob("elites_gen_*.csv"))
    if not files:
        return
    try:
        import pandas as pd
    except ImportError:
        return
    frames: dict[int, list[dict[str, float]]] = {i: [] for i in range(cfg.islands)}
    for file in files[::max(1, cfg.animation_stride)]:
        frame = pd.read_csv(file)
        generation = int(file.stem.rsplit("_", 1)[-1])
        for island in range(cfg.islands):
            block = frame[(frame["island"] == island) & (frame["valid"] == True)]
            if block.empty:
                continue
            best = block.sort_values(["rank", "dz_peak_m", "barrier_ev"]).iloc[0]
            frames[island].append(dict(generation=generation,
                                       dz=float(best["dz_peak_m"])*1e6,
                                       barrier=float(best["barrier_ev"])*1e3))
    target = cfg.output_dir / "animations"
    target.mkdir(exist_ok=True)
    for island, values in frames.items():
        if len(values) < 2:
            continue
        x = np.array([v["dz"] for v in values])
        y = np.array([v["barrier"] for v in values])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set(xlabel="peak dz [um]", ylabel="RF barrier [meV]",
               title=f"Island {island}: retained elite evolution")
        ax.grid(alpha=.3)
        ax.set_xlim(max(0, np.min(x)*.85), np.max(x)*1.10)
        ax.set_ylim(max(0, np.min(y)*.85), np.max(y)*1.10)
        point, = ax.plot([], [], "ro", ms=8)
        path, = ax.plot([], [], "b-", lw=1.5, alpha=.7)
        label = ax.text(.03, .97, "", transform=ax.transAxes, va="top")
        def update(frame_index: int, point=point, path=path, label=label, ax=ax):
            point.set_data([x[frame_index]], [y[frame_index]])
            path.set_data(x[:frame_index+1], y[:frame_index+1])
            label.set_text(f"generation {values[frame_index]['generation']}\n"
                           f"dz={x[frame_index]:.3f} um\nU={y[frame_index]:.3f} meV")
            return point, path, label
        animation = FuncAnimation(fig, update, frames=len(values), interval=350, blit=False)
        try:
            animation.save(target / f"island_{island:02d}_elite_evolution.gif",
                           writer=PillowWriter(fps=3), dpi=120)
        except Exception as error:
            print(f"Animation skipped for island {island}: {type(error).__name__}: {error}")
        plt.close(fig)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--islands", type=int, default=12)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--offspring", type=int, default=100)
    p.add_argument("--generations", type=int, default=18)
    p.add_argument("--migration-interval", type=int, default=3)
    p.add_argument("--migrants", type=int, default=5)
    p.add_argument("--checkpoint-interval", type=int, default=1)
    p.add_argument("--retain-per-island", type=int, default=5)
    p.add_argument("--coarse-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=181)
    p.add_argument("--fine-top", type=int, default=25)
    p.add_argument("--max-panels", type=int, default=7000)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--animation-stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--output-dir", type=Path, default=Path("reports/08_c4v_full_polynomial"))
    return p


def main() -> None:
    args = parser().parse_args()
    backend = available_backend(args.backend)
    if not backend.available:
        raise RuntimeError(backend.reason)
    if not 1 <= args.islands <= len(STYLES):
        raise ValueError(f"--islands must be 1..{len(STYLES)}")
    if args.population < 8 or args.offspring < 1 or args.generations < 1:
        raise ValueError("population >= 8, offspring >= 1, generations >= 1")
    cfg = Settings(backend.selected, args.islands, args.population, args.offspring,
                   args.generations, args.migration_interval, args.migrants,
                   args.checkpoint_interval, args.retain_per_island, args.seed,
                   args.coarse_points, args.fine_points, args.fine_top,
                   args.max_panels, args.fast, args.animation_stride, args.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "settings.json").write_text(json.dumps(dict(
        settings=asdict(cfg), backend=backend.as_dict(), knots_m=KNOTS_M.tolist(),
        polynomial_basis="Chebyshev T_0..T_8 on [-1,1]", max_degree=MAX_DEGREE,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S), indent=2, default=str),
        encoding="utf-8")
    print("Full polynomial C4v island GA | backend:", backend.selected,
          "| device:", backend.device_name, flush=True)
    print(f"Genome: {N_GENES} genes: degree_in/out, 9+9 Chebyshev coefficients, 8 globals.")
    print("Mutation: high through 75% of generations; cosine annealing in final 25%.")
    print("Fine-only certification: dz < 3 um AND RF barrier < 1 meV.")

    master_rng = np.random.default_rng(cfg.seed)
    seed_list = seed_genomes()
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    best_z: tuple[np.ndarray, dict[str, Any]] | None = None
    best_u: tuple[np.ndarray, dict[str, Any]] | None = None
    best_joint: tuple[np.ndarray, dict[str, Any]] | None = None

    def current_joint_score(result: dict[str, Any]) -> float:
        return max(result["dz_peak_m"]/10e-6, result["barrier_ev"]/10e-3)

    def update_live(x: np.ndarray, result: dict[str, Any]) -> bool:
        nonlocal best_z, best_u, best_joint
        if not result["valid"]:
            return False
        changed = False
        if best_z is None or result["dz_peak_m"] < best_z[1]["dz_peak_m"]:
            best_z = (x.copy(), result.copy()); changed = True
        if best_u is None or result["barrier_ev"] < best_u[1]["barrier_ev"]:
            best_u = (x.copy(), result.copy()); changed = True
        if best_joint is None or current_joint_score(result) < current_joint_score(best_joint[1]):
            best_joint = (x.copy(), result.copy()); changed = True
        if changed and best_z is not None and best_u is not None and best_joint is not None:
            payload = dict(
                best_dz=dict(genome=best_z[0].tolist(), **best_z[1]),
                best_barrier=dict(genome=best_u[0].tolist(), **best_u[1]),
                best_10x10_compromise=dict(genome=best_joint[0].tolist(), **best_joint[1]),
            )
            (cfg.output_dir / "best_live_coarse.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return changed

    def set_postfix(bar: tqdm) -> None:
        if best_z is None or best_u is None or best_joint is None:
            bar.set_postfix_str("waiting for valid trace", refresh=False)
            return
        bar.set_postfix(dict(
            dz=f'{best_z[1]["dz_peak_m"]*1e6:.2f}um/U{best_z[1]["barrier_ev"]*1e3:.1f}',
            U=f'{best_u[1]["barrier_ev"]*1e3:.2f}meV/dz{best_u[1]["dz_peak_m"]*1e6:.1f}',
            joint=f'dz{best_joint[1]["dz_peak_m"]*1e6:.2f}/U{best_joint[1]["barrier_ev"]*1e3:.2f}',
        ), refresh=False)

    def coarse(x: np.ndarray, bar: tqdm | None = None) -> dict[str, Any]:
        x = repair(x)
        key = tuple(np.round(x, 12))
        if key not in cache:
            cache[key], _ = evaluate(x, cfg, stage="coarse")
        result = dict(cache[key])
        update_live(x, result)
        if bar is not None:
            set_postfix(bar)
        return result

    islands: list[Island] = []
    with tqdm(total=cfg.islands*cfg.population, desc="Initial BEM", unit="candidate") as bar:
        for island_id, style in enumerate(STYLES[:cfg.islands]):
            rng = np.random.default_rng(int(master_rng.integers(0, 2**32-1)))
            population: list[Individual] = []
            for name, x in seed_list:
                if len(population) == cfg.population:
                    break
                x = repair(x)
                population.append(Individual(x, coarse(x, bar), island_id, 0, name))
                bar.update()
            while len(population) < cfg.population:
                seed = seed_list[int(rng.integers(len(seed_list)))][1]
                perturb = rng.normal(0.0, (HIGH-LOW) * .18 * style.explore_scale)
                x = repair(seed + perturb)
                population.append(Individual(x, coarse(x, bar), island_id, 0, "initial"))
                bar.update()
            rank_and_crowding(population)
            islands.append(Island(style, rng, population))

    initial = [candidate for island in islands for candidate in island.population]
    if not any(candidate.valid for candidate in initial):
        failures = sorted({candidate.result["reason"] for candidate in initial})
        raise RuntimeError("No valid initial candidate: " + repr(failures[:20]))

    archive: list[Individual] = []
    history: list[dict[str, Any]] = []

    def update_archive(candidates: list[Individual]) -> None:
        nonlocal archive
        unique = {tuple(np.round(item.genome, 12)): item.copy()
                  for item in archive+candidates if item.valid}
        pool = list(unique.values())
        if not pool:
            archive = []
            return
        archive = [pool[i].copy() for i in rank_and_crowding(pool)[0]]
        archive.sort(key=lambda item: -item.crowding)
        archive = archive[:1000]

    def status(generation: int) -> dict[str, Any]:
        valid = [candidate for island in islands for candidate in island.population if candidate.valid]
        z = min(valid, key=lambda item: item.result["dz_peak_m"])
        u = min(valid, key=lambda item: item.result["barrier_ev"])
        j = min(valid, key=lambda item: current_joint_score(item.result))
        return dict(
            generation=generation, valid=len(valid), archive=len(archive),
            best_dz_um=z.result["dz_peak_m"]*1e6,
            U_at_best_dz_mev=z.result["barrier_ev"]*1e3,
            best_U_mev=u.result["barrier_ev"]*1e3,
            dz_at_best_U_um=u.result["dz_peak_m"]*1e6,
            compromise_dz_um=j.result["dz_peak_m"]*1e6,
            compromise_U_mev=j.result["barrier_ev"]*1e3,
            coarse_joint_hits=sum(item.result["dz_peak_m"] < cfg.dz_limit_m and
                                  item.result["barrier_ev"] < cfg.barrier_limit_ev for item in valid),
            unique_evaluations=len(cache),
        )

    update_archive(initial)
    history.append(status(0))
    write_generation_snapshot(islands, 0, cfg)
    print("G000", history[-1], flush=True)

    for generation in range(1, cfg.generations+1):
        next_islands: list[Island] = []
        with tqdm(total=cfg.islands*cfg.offspring, desc=f"Generation {generation}", unit="candidate") as bar:
            for island_id, island in enumerate(islands):
                rank_and_crowding(island.population)
                children: list[Individual] = []
                while len(children) < cfg.offspring:
                    left = tournament(island.population, island.style, island.rng)
                    if island.rng.random() < .90:
                        right = tournament(island.population, island.style, island.rng)
                        alpha = island.rng.uniform(-.18, 1.18, N_GENES)
                        child = repair(alpha*left.genome + (1.0-alpha)*right.genome)
                        origin = "crossover"
                    else:
                        child = left.genome.copy()
                        origin = "clone"
                    child = mutate(child, island.style, island.rng, generation/cfg.generations)
                    children.append(Individual(child, coarse(child, bar), island_id, generation, origin))
                    bar.update()
                next_islands.append(Island(island.style, island.rng,
                                           survivors(island.population+children, cfg.population)))
        islands = next_islands
        if cfg.migration_interval > 0 and generation % cfg.migration_interval == 0 and len(islands) > 1:
            outgoing = []
            for island in islands:
                rank_and_crowding(island.population)
                outgoing.append([item.copy() for item in sorted(island.population,
                    key=lambda item: (item.rank, -item.crowding,
                                      scalar_score(item, island.style)))[:cfg.migrants]])
            for source, migrants in enumerate(outgoing):
                destination = (source+1) % len(islands)
                target = islands[destination]
                islands[destination] = Island(target.style, target.rng,
                    survivors(target.population+migrants, cfg.population))
        update_archive([candidate for island in islands for candidate in island.population])
        history.append(status(generation))
        print(f"G{generation:03d}", history[-1], flush=True)
        save_csv(history, cfg.output_dir / "history.csv")
        save_csv([serialize(item) for item in archive], cfg.output_dir / "pareto_coarse.csv")
        if generation % cfg.checkpoint_interval == 0 or generation == cfg.generations:
            write_generation_snapshot(islands, generation, cfg)
            save_csv([serialize(item) for island in islands for item in island.population],
                     cfg.output_dir / f"population_gen_{generation:04d}.csv")

    animation_from_elites(cfg)
    if not archive:
        print("No Pareto archive candidates; inspect island_elites and checkpoint files.")
        return

    # Fine candidates: endpoints plus intermediate compromise points. Deduplicate
    # by the actual fixed-length genome before expensive fine BEM reruns.
    selected = sorted(archive, key=lambda item: item.result["dz_peak_m"])[:cfg.fine_top]
    selected += sorted(archive, key=lambda item: item.result["barrier_ev"])[:cfg.fine_top]
    selected += sorted(archive, key=lambda item: current_joint_score(item.result))[:cfg.fine_top]
    selected += sorted(archive, key=lambda item: max(item.objectives))[:cfg.fine_top]
    unique = {tuple(np.round(item.genome, 12)): item for item in selected}

    fine_rows: list[dict[str, Any]] = []
    for number, item in enumerate(tqdm(unique.values(), desc="Fine validation", unit="candidate"), 1):
        result, data = evaluate(item.genome, cfg, stage="fine", retain=True)
        row = dict(candidate=number, **serialize(item), fine_result=result)
        # flattened principal columns make the CSV convenient for sorting
        row.update({"fine_"+key: value for key, value in result.items()})
        fine_rows.append(row)
        if data is not None:
            directory = cfg.output_dir / "fine" / f"candidate_{number:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            profile_plot(data, result, directory, f"Fine candidate {number}")
            (directory / "full_candidate.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    save_csv(fine_rows, cfg.output_dir / "fine_validation.csv")

    fine_valid = [row for row in fine_rows if row["fine_valid"]]
    if not fine_valid:
        top5: list[dict[str, Any]] = []
    else:
        # Prefer true strict hits. If none, retain the five closest fine candidates
        # to the requested 3 um / 1 meV corner for the next design cycle.
        strict = [row for row in fine_valid if row["fine_verified"]]
        candidates = strict if strict else fine_valid
        top5 = sorted(candidates, key=lambda row: max(row["fine_dz_peak_m"]/3e-6,
                                                       row["fine_barrier_ev"]/1e-3))[:5]
    (cfg.output_dir / "top5_lithography_candidates.json").write_text(
        json.dumps(top5, indent=2), encoding="utf-8")
    save_csv(top5, cfg.output_dir / "top5_lithography_candidates.csv")
    verified = [row for row in fine_rows if row["fine_verified"]]
    (cfg.output_dir / "verified_hits.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(f"Fine verified hits: {len(verified)} | top-5 full design records: "
          f"{cfg.output_dir / 'top5_lithography_candidates.json'}")


if __name__ == "__main__":
    main()
