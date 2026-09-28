"""Open-centre C4v X-junction: CUDA-capable, target-directed island NSGA-II.

Put this file at workflows/07_targeted_c4v_island_ga_v2.py.

No central RF cross is added. Four existing RF regions receive the same RF
boundary voltage in the BEM. Their physical electrical connections outside
this finite mask are NOT designed or validated by this workflow.

Coarse results are discoveries, not certifications. Fine replay certifies
only complete traces with max|z-z_target| < 3 um and RF barrier < 1 meV,
with the stated 100 V peak RF drive and project RF angular frequency.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.targets import TARGET_ION_HEIGHT_M, RF_ANGULAR_FREQUENCY_RAD_S
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.barrier import pseudopotential_profile_ev, compute_barrier_metrics

# 10 active values per contour; end values at 0 and 210 um stay zero.
KNOTS_M = np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 210], dtype=float) * 1e-6
NC = len(KNOTS_M) - 2
INNER = slice(0, NC)
OUTER = slice(NC, 2 * NC)
CI, CO, LENGTH, POWER, START, BULGE, CENTER, SIGMA = range(2 * NC, 2 * NC + 8)
NGENES = 2 * NC + 8
LO = np.array([-30e-6] * NC + [-30e-6] * NC +
              [-45e-6, -5e-6, 55e-6, .7, 5e-6, -30e-6, 10e-6, 8e-6])
HI = np.array([30e-6] * NC + [30e-6] * NC +
              [-5e-6, 55e-6, 300e-6, 5.0, 50e-6, 30e-6, 130e-6, 75e-6])


@dataclass(frozen=True)
class Config:
    backend: str
    islands: int
    population: int
    offspring: int
    generations: int
    migration_interval: int
    migrants: int
    checkpoint_interval: int
    seed: int
    fast: bool
    path_points: int
    fine_points: int
    verify_top: int
    max_panels: int
    out: Path
    target_height_m: float = TARGET_ION_HEIGHT_M
    rf_peak_v: float = 100.0
    height_limit_m: float = 3e-6
    barrier_limit_ev: float = 1e-3
    lateral_limit_m: float = 5e-6


@dataclass(frozen=True)
class Style:
    name: str
    weights: tuple[float, float]
    sigma: float
    init_sigma: float
    focus: str


STYLES = (
    Style("height_escape", (5, 1), .20, .38, "inner"),
    Style("barrier_escape", (1, 5), .20, .38, "outer"),
    Style("central_escape", (2, 2), .23, .42, "central"),
    Style("coupled_escape", (2, 2), .18, .36, "both"),
    Style("local_refinement", (2, 2), .045, .12, "local"),
    Style("broad_restart", (2, 2), .28, .55, "both"),
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
            return np.array([np.inf, np.inf])
        return np.array([
            float(self.result["dz_peak_m"]) / 3e-6,
            float(self.result["barrier_ev"]) / 1e-3,
        ])

    def copy(self) -> "Individual":
        return Individual(self.genome.copy(), dict(self.result), self.island,
                          self.generation, self.origin, self.rank, self.crowding)


@dataclass
class Island:
    style: Style
    rng: np.random.Generator
    pop: list[Individual]


class ElectricField:
    def __init__(self, bem: FixedMeshBEM, sigma: Any):
        self.bem = bem
        self.sigma = sigma

    def electric_field(self, x_m: float, y_m: float, z_m: float):
        field = self.bem.field_batch(np.array([x_m]), np.array([y_m]),
                                     np.array([z_m]), self.sigma)
        return tuple(float(a) for a in self.bem.asnumpy(field)[0, 0])


def baseline() -> np.ndarray:
    x = np.zeros(NGENES)
    x[CI:] = [-25e-6, 20e-6, 150e-6, 2.0, 30e-6, 0.0, 70e-6, 25e-6]
    return x


def seeds() -> list[tuple[str, np.ndarray]]:
    base = baseline()
    height = base.copy()
    height[[6, 7, 8]] = [-12e-6, -12e-6, -9e-6]
    barrier = base.copy()
    barrier[[6, 7, 8]] = [-12e-6, -8e-6, -9e-6]
    entries = [("baseline", base), ("07f_height_geometry", height),
               ("07f_barrier_geometry", barrier)]
    for tag, parent in (("height", height), ("barrier", barrier)):
        for amplitude in (-10e-6, 10e-6):
            x = parent.copy()
            x[[NC + 6, NC + 7, NC + 8]] = amplitude
            entries.append((f"{tag}_outer_{amplitude * 1e6:+.0f}um", x))
    return entries


def repair(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    if x.shape != (NGENES,) or not np.all(np.isfinite(x)):
        raise ValueError(f"expected {NGENES} finite genome values")
    x = np.clip(x, LO, HI)
    # Piecewise-linear slopes must remain < 0.80 per XJunctionParameters.
    for section in (INNER, OUTER):
        v = np.r_[0.0, x[section], 0.0]
        for _ in range(4):
            for j in range(1, len(v)):
                limit = .75 * (KNOTS_M[j] - KNOTS_M[j-1])
                v[j] = np.clip(v[j], v[j-1] - limit, v[j-1] + limit)
            for j in range(len(v) - 2, -1, -1):
                limit = .75 * (KNOTS_M[j+1] - KNOTS_M[j])
                v[j] = np.clip(v[j], v[j+1] - limit, v[j+1] + limit)
        x[section] = v[1:-1]
    return x


def parameters(x: np.ndarray):
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M, arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=float(x[LENGTH]), taper_power=float(x[POWER]),
        inner_edge_shift_at_centre_m=float(x[CI]),
        outer_edge_shift_at_centre_m=float(x[CO]),
        rf_start_radius_override_m=float(x[START]),
        outer_bulge_amplitude_m=float(x[BULGE]),
        outer_bulge_center_m=float(x[CENTER]),
        outer_bulge_sigma_m=float(x[SIGMA]),
        inner_contour_knots_m=tuple(KNOTS_M),
        inner_contour_offsets_m=tuple(np.r_[0.0, x[INNER], 0.0]),
        outer_contour_knots_m=tuple(KNOTS_M),
        outer_contour_offsets_m=tuple(np.r_[0.0, x[OUTER], 0.0]),
    )


def invalid(reason: str, stage: str) -> dict[str, Any]:
    return dict(valid=False, verified=False, reason=reason, stage=stage,
                dz_peak_m=np.inf, barrier_ev=np.inf, dz_rms_m=np.inf,
                lateral_peak_m=np.inf, points=0, converged=0, panels=0)


def evaluate(x: np.ndarray, cfg: Config, stage: str = "coarse", retain: bool = False):
    x = repair(x)
    try:
        p = parameters(x)
        report = check_x_junction_manufacturability(p)
        if not report.valid:
            return invalid("geometry: " + "; ".join(map(str, report.messages)), stage), None
        s = np.linspace(0, p.arm_length_m, 1001)
        ri, ro = p.rail_boundaries_m(s)
        if np.min(ri) <= .5e-6 or np.min(ro-ri) < 18e-6:
            return invalid("rail width or inner-axis clearance", stage), None
        if cfg.fast and stage == "coarse":
            model = build_geometry_aware_quadtree_x_junction_bem(
                p, central_half_extent_m=145e-6, central_max_cell_m=46e-6,
                boundary_max_cell_m=17e-6, outer_max_cell_m=180e-6,
                min_cell_m=8e-6)
        else:
            model = build_geometry_aware_quadtree_x_junction_bem(
                p, central_half_extent_m=180e-6, central_max_cell_m=30e-6,
                boundary_max_cell_m=10e-6, outer_max_cell_m=180e-6,
                min_cell_m=5e-6)
        if model.n_panels > cfg.max_panels:
            return invalid(f"panel budget {model.n_panels} > {cfg.max_panels}", stage), None
        bem = FixedMeshBEM(model.bem.panels_m, backend=cfg.backend)
        bem.assemble(block_rows=64)
        bem.factorize()
        bem.release_matrix()
        sigma = bem.solve_masks(model.bem.electrode_voltages_v)
        field = ElectricField(bem, sigma)
        n = cfg.fine_points if stage == "fine" else cfg.path_points
        xs = np.linspace(-350e-6, 350e-6, n)
        trace = trace_rf_transverse_minimum(
            field, xs, initial_y_m=0.0,
            initial_z_m=cfg.target_height_m,
            residual_tolerance_v_m=1e-3,
            max_transverse_shift_m=25e-6)
        ok = np.asarray(trace.converged, dtype=bool)
        z = np.asarray(trace.z_m, dtype=float)
        y = np.asarray(trace.y_m, dtype=float)
        if not trace.valid or ok.shape != (n,) or not np.all(ok):
            return invalid(f"incomplete RF-null trace {int(np.sum(ok))}/{n}", stage), None
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            return invalid("nonfinite RF-null trace", stage), None
        energy = np.asarray(pseudopotential_profile_ev(
            field, trace, rf_voltage_peak_v=cfg.rf_peak_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S), dtype=float)
        barrier = compute_barrier_metrics(energy, trace)
        if not barrier.valid or energy.shape != (n,) or not np.all(np.isfinite(energy)):
            return invalid("invalid pseudopotential profile", stage), None
        peak = float(np.max(np.abs(z-cfg.target_height_m)))
        rms = float(np.sqrt(np.mean((z-cfg.target_height_m)**2)))
        lateral = float(np.max(np.abs(y)))
        barrier_ev = float(barrier.barrier_height_ev)
        if not np.all(np.isfinite([peak, rms, lateral, barrier_ev])):
            return invalid("nonfinite physical metrics", stage), None
        result = dict(
            valid=True, verified=bool(stage == "fine" and peak < cfg.height_limit_m
                                      and barrier_ev < cfg.barrier_limit_ev
                                      and lateral < cfg.lateral_limit_m),
            reason="ok", stage=stage, backend=bem.backend_name,
            dz_peak_m=peak, dz_rms_m=rms,
            barrier_ev=barrier_ev, lateral_peak_m=lateral,
            barrier_reference_ev=float(barrier.reference_energy_ev),
            points=n, converged=int(np.sum(ok)), panels=int(model.n_panels))
        return result, (model, trace, energy) if retain else None
    except Exception as exc:
        if cfg.backend == "cuda" and "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA OOM: reduce --max-panels or select a coarser mesh") from exc
        return invalid(f"{type(exc).__name__}: {exc}", stage), None


def dominates(a: Individual, b: Individual) -> bool:
    if a.valid != b.valid:
        return a.valid
    if not a.valid:
        return False
    # No intermediate 8 um / 10 meV violation ranking: it collapses the front.
    return bool(np.all(a.objectives <= b.objectives)
                and np.any(a.objectives < b.objectives))


def rank(pop: list[Individual]) -> list[list[int]]:
    n = len(pop)
    defeated = [[] for _ in pop]
    losses = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(i+1, n):
            if dominates(pop[i], pop[j]):
                defeated[i].append(j); losses[j] += 1
            elif dominates(pop[j], pop[i]):
                defeated[j].append(i); losses[i] += 1
    fronts[0] = [i for i in range(n) if losses[i] == 0]
    while fronts[-1]:
        nxt = []
        for i in fronts[-1]:
            for j in defeated[i]:
                losses[j] -= 1
                if losses[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
    fronts.pop()
    for level, group in enumerate(fronts):
        for i in group:
            pop[i].rank = level
            pop[i].crowding = 0.0
        ids = [i for i in group if pop[i].valid]
        if len(ids) <= 2:
            for i in ids:
                pop[i].crowding = np.inf
            continue
        for axis in range(2):
            order = sorted(ids, key=lambda i: pop[i].objectives[axis])
            low, high = pop[order[0]].objectives[axis], pop[order[-1]].objectives[axis]
            if high <= low:
                continue
            pop[order[0]].crowding = pop[order[-1]].crowding = np.inf
            for k in range(1, len(order)-1):
                item = pop[order[k]]
                if np.isfinite(item.crowding):
                    item.crowding += (pop[order[k+1]].objectives[axis]
                                      - pop[order[k-1]].objectives[axis]) / (high-low)
    return fronts


def survivors(pool: list[Individual], size: int) -> list[Individual]:
    fronts = rank(pool)
    chosen = []
    for group in fronts:
        group.sort(key=lambda i: pool[i].crowding, reverse=True)
        chosen.extend(pool[i].copy() for i in group[:size-len(chosen)])
        if len(chosen) == size:
            break
    return chosen


def score(item: Individual, style: Style) -> float:
    if not item.valid:
        return np.inf
    return float(np.dot(np.array(style.weights), item.objectives))


def tournament(pop: list[Individual], rng: np.random.Generator, style: Style) -> Individual:
    a, b = rng.choice(pop, 2, replace=True)
    # Style-specific tournament half the time; Pareto/crowding otherwise.
    if rng.random() < .5:
        return a if score(a, style) < score(b, style) else b
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.crowding != b.crowding:
        return a if a.crowding > b.crowding else b
    return a if score(a, style) < score(b, style) else b


def mutate(x: np.ndarray, style: Style, rng: np.random.Generator, progress: float) -> np.ndarray:
    y = x.copy()
    factor = style.sigma * (1 - .6*progress)
    probabilities = np.full(NGENES, .6)
    emphasis = (INNER if style.focus == "inner" else
                OUTER if style.focus == "outer" else slice(0, NGENES))
    probabilities[emphasis] = .9
    active = rng.random(NGENES) < probabilities
    y[active] += rng.normal(0.0, (HI-LO)[active]*factor)
    shape = np.array([.25, .55, .85, 1., .85, .6, .35, .18, .08, 0.])
    if style.focus in ("inner", "both", "central"):
        y[INNER] += rng.normal(0.0, 13e-6*(1-.5*progress)) * shape
    if style.focus in ("outer", "both", "central"):
        y[OUTER] += rng.normal(0.0, 13e-6*(1-.5*progress)) * shape
    return repair(y)


def record(item: Individual) -> dict[str, Any]:
    return dict(island=item.island, generation=item.generation, origin=item.origin,
                genome=item.genome.tolist(), rank=item.rank,
                crowding=item.crowding, **item.result)


def csv_save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def plots(data: tuple[Any, Any, np.ndarray], result: dict[str, Any], directory: Path) -> None:
    model, trace, energy = data
    panels = np.asarray(model.bem.panels_m)
    mask = np.asarray(model.bem.electrode_voltages_v) > .5
    patches = [Rectangle((p[0]*1e6, p[2]*1e6), (p[1]-p[0])*1e6,
                         (p[3]-p[2])*1e6) for p in panels]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = PatchCollection(patches, array=mask.astype(float), cmap="RdYlBu_r")
    ax.add_collection(colors)
    ax.plot(np.asarray(trace.x_m)*1e6, np.asarray(trace.y_m)*1e6, "k-", lw=1.5)
    ax.set(xlim=(-230, 230), ylim=(-230, 230), xlabel="x [um]", ylabel="y [um]",
           title="Open-centre C4v RF mask and null path")
    ax.set_aspect("equal")
    fig.colorbar(colors, ax=ax)
    fig.tight_layout(); fig.savefig(directory/"layout.png", dpi=170); plt.close(fig)
    x = np.asarray(trace.x_m)*1e6
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
    axes[0].plot(x, (np.asarray(trace.z_m)-TARGET_ION_HEIGHT_M)*1e6)
    axes[0].axhline(3, ls="--", color="r"); axes[0].axhline(-3, ls="--", color="r")
    axes[0].set_ylabel("z-target [um]")
    axes[1].plot(x, (energy-result["barrier_reference_ev"])*1e3)
    axes[1].axhline(1, ls="--", color="r")
    axes[1].set_ylabel("RF pseudo - reference [meV]")
    axes[2].plot(x, np.asarray(trace.y_m)*1e6)
    axes[2].set_ylabel("y [um]"); axes[2].set_xlabel("x [um]")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(directory/"potential_and_height.png", dpi=170); plt.close(fig)


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--islands", type=int, default=6)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--offspring", type=int, default=24)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--migration-interval", type=int, default=4)
    p.add_argument("--migrants", type=int, default=2)
    p.add_argument("--checkpoint-interval", type=int, default=5)
    p.add_argument("--path-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=161)
    p.add_argument("--verify-top", type=int, default=12)
    p.add_argument("--max-panels", type=int, default=7000)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output-dir", type=Path, default=Path("reports/07_targeted_v2"))
    return p


def main() -> None:
    a = args_parser().parse_args()
    status = available_backend(a.backend)
    if not status.available:
        raise RuntimeError(status.reason)
    if not 1 <= a.islands <= len(STYLES) or a.population < 8:
        raise ValueError("islands must be 1..6 and population >= 8")
    if min(a.offspring, a.generations, a.checkpoint_interval, a.path_points,
           a.fine_points, a.max_panels) < 1:
        raise ValueError("all numerical sizes must be positive")
    cfg = Config(status.selected, a.islands, a.population, a.offspring,
                 a.generations, a.migration_interval, a.migrants,
                 a.checkpoint_interval, a.seed, a.fast, a.path_points,
                 a.fine_points, a.verify_top, a.max_panels, a.output_dir)
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out/"settings.json").write_text(json.dumps(dict(
        backend=status.as_dict(), target_height_m=cfg.target_height_m,
        rf_peak_v=cfg.rf_peak_v, rf_omega_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        height_limit_m=cfg.height_limit_m, barrier_limit_ev=cfg.barrier_limit_ev,
        knots_m=KNOTS_M.tolist(), fast=cfg.fast, path_points=cfg.path_points,
        fine_points=cfg.fine_points), indent=2), encoding="utf-8")
    print("Targeted open-centre C4v island GA | backend:", status.selected,
          "| device:", status.device_name, flush=True)
    print("Limits: dz < 3 um AND RF barrier < 1 meV; verified on fine mesh only.")
    rng_master = np.random.default_rng(cfg.seed)
    seed_list = seeds()
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    best_dz_live: tuple[np.ndarray, dict[str, Any]] | None = None
    best_u_live: tuple[np.ndarray, dict[str, Any]] | None = None

    def coarse(x: np.ndarray, bar: tqdm | None = None) -> dict[str, Any]:
        nonlocal best_dz_live, best_u_live

        x = repair(x)
        key = tuple(np.round(x, 12))
        if key not in cache:
            cache[key], _ = evaluate(x, cfg)

        result = dict(cache[key])
        changed = False

        if result["valid"]:
            if (best_dz_live is None or
                    result["dz_peak_m"] < best_dz_live[1]["dz_peak_m"]):
                best_dz_live = (x.copy(), result.copy())
                changed = True

            if (best_u_live is None or
                    result["barrier_ev"] < best_u_live[1]["barrier_ev"]):
                best_u_live = (x.copy(), result.copy())
                changed = True

        if changed:
            (cfg.out / "best_live_coarse.json").write_text(
                json.dumps(
                    {
                        "best_dz": {
                            "genome": best_dz_live[0].tolist(),
                            **best_dz_live[1],
                        },
                        "best_barrier": {
                            "genome": best_u_live[0].tolist(),
                            **best_u_live[1],
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        if bar is not None:
            if best_dz_live is None:
                bar.set_postfix_str("ожидаем пригодный trace", refresh=False)
            else:
                z = best_dz_live[1]
                u = best_u_live[1]
                bar.set_postfix(
                    {
                        "min_dz": (
                            f'{z["dz_peak_m"] * 1e6:.2f}um '
                            f'(U={z["barrier_ev"] * 1e3:.2f}meV)'
                        ),
                        "min_U": (
                            f'{u["barrier_ev"] * 1e3:.2f}meV '
                            f'(dz={u["dz_peak_m"] * 1e6:.2f}um)'
                        ),
                    },
                    refresh=False,
                )

        return result
    islands: list[Island] = []
    with tqdm(total=cfg.islands*cfg.population, desc="Initial BEM", unit="individual") as bar:
        for i, style in enumerate(STYLES[:cfg.islands]):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**32-1)))
            base_seeds = seed_list if i == 0 else seed_list[:3]
            pop = []
            for name, x in base_seeds[:cfg.population]:
                x = repair(x)
                pop.append(Individual(x, coarse(x), i, 0, name)); bar.update()
            while len(pop) < cfg.population:
                source = seed_list[int(rng.integers(len(seed_list)))][1]
                x = repair(source + rng.normal(0, style.init_sigma*(HI-LO)))
                pop.append(Individual(x, coarse(x), i, 0, "initial_exploration")); bar.update()
            rank(pop)
            islands.append(Island(style, rng, pop))
    initial = [item for island in islands for item in island.pop]
    if not any(item.valid for item in initial):
        reasons = sorted({item.result["reason"] for item in initial})
        raise RuntimeError("No evaluable initial candidate: " + repr(reasons[:20]))
    archive: list[Individual] = []
    history: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    def update_archive(new: list[Individual]) -> None:
        nonlocal archive
        unique = {tuple(np.round(item.genome, 12)): item.copy()
                  for item in archive+new if item.valid}
        pool = list(unique.values())
        if not pool:
            archive = []
            return
        first = rank(pool)[0]
        archive = [pool[i].copy() for i in first]
        archive.sort(key=lambda item: -item.crowding)
        archive = archive[:512]
    def snapshot(g: int) -> dict[str, Any]:
        good = [item for island in islands for item in island.pop if item.valid]
        hz = min(good, key=lambda item: item.result["dz_peak_m"])
        hb = min(good, key=lambda item: item.result["barrier_ev"])
        return dict(generation=g, valid=len(good), archive=len(archive),
                    best_dz_um=hz.result["dz_peak_m"]*1e6,
                    barrier_at_best_dz_mev=hz.result["barrier_ev"]*1e3,
                    best_barrier_mev=hb.result["barrier_ev"]*1e3,
                    dz_at_best_barrier_um=hb.result["dz_peak_m"]*1e6,
                    coarse_joint_hits=sum(item.result["dz_peak_m"]<cfg.height_limit_m
                                          and item.result["barrier_ev"]<cfg.barrier_limit_ev
                                          for item in good),
                    unique_evaluations=len(cache))
    update_archive(initial)
    history.append(snapshot(0)); print("G000", history[-1], flush=True)
    for gen in range(1, cfg.generations+1):
        with tqdm(total=cfg.islands*cfg.offspring, desc=f"Generation {gen}", unit="individual") as bar:
            following = []
            for i, island in enumerate(islands):
                rank(island.pop)
                children = []
                while len(children)<cfg.offspring:
                    first = tournament(island.pop, island.rng, island.style)
                    if island.rng.random()<.9:
                        second = tournament(island.pop, island.rng, island.style)
                        alpha = island.rng.uniform(-.15, 1.15, NGENES)
                        x = repair(alpha*first.genome+(1-alpha)*second.genome)
                        origin = "crossover"
                    else:
                        x = first.genome.copy(); origin = "clone"
                    x = mutate(x, island.style, island.rng, gen/cfg.generations)
                    children.append(Individual(x, coarse(x), i, gen, origin))
                    bar.update()
                following.append(Island(island.style, island.rng,
                                         survivors(island.pop+children, cfg.population)))
            islands = following
        if cfg.migration_interval>0 and gen%cfg.migration_interval==0 and len(islands)>1:
            outbound = []
            for island in islands:
                rank(island.pop)
                outbound.append([i.copy() for i in sorted(island.pop,
                    key=lambda item: (item.rank, -item.crowding, score(item, island.style)))[:cfg.migrants]])
            for source, group in enumerate(outbound):
                target = (source+1)%len(islands)
                island = islands[target]
                islands[target] = Island(island.style, island.rng,
                                         survivors(island.pop+group, cfg.population))
                migrations.append(dict(generation=gen, source=source, target=target, count=len(group)))
        update_archive([item for island in islands for item in island.pop])
        history.append(snapshot(gen)); print(f"G{gen:03d}", history[-1], flush=True)
        csv_save(history, cfg.out/"history.csv")
        csv_save([record(item) for item in archive], cfg.out/"pareto_coarse.csv")
        if gen%cfg.checkpoint_interval==0 or gen==cfg.generations:
            csv_save([record(item) for island in islands for item in island.pop],
                     cfg.out/f"population_gen_{gen:04d}.csv")
    csv_save(migrations, cfg.out/"migrations.csv")
    if not archive:
        print("No coarse Pareto candidates; inspect population files.")
        return
    candidates = sorted(archive, key=lambda item: item.result["dz_peak_m"])[:cfg.verify_top]
    candidates += sorted(archive, key=lambda item: item.result["barrier_ev"])[:cfg.verify_top]
    candidates += sorted(archive, key=lambda item: max(item.objectives))[:cfg.verify_top]
    unique = {tuple(np.round(item.genome, 12)): item for item in candidates}
    fine_rows = []
    for idx, item in enumerate(tqdm(unique.values(), desc="Fine validation", unit="candidate"), start=1):
        result, data = evaluate(item.genome, cfg, "fine", retain=True)
        row = dict(rank=idx, genome=item.genome.tolist(), **result)
        fine_rows.append(row)
        if data is not None:
            directory = cfg.out/f"fine_{idx:03d}"
            directory.mkdir(exist_ok=True)
            plots(data, result, directory)
            (directory/"result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    csv_save(fine_rows, cfg.out/"fine_validation.csv")
    hits = [r for r in fine_rows if r["verified"]]
    (cfg.out/"verified_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
    print(f"Fine-verified joint hits: {len(hits)}; results: {cfg.out}")


if __name__ == "__main__":
    main()