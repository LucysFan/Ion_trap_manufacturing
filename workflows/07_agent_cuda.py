"""CUDA-capable C4v X-junction island GA, using the project's 07 geometry.

Place at workflows/07_agent_cuda.py and run from the repository root.
Targets are verified, not promised: |z_null-z_target| < 3 um AND RF barrier
< 1 meV at 100 V peak, with a complete transport trace and connected RF mask.

The existing geometry-aware quadtree builds panels on CPU for each candidate.
The existing core.ga.fixed_bem.FixedMeshBEM assembles, factorizes and evaluates
fields on CUDA when CuPy and a GPU are available. No fake CUDA flag is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.ndimage import label as connected_components
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.targets import TARGET_ION_HEIGHT_M, RF_ANGULAR_FREQUENCY_RAD_S
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.junction_templates import make_house_style_x_junction, baseline_x_junction_rf_mask
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.barrier import pseudopotential_profile_ev, compute_barrier_metrics

# Fixed end knots are zero; seven independent inner and seven outer controls.
KNOTS = np.array([0, 15, 30, 45, 60, 90, 120, 150, 210], dtype=float) * 1e-6
ACTIVE = 7
N = 2 * ACTIVE + 5
LOW = np.array([-30e-6] * 7 + [-30e-6] * 7 + [-45e-6, 5e-6, 85e-6, 1.0, 8e-6])
HIGH = np.array([30e-6] * 7 + [30e-6] * 7 + [-10e-6, 50e-6, 250e-6, 4.0, 35e-6])
STEPS = np.diff(KNOTS)


@dataclass(frozen=True)
class Settings:
    backend: str
    islands: int
    population: int
    offspring: int
    generations: int
    migration_interval: int
    migrants: int
    seed: int
    path_points: int
    fine_points: int
    verify_top: int
    output: Path
    rf_peak_v: float = 100.0
    target_height_m: float = TARGET_ION_HEIGHT_M
    target_dz_m: float = 3e-6
    target_barrier_ev: float = 0.001
    route_extent_m: float = 350e-6
    max_panels: int = 4500


@dataclass(frozen=True)
class IslandStyle:
    name: str
    gene_scale: np.ndarray
    weights: tuple[float, float]
    sigma: float
    coherent: str


@dataclass
class Item:
    x: np.ndarray
    result: dict
    island: int
    generation: int
    origin: str
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result.get("search_valid", False))

    @property
    def obj(self) -> np.ndarray:
        return np.array([self.result["dz_peak_m"] / 3e-6,
                         self.result["barrier_ev"] / .001], dtype=float)

    def copy(self):
        return Item(self.x.copy(), self.result.copy(), self.island,
                    self.generation, self.origin, self.rank, self.crowding)


STYLES = (
    IslandStyle("height", np.array([1.5]*7 + [.7]*7 + [1, 1, 1, 1, 1]), (5., 1.), .045, "inner"),
    IslandStyle("barrier", np.array([.7]*7 + [1.7]*7 + [1, 1.5, 1.3, 1, 1]), (1., 5.), .060, "outer"),
    IslandStyle("central", np.array([1.4]*3 + [.8]*4 + [1.4]*3 + [.8]*4 + [1.3, 1.3, 1.2, 1.2, 1.4]), (2., 2.), .055, "central"),
    IslandStyle("balanced", np.ones(N), (2., 2.), .035, "both"),
    IslandStyle("broad", np.ones(N) * 1.5, (2., 2.), .09, "both"),
    IslandStyle("refinement", np.ones(N) * .7, (2., 2.), .020, "both"),
)


class SolvedField:
    """Scalar field API expected by the project's trace and barrier modules."""
    def __init__(self, solver: FixedMeshBEM, sigma):
        self.solver = solver
        self.sigma = sigma

    def electric_field(self, x_m: float, y_m: float, z_m: float):
        field = self.solver.field_batch(
            np.array([float(x_m)]), np.array([float(y_m)]),
            np.array([float(z_m)]), self.sigma,
        )
        return tuple(float(v) for v in self.solver.asnumpy(field)[0, 0])


def base_vector() -> np.ndarray:
    x = np.zeros(N)
    x[14:] = [-25e-6, 20e-6, 150e-6, 2.0, 30e-6]
    return x


def seeded_vectors():
    base = base_vector()
    height = base.copy()
    height[[3, 4, 5]] = [-12e-6, -12e-6, -9e-6]  # 60/90/120 um
    barrier = base.copy()
    barrier[[3, 4, 5]] = [-12e-6, -8e-6, -9e-6]
    seeds = [("baseline", base), ("07f_height_geometry", height),
             ("07f_barrier_geometry", barrier)]
    for name, original in (("height", height), ("barrier", barrier)):
        for amp in (-10e-6, 10e-6):
            child = original.copy()
            child[[10, 11, 12]] = amp  # outer 60/90/120 um
            seeds.append((f"{name}_outer_{amp*1e6:+.0f}um", child))
    return seeds


def repair(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.shape != (N,) or not np.all(np.isfinite(x)):
        raise ValueError(f"Genome must contain {N} finite numbers")
    x = np.clip(x, LOW, HIGH)
    for offset in (0, ACTIVE):
        controls = np.r_[0.0, x[offset:offset+ACTIVE], 0.0]
        # Existing XJunctionParameters permits offsets <=30 um, slope <=0.8.
        for _ in range(3):
            for k in range(1, len(controls)):
                limit = .78 * STEPS[k-1]
                controls[k] = np.clip(controls[k], controls[k-1]-limit, controls[k-1]+limit)
            for k in range(len(controls)-2, -1, -1):
                limit = .78 * STEPS[k]
                controls[k] = np.clip(controls[k], controls[k+1]-limit, controls[k+1]+limit)
        x[offset:offset+ACTIVE] = controls[1:-1]
    return x


def params(x: np.ndarray):
    inner = tuple(np.r_[0., x[:ACTIVE], 0.])
    outer = tuple(np.r_[0., x[ACTIVE:2*ACTIVE], 0.])
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6, outer_extent_m=900e-6,
        taper_length_m=float(x[16]),
        inner_edge_shift_at_centre_m=float(x[14]),
        outer_edge_shift_at_centre_m=float(x[15]),
        taper_power=float(x[17]),
        rf_start_radius_override_m=float(x[18]),
        inner_contour_knots_m=tuple(KNOTS),
        inner_contour_offsets_m=inner,
        outer_contour_knots_m=tuple(KNOTS),
        outer_contour_offsets_m=outer,
    )


def geometry_reason(p) -> str | None:
    report = check_x_junction_manufacturability(p)
    if not report.valid:
        return "manufacturability: " + "; ".join(
            map(str, report.messages)
        )

    s = np.linspace(0.0, p.arm_length_m, 1201)
    inner, outer = p.rail_boundaries_m(s)

    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return "non-finite rail boundaries"

    if np.min(inner) <= 0.5e-6:
        return "inner boundary reaches transport axis"

    if np.min(outer - inner) < 18e-6:
        return "RF rail narrower than 18 um"

    # The established 07 open-centre mask has four quadrant RF components.
    # Check that each quadrant contains exactly ONE continuous component,
    # rather than incorrectly requiring all four to meet at the centre.
    axis = np.linspace(-260e-6, 260e-6, 261)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")

    mask = np.asarray(
        baseline_x_junction_rf_mask(xx, yy, p),
        dtype=bool,
    )

    cross_connectivity = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )

    labels, n_components = connected_components(
        mask,
        structure=cross_connectivity,
    )

    if n_components != 4:
        return (
            "unexpected RF topology on 2 um raster: "
            f"{n_components} components; expected four quadrant rails"
        )

    quadrants = (
        (xx > 0.0) & (yy > 0.0),
        (xx < 0.0) & (yy > 0.0),
        (xx < 0.0) & (yy < 0.0),
        (xx > 0.0) & (yy < 0.0),
    )

    quadrant_labels = []

    for index, quadrant in enumerate(quadrants, start=1):
        present = np.unique(labels[mask & quadrant])
        present = present[present != 0]

        if len(present) != 1:
            return (
                f"quadrant {index} has {len(present)} "
                "separate RF regions"
            )

        quadrant_labels.append(int(present[0]))

    if len(set(quadrant_labels)) != 4:
        return "quadrant RF regions have unexpected connections"

    return None


def geometry_model(p, fidelity: str):
    if fidelity == "fine":
        return build_geometry_aware_quadtree_x_junction_bem(
            p, central_half_extent_m=180e-6, central_max_cell_m=30e-6,
            boundary_max_cell_m=10e-6, outer_max_cell_m=180e-6,
            min_cell_m=5e-6,
        )
    return build_geometry_aware_quadtree_x_junction_bem(
        p, central_half_extent_m=150e-6, central_max_cell_m=40e-6,
        boundary_max_cell_m=15e-6, outer_max_cell_m=180e-6,
        min_cell_m=7.5e-6,
    )


def invalid(reason: str, fidelity: str) -> dict:
    return dict(search_valid=False, verified=False, reason=reason,
                fidelity=fidelity, dz_peak_m=float("inf"),
                dz_rms_m=float("inf"), barrier_ev=float("inf"),
                lateral_peak_m=float("inf"), converged=0, points=0)


def evaluate(x: np.ndarray, cfg: Settings, fidelity="coarse", artifacts=False):
    x = repair(x)
    try:
        p = params(x)
        reason = geometry_reason(p)
        if reason:
            return invalid(reason, fidelity), None
        model = geometry_model(p, fidelity)
        panels = model.bem.panels_m
        if model.n_panels > cfg.max_panels:
            return invalid(f"panel budget {model.n_panels}>{cfg.max_panels}", fidelity), None
        # The builder's BEM2D defines the mesh only. Its CPU solve is NOT run.
        solver = FixedMeshBEM(panels, backend=cfg.backend)
        solver.assemble(block_rows=64)
        solver.factorize()
        solver.release_matrix()
        sigma = solver.solve_masks(model.bem.electrode_voltages_v)
        field = SolvedField(solver, sigma)
        count = cfg.fine_points if fidelity == "fine" else cfg.path_points
        route = np.linspace(-cfg.route_extent_m, cfg.route_extent_m, count)
        trace = trace_rf_transverse_minimum(
            field, route, initial_y_m=0.0,
            initial_z_m=cfg.target_height_m,
            residual_tolerance_v_m=1e-3, max_transverse_shift_m=25e-6,
        )
        good = np.asarray(trace.converged, dtype=bool)
        z = np.asarray(trace.z_m, dtype=float)
        y = np.asarray(trace.y_m, dtype=float)
        # No partial-path candidate is declared successful.
        complete = bool(trace.valid and good.shape == (count,) and np.all(good))
        if not complete or not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            return invalid(f"incomplete trace: {np.count_nonzero(good)}/{count}", fidelity), None
        energies = np.asarray(pseudopotential_profile_ev(
            field, trace, rf_voltage_peak_v=cfg.rf_peak_v,
            rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
        ), dtype=float)
        barrier = compute_barrier_metrics(energies, trace)
        dz = z - cfg.target_height_m
        dz_peak = float(np.max(np.abs(dz)))
        dz_rms = float(np.sqrt(np.mean(dz**2)))
        barrier_ev = float(barrier.barrier_height_ev)
        lateral = float(np.max(np.abs(y)))
        sound = (bool(barrier.valid) and len(energies) == count and
                 np.all(np.isfinite(energies)) and
                 np.all(np.isfinite([dz_peak, dz_rms, barrier_ev, lateral])) and
                 barrier_ev >= -1e-9)
        if not sound:
            return invalid("nonfinite trace or barrier", fidelity), None
        result = dict(search_valid=True,
                      verified=bool(fidelity == "fine" and dz_peak < cfg.target_dz_m
                                    and barrier_ev < cfg.target_barrier_ev and lateral <= 5e-6),
                      fidelity=fidelity, reason="ok", backend=solver.backend_name,
                      panels=int(model.n_panels), dz_peak_m=dz_peak,
                      dz_rms_m=dz_rms, barrier_ev=barrier_ev,
                      lateral_peak_m=lateral, converged=int(np.sum(good)), points=count,
                      rail_width_min_m=float(np.min(p.rail_boundaries_m(
                          np.linspace(0, p.arm_length_m, 1201))[1] -
                          p.rail_boundaries_m(np.linspace(0, p.arm_length_m, 1201))[0])))
        data = (model, trace, energies, p) if artifacts else None
        return result, data
    except (MemoryError, Exception) as exc:
        # Preserve the exact exception text: do not hide a broken evaluator.
        if cfg.backend == "cuda" and "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA out of memory; reduce --max-panels or use --backend cpu") from exc
        return invalid(f"{type(exc).__name__}: {exc}", fidelity), None


def dominance(a: Item, b: Item) -> bool:
    if a.valid != b.valid:
        return a.valid
    if not a.valid:
        return False
    return bool(np.all(a.obj <= b.obj) and np.any(a.obj < b.obj))


def ranks(pop: list[Item]) -> list[list[int]]:
    n = len(pop)
    beats = [[] for _ in pop]
    losses = np.zeros(n, dtype=int)
    front = [[]]
    for i in range(n):
        for j in range(i+1, n):
            if dominance(pop[i], pop[j]):
                beats[i].append(j); losses[j] += 1
            elif dominance(pop[j], pop[i]):
                beats[j].append(i); losses[i] += 1
    front[0] = [i for i in range(n) if losses[i] == 0]
    while front[-1]:
        following = []
        for i in front[-1]:
            for j in beats[i]:
                losses[j] -= 1
                if losses[j] == 0:
                    following.append(j)
        front.append(following)
    front.pop()
    for rank, group in enumerate(front):
        for i in group:
            pop[i].rank = rank; pop[i].crowding = 0.0
        valid = [i for i in group if pop[i].valid]
        if len(valid) <= 2:
            for i in valid:
                pop[i].crowding = np.inf
            continue
        for objective in range(2):
            ordered = sorted(valid, key=lambda i: pop[i].obj[objective])
            lo, hi = pop[ordered[0]].obj[objective], pop[ordered[-1]].obj[objective]
            if hi <= lo:
                continue
            pop[ordered[0]].crowding = pop[ordered[-1]].crowding = np.inf
            for k in range(1, len(ordered)-1):
                if np.isfinite(pop[ordered[k]].crowding):
                    pop[ordered[k]].crowding += (pop[ordered[k+1]].obj[objective] -
                                                  pop[ordered[k-1]].obj[objective]) / (hi-lo)
    return front


def select(pop: list[Item], n: int) -> list[Item]:
    fronts = ranks(pop)
    picked = []
    for group in fronts:
        group.sort(key=lambda i: (pop[i].crowding, -float(np.sum(pop[i].obj)) if pop[i].valid else 0), reverse=True)
        picked.extend(pop[i].copy() for i in group[:n-len(picked)])
        if len(picked) >= n:
            break
    return picked


def weighted(item: Item, style: IslandStyle) -> float:
    if not item.valid:
        return np.inf
    return float(np.dot(item.obj, style.weights) + .03 * roughness(item.x))


def roughness(x: np.ndarray) -> float:
    return float(np.sum((np.diff(x[:7], n=2)/20e-6)**2) +
                 np.sum((np.diff(x[7:14], n=2)/20e-6)**2))


def tournament(pop, rng, style):
    a, b = rng.integers(len(pop), size=2)
    a, b = pop[int(a)], pop[int(b)]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.crowding != b.crowding:
        return a if a.crowding > b.crowding else b
    return a if weighted(a, style) < weighted(b, style) else b


def mutate(x, rng, style, progress):
    out = x.copy()
    std = style.sigma * (HIGH-LOW) * style.gene_scale * (1-.65*progress)
    mask = rng.random(N) < .6
    out[mask] += rng.normal(0, std[mask])
    if style.coherent in ("outer", "both"):
        out[7:14] += rng.normal(0, 3e-6) * np.array([.2, .6, 1, 1, 1, .5, .2])
    if style.coherent in ("inner", "both"):
        out[:7] += rng.normal(0, 3e-6) * np.array([.2, .5, 1, 1, 1, .6, .2])
    if style.coherent == "central":
        out[[0, 1, 7, 8, 14, 15, 18]] += rng.normal(0, 2e-6, 7)
    return repair(out)


def record(item: Item) -> dict:
    return dict(island=item.island, generation=item.generation,
                origin=item.origin, genome=item.x.tolist(), **item.result)


def save_csv(rows, path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def plot_candidate(data, result, directory):
    model, trace, energies, p = data
    panels = np.asarray(model.bem.panels_m)
    rf = np.asarray(model.bem.electrode_voltages_v) > .5
    fig, ax = plt.subplots(figsize=(7, 7))
    patches = [Rectangle((q[0]*1e6, q[2]*1e6), (q[1]-q[0])*1e6,
                         (q[3]-q[2])*1e6) for q in panels]
    coll = PatchCollection(patches, array=rf.astype(float), cmap="RdYlBu_r")
    ax.add_collection(coll)
    ax.plot(np.asarray(trace.x_m)*1e6, np.asarray(trace.y_m)*1e6, "k-", lw=1)
    ax.set(xlim=(-240, 240), ylim=(-240, 240), xlabel="x [um]", ylabel="y [um]",
           title="C4v RF mask and null path")
    ax.set_aspect("equal"); fig.colorbar(coll, ax=ax, label="RF mask [1 V basis]")
    fig.tight_layout(); fig.savefig(directory/"layout.png", dpi=160); plt.close(fig)
    x = np.asarray(trace.x_m)*1e6
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
    axes[0].plot(x, (np.asarray(trace.z_m)-TARGET_ION_HEIGHT_M)*1e6)
    axes[0].axhline(3, color="r", ls="--"); axes[0].axhline(-3, color="r", ls="--")
    axes[0].set_ylabel("z - target [um]")
    axes[1].plot(x, np.asarray(trace.y_m)*1e6); axes[1].set_ylabel("y [um]")
    axes[2].plot(x, energies*1e3)
    axes[2].axhline(1, color="r", ls="--"); axes[2].set_ylabel("RF pseudo [meV]")
    axes[2].set_xlabel("transport x [um]")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directory/"profiles.png", dpi=160); plt.close(fig)
    (directory/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--islands", type=int, default=6)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--offspring", type=int, default=16)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--migration-interval", type=int, default=5)
    p.add_argument("--migrants", type=int, default=2)
    p.add_argument("--path-points", type=int, default=41)
    p.add_argument("--fine-points", type=int, default=121)
    p.add_argument("--verify-top", type=int, default=6)
    p.add_argument("--max-panels", type=int, default=4500)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output-dir", type=Path, default=Path("reports/07_agent_cuda"))
    return p.parse_args()


def main():
    a = parse_args()
    status = available_backend(a.backend)
    if not status.available:
        raise RuntimeError(status.reason)
    if not 1 <= a.islands <= len(STYLES) or a.population < 8 or a.offspring < 1:
        raise ValueError("islands must be 1..6, population >=8, offspring >=1")
    if a.generations < 1 or a.path_points < 11 or a.fine_points < a.path_points:
        raise ValueError("generations >=1, path-points >=11, fine-points >= path-points")
    cfg = Settings(status.selected, a.islands, a.population, a.offspring,
                   a.generations, a.migration_interval, a.migrants, a.seed,
                   a.path_points, a.fine_points, a.verify_top, a.output_dir,
                   max_panels=a.max_panels)
    cfg.output.mkdir(parents=True, exist_ok=True)
    print("Backend:", status.selected, status.device_name, status.reason, flush=True)
    print("NOTE: the GA seeks <3 um and <1 meV; it cannot guarantee either.", flush=True)
    (cfg.output/"settings.json").write_text(json.dumps({
        "backend": status.as_dict(), "target_height_m": cfg.target_height_m,
        "rf_peak_v": cfg.rf_peak_v, "omega_rad_s": RF_ANGULAR_FREQUENCY_RAD_S,
        "dz_limit_m": cfg.target_dz_m, "barrier_limit_ev": cfg.target_barrier_ev,
        "knots_m": KNOTS.tolist(), "coarse_points": cfg.path_points,
        "fine_points": cfg.fine_points,
    }, indent=2), encoding="utf-8")
    master = np.random.default_rng(cfg.seed)
    specs = STYLES[:cfg.islands]
    seeds = seeded_vectors()
    islands = []
    initial = []
    total = cfg.islands*cfg.population
    with tqdm(total=total, desc="Initial BEM", unit="candidate") as bar:
        for index, style in enumerate(specs):
            rng = np.random.default_rng(int(master.integers(0, 2**32-1)))
            population = []
            # One of each known geometry per island, then nearby exploration.
            for name, vector in seeds:
                res, _ = evaluate(vector, cfg)
                population.append(Item(repair(vector), res, index, 0, name))
                bar.update()
            while len(population) < cfg.population:
                origin = seeds[int(rng.integers(len(seeds)))][1]
                vector = repair(origin + rng.normal(0, .07*(HIGH-LOW)*style.gene_scale))
                res, _ = evaluate(vector, cfg)
                population.append(Item(vector, res, index, 0, "random"))
                bar.update()
            ranks(population)
            islands.append((style, rng, population))
            initial.extend(population)
    if not any(item.valid for item in initial):
        reasons = sorted({item.result["reason"] for item in initial})
        raise RuntimeError("No evaluable seed; fix BEM/geometry first: " + repr(reasons[:12]))
    archive = []
    history = []
    events = []

    def snapshot(gen):
        all_items = [item for _, _, pop in islands for item in pop]
        good = [item for item in all_items if item.valid]
        return dict(generation=gen, evaluable=len(good),
                    minimum_dz_um=min((item.result["dz_peak_m"]*1e6 for item in good), default=float("inf")),
                    minimum_barrier_mev=min((item.result["barrier_ev"]*1e3 for item in good), default=float("inf")),
                    coarse_joint_count=sum(item.result["dz_peak_m"] < cfg.target_dz_m and
                                           item.result["barrier_ev"] < cfg.target_barrier_ev for item in good),
                    archive=len(archive))

    def update_archive(items):
        nonlocal archive
        pool = [item.copy() for item in archive+items if item.valid]
        # Deduplicate repeated seeds and migrants before nondominated sorting.
        unique = {tuple(np.round(item.x, 12)): item for item in pool}
        pool = list(unique.values())
        if pool:
            fronts = ranks(pool)
            archive = [pool[i].copy() for i in fronts[0]]
            archive = sorted(archive, key=lambda i: -i.crowding)[:256]

    update_archive(initial)
    history.append(snapshot(0))
    print("G0", history[-1], flush=True)
    for gen in range(1, cfg.generations+1):
        next_islands = []
        with tqdm(total=cfg.islands*cfg.offspring, desc=f"G{gen} BEM", unit="candidate") as bar:
            for index, (style, rng, pop) in enumerate(islands):
                ranks(pop)
                offspring = []
                while len(offspring) < cfg.offspring:
                    left = tournament(pop, rng, style)
                    if rng.random() < .9:
                        right = tournament(pop, rng, style)
                        alpha = rng.uniform(-.1, 1.1, N)
                        vector = repair(alpha*left.x + (1-alpha)*right.x)
                    else:
                        vector = left.x.copy()
                    vector = mutate(vector, rng, style, gen/cfg.generations)
                    result, _ = evaluate(vector, cfg)
                    offspring.append(Item(vector, result, index, gen, "offspring"))
                    bar.update()
                next_islands.append((style, rng, select(pop+offspring, cfg.population)))
        islands = next_islands
        if cfg.migration_interval > 0 and gen % cfg.migration_interval == 0 and cfg.islands > 1:
            outgoing = []
            for style, rng, pop in islands:
                ranks(pop)
                outgoing.append([i.copy() for i in sorted(pop,
                    key=lambda v: (v.rank, -v.crowding, weighted(v, style)))[:cfg.migrants]])
            for j, group in enumerate(outgoing):
                k = (j+1) % len(islands)
                style, rng, pop = islands[k]
                islands[k] = (style, rng, select(pop+group, cfg.population))
                events.append(dict(generation=gen, from_island=j, to_island=k, count=len(group)))
        update_archive([item for _, _, pop in islands for item in pop])
        history.append(snapshot(gen))
        print(f"G{gen}: {history[-1]}", flush=True)
        save_csv(history, cfg.output/"history.csv")
        save_csv([record(item) for item in archive], cfg.output/"pareto_coarse.csv")
    save_csv([record(item) for _, _, pop in islands for item in pop], cfg.output/"population.csv")
    save_csv(events, cfg.output/"migrations.csv")
    if not archive:
        print("No valid coarse candidates; see population.csv.")
        return
    xs = [i.result["dz_peak_m"]*1e6 for i in archive]
    ys = [i.result["barrier_ev"]*1e3 for i in archive]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=26); ax.axvline(3, ls="--", color="red")
    ax.axhline(1, ls="--", color="red"); ax.set(xlabel="fixed-target dz [um]", ylabel="RF barrier [meV]")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(cfg.output/"pareto.png", dpi=160); plt.close(fig)
    # Validate physical claims ONLY with a finer geometry-aware mesh and denser path.
    by_height = sorted(archive, key=lambda i: i.obj[0])[:cfg.verify_top]
    by_barrier = sorted(archive, key=lambda i: i.obj[1])[:cfg.verify_top]
    by_joint = sorted(archive, key=lambda i: max(i.obj))[:cfg.verify_top]
    chosen = list({tuple(np.round(i.x, 12)): i for i in by_height+by_barrier+by_joint}.values())
    verified = []
    for idx, item in enumerate(tqdm(chosen, desc="Fine validation", unit="candidate"), 1):
        result, data = evaluate(item.x, cfg, "fine", artifacts=True)
        row = dict(origin=item.origin, genome=item.x.tolist(), **result)
        verified.append(row)
        if data is not None:
            directory = cfg.output/f"fine_{idx:03d}"
            directory.mkdir(exist_ok=True)
            plot_candidate(data, result, directory)
    save_csv(verified, cfg.output/"fine_validation.csv")
    hits = [r for r in verified if r["verified"]]
    (cfg.output/"verified_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
    print(f"Fine verified joint hits: {len(hits)}; outputs: {cfg.output}")


if __name__ == "__main__":
    main()
