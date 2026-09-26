"""Workflow 09c: mixed ordinary/special connected-contour island GA.

This module deliberately imports the established workflow-09 implementation
instead of duplicating its electrostatics, BEM, RF-null trace, NSGA-II,
reporting, or conventional C4v geometry pipeline.  Special islands use the
same full numerical genome, but project it onto the connected 07c--07g contour
family before every geometry evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE_PATH = Path(__file__).with_name("09_movable_knot_full_genome_island_ga.py")
_SPEC = importlib.util.spec_from_file_location("workflow_09_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_BASE_PATH}")
w09 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = w09
_SPEC.loader.exec_module(w09)

from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.movable_knot_xjunction import fixed_knots_to_logits

UM = 1e-6
SPECIAL_KNOTS_M = np.array([0, 7, 15, 27, 43, 65, 92, 122, 150], dtype=float) * UM
SPECIAL_LOCK_M = 150 * UM
SPECIAL_LOGITS = fixed_knots_to_logits(SPECIAL_KNOTS_M, min_gap_m=w09.MIN_GAP_M)
SPECIAL_INTERIOR_KNOTS_UM = SPECIAL_KNOTS_M[1:-1] / UM
SPECIAL_BASELINE = {
    "center_in": -25 * UM,
    "center_out": 20 * UM,
    "start": 30 * UM,
    "length": 150 * UM,
    "power": 2.0,
    "bulge": 0.0,
    "bulge_center": 70 * UM,
    "bulge_width": 25 * UM,
}
# Physical target nodes use the existing linear movable-knot contour.  Values
# at 60/90/120 um are encoded by linear interpolation between neighbouring
# fixed 09 knot positions.  Inverting that interpolation below verifies the
# mapping, rather than assuming an unsafe genome slice.
SPECIAL_X_UM = np.array([60.0, 90.0, 120.0])


def _interpolation_matrix() -> np.ndarray:
    nodes = SPECIAL_INTERIOR_KNOTS_UM
    matrix = np.zeros((3, len(nodes)), dtype=float)
    for row, x in enumerate(SPECIAL_X_UM):
        hi = int(np.searchsorted(nodes, x, side="right"))
        lo = hi - 1
        if hi >= len(nodes):
            hi = len(nodes) - 1
            lo = hi - 1
        xl, xr = nodes[lo], nodes[hi]
        t = (x - xl) / (xr - xl)
        matrix[row, lo] = 1.0 - t
        matrix[row, hi] = t
    return matrix


SPECIAL_MAP = _interpolation_matrix()
SPECIAL_PSEUDOINVERSE = np.linalg.pinv(SPECIAL_MAP)


def _decode_special_controls(genome: np.ndarray) -> tuple[np.ndarray, float]:
    """Read physical d_in(60,90,120) and broad outer amplitude from full genome."""
    x = w09.repair(genome)
    inner = np.asarray(x[w09.INNER], dtype=float)
    outer = np.asarray(x[w09.OUTER], dtype=float)
    d = SPECIAL_MAP @ inner
    a = float(np.mean(SPECIAL_MAP @ outer))
    return d, a


def _encode_special_controls(d: np.ndarray, a_out: float) -> tuple[np.ndarray, np.ndarray]:
    """Return full interior arrays whose contour values reproduce special controls."""
    d = np.asarray(d, dtype=float)
    # Minimum-norm exact solution of the three independent interpolation
    # constraints.  The unused nearby full-genome knots remain zero, giving a
    # smooth local feature with endpoint attachment retained by workflow 09.
    inner = SPECIAL_PSEUDOINVERSE @ d
    outer = SPECIAL_PSEUDOINVERSE @ np.full(3, float(a_out))
    if not np.allclose(SPECIAL_MAP @ inner, d, atol=1e-15):
        raise RuntimeError("special inner-contour mapping is not exact")
    if not np.allclose(SPECIAL_MAP @ outer, a_out, atol=1e-15):
        raise RuntimeError("special outer-contour mapping is not exact")
    return inner, outer


def special_07_repair(genome: np.ndarray) -> np.ndarray:
    """Project an arbitrary full genome to the validated connected 07c--07g family."""
    raw = np.asarray(genome, dtype=float)
    if raw.shape != (w09.N_GENES,) or not np.all(np.isfinite(raw)):
        raw = w09.baseline()
    d, a_out = _decode_special_controls(raw)
    d = np.clip(d, -18 * UM, 4 * UM)
    a_out = float(np.clip(a_out, -10 * UM, 10 * UM))
    inner, outer = _encode_special_controls(d, a_out)
    x = w09.baseline()
    x[w09.GAPS] = SPECIAL_LOGITS
    x[w09.INNER] = inner
    x[w09.OUTER] = outer
    x[w09.LOCK] = SPECIAL_LOCK_M
    x[w09.CENTER_IN] = SPECIAL_BASELINE["center_in"]
    x[w09.CENTER_OUT] = SPECIAL_BASELINE["center_out"]
    x[w09.START] = SPECIAL_BASELINE["start"]
    x[w09.LENGTH] = SPECIAL_BASELINE["length"]
    x[w09.POWER] = SPECIAL_BASELINE["power"]
    x[w09.BULGE] = SPECIAL_BASELINE["bulge"]
    x[w09.BULGE_CENTER] = SPECIAL_BASELINE["bulge_center"]
    x[w09.BULGE_WIDTH] = SPECIAL_BASELINE["bulge_width"]
    return w09.repair(x)


def special_feature_seed(rng: np.random.Generator) -> tuple[str, np.ndarray]:
    """Random seed in the constrained connected 07c--07g contour family."""
    mode = str(rng.choice(("triangular", "broad_inner", "coupled", "outer_inward", "outer_outward")))
    if mode == "triangular":
        d90 = rng.uniform(-16, -11) * UM
        d = np.array([d90 * rng.uniform(.35, .78), d90, d90 * rng.uniform(.30, .72)])
        a = 0.0
    elif mode == "broad_inner":
        level = rng.uniform(-13, -8) * UM
        d = level + rng.normal(0.0, 0.8 * UM, 3)
        a = 0.0
    else:
        d = np.array([rng.uniform(-15, -6), rng.uniform(-16, -8), rng.uniform(-12, -6)]) * UM
        if mode == "outer_inward":
            a = rng.uniform(-10, -3) * UM
        elif mode == "outer_outward":
            a = rng.uniform(3, 10) * UM
        else:
            a = rng.uniform(-10, 10) * UM
    inner, outer = _encode_special_controls(d, a)
    x = w09.baseline()
    x[w09.INNER] = inner
    x[w09.OUTER] = outer
    return mode, special_07_repair(x)


def mutate_special_07_family(
    parent: np.ndarray,
    rng: np.random.Generator,
    fraction: float = 0.0,
    burst: bool = False,
    burst_mutation_multiplier: float = 1.0,
) -> np.ndarray:
    """Mutate only physical d60/d90/d120/a_out; no chained NumPy indexing."""
    child = special_07_repair(parent)
    d, a = _decode_special_controls(child)
    scale = (1.0 - 0.55 * min(1.0, fraction)) * (burst_mutation_multiplier if burst else 1.0)
    active = rng.random(4) < (0.86 if burst else 0.58)
    if not np.any(active):
        active[int(rng.integers(4))] = True
    values = np.r_[d, a]
    sigma = np.array([2.2, 2.4, 2.0, 2.2]) * UM * scale
    values[active] += rng.normal(0.0, sigma[active])
    values[:3] = np.clip(values[:3], -18 * UM, 4 * UM)
    values[3] = np.clip(values[3], -10 * UM, 10 * UM)
    inner, outer = _encode_special_controls(values[:3], values[3])
    # Explicit copy/write avoids boolean chained-indexing temporaries.
    middle = child[w09.INNER].copy()
    middle[:] = inner
    child[w09.INNER] = middle
    outer_middle = child[w09.OUTER].copy()
    outer_middle[:] = outer
    child[w09.OUTER] = outer_middle
    return special_07_repair(child)


@dataclass(frozen=True)
class Settings:
    backend: str
    ordinary_islands: int = 21
    special_islands: int = 9
    population: int = 20
    offspring: int = 8
    generations: int = 100
    migration_interval: int = 3
    migrants: int = 3
    retain_per_island: int = 3
    seed: int = 20260926
    coarse_points: int = 61
    fine_points: int = 181
    fine_top: int = 25
    max_panels: int = 7000
    fast: bool = False
    animation_stride: int = 1
    stall_generations: int = 5
    burst_generations: int = 3
    improvement_fraction: float = 0.01
    burst_mutation_multiplier: float = 3.0
    output_dir: Path = Path("reports/09c_mixed_island_ga")


@dataclass
class MixedIndividual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    family: str
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result.get("valid", False))

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf])
        return np.array([self.result["dz_peak_m"] / 3e-6, self.result["barrier_ev"] / 1e-3])

    def copy(self) -> "MixedIndividual":
        return MixedIndividual(self.genome.copy(), dict(self.result), self.island, self.generation,
                               self.origin, self.family, self.rank, self.crowding)


@dataclass
class MixedIsland:
    style: Any
    rng: np.random.Generator
    population: list[MixedIndividual]
    family: str


def _ordinary_style(index: int) -> Any:
    return w09.STYLES[index % len(w09.STYLES)]


def _result_for(genome: np.ndarray, family: str, cfg: Settings, cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]]) -> dict[str, Any]:
    x = special_07_repair(genome) if family == "special" else w09.repair(genome)
    key = (family, tuple(np.round(x, 12)))
    if key not in cache:
        base_cfg = w09.Settings(
            cfg.backend, 1, cfg.population, cfg.offspring, cfg.generations,
            cfg.migration_interval, cfg.migrants, cfg.retain_per_island, cfg.seed,
            cfg.coarse_points, cfg.fine_points, cfg.fine_top, cfg.max_panels,
            cfg.fast, cfg.animation_stride, cfg.stall_generations, cfg.burst_generations,
            cfg.improvement_fraction, cfg.burst_mutation_multiplier, cfg.output_dir,
        )
        cache[key], _ = w09.evaluate(x, base_cfg, "coarse")
    return dict(cache[key])


def _serialize(ind: MixedIndividual) -> dict[str, Any]:
    base = w09.serialize(w09.Individual(ind.genome, ind.result, ind.island, ind.generation,
                                        ind.origin, ind.rank, ind.crowding))
    base["family"] = ind.family
    return base


def _rank(pop: list[MixedIndividual]) -> None:
    proxy = [w09.Individual(i.genome, i.result, i.island, i.generation, i.origin, i.rank, i.crowding) for i in pop]
    w09.rank_and_crowding(proxy)
    for source, target in zip(proxy, pop):
        target.rank, target.crowding = source.rank, source.crowding


def _survivors(pool: list[MixedIndividual], size: int) -> list[MixedIndividual]:
    _rank(pool)
    out: list[MixedIndividual] = []
    for rank in sorted({i.rank for i in pool}):
        front = [i for i in pool if i.rank == rank]
        front.sort(key=lambda i: i.crowding, reverse=True)
        out.extend(i.copy() for i in front[:size - len(out)])
        if len(out) == size:
            break
    return out


def _joint(result: dict[str, Any]) -> float:
    if not result.get("valid", False):
        return np.inf
    return max(result["dz_peak_m"] / 10e-6, result["barrier_ev"] / 10e-3)


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    w09.save_csv(rows, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--ordinary-islands", type=int, default=21)
    p.add_argument("--special-islands", type=int, default=9)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--offspring", type=int, default=8)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--migration-interval", type=int, default=3)
    p.add_argument("--migrants", type=int, default=3)
    p.add_argument("--retain-per-island", type=int, default=3)
    p.add_argument("--coarse-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=181)
    p.add_argument("--fine-top", type=int, default=25)
    p.add_argument("--max-panels", type=int, default=7000)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--animation-stride", type=int, default=1)
    p.add_argument("--stall-generations", type=int, default=5)
    p.add_argument("--burst-generations", type=int, default=3)
    p.add_argument("--improvement-fraction", type=float, default=0.01)
    p.add_argument("--burst-mutation-multiplier", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--output-dir", type=Path, default=Path("reports/09c_mixed_island_ga"))
    return p


def settings_from_args(a: argparse.Namespace) -> Settings:
    if a.smoke:
        return Settings(a.backend, 2, 1, 8, 2, 2, 1, 1, 3, a.seed, 31, 61, 5,
                        5000, True, 1, 1, 1, a.improvement_fraction,
                        a.burst_mutation_multiplier, a.output_dir)
    return Settings(a.backend, a.ordinary_islands, a.special_islands, a.population, a.offspring,
                    a.generations, a.migration_interval, a.migrants, a.retain_per_island,
                    a.seed, a.coarse_points, a.fine_points, a.fine_top, a.max_panels, a.fast,
                    a.animation_stride, a.stall_generations, a.burst_generations,
                    a.improvement_fraction, a.burst_mutation_multiplier, a.output_dir)


def _seed_population(iid: int, family: str, style: Any, rng: np.random.Generator, cfg: Settings,
                     cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]]) -> list[MixedIndividual]:
    pop: list[MixedIndividual] = []
    for n in range(cfg.population):
        if family == "special":
            name, x = special_feature_seed(rng)
        elif iid == 0 and n == 0:
            name, x = "conventional_exact", w09.conventional_cross()
        else:
            name, seed = w09.seeds()[int(rng.integers(len(w09.seeds())))]
            x = w09.repair(seed + rng.normal(0.0, (w09.HIGH - w09.LOW) * .012 * style.explore))
        pop.append(MixedIndividual(x, _result_for(x, family, cfg, cache), iid, 0, f"seed_{name}", family))
    return pop


def run(cfg: Settings) -> None:
    backend = w09.available_backend(cfg.backend)
    if not backend.available:
        raise RuntimeError(backend.reason)
    if cfg.ordinary_islands < 1 or cfg.special_islands < 0 or cfg.population < 2:
        raise ValueError("require ordinary_islands>=1, special_islands>=0, population>=2")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "settings.json").write_text(json.dumps({"settings": asdict(cfg), "backend": backend.as_dict(),
        "special_mapping": {"physical_nodes_um": SPECIAL_X_UM.tolist(), "fixed_knots_um": (SPECIAL_KNOTS_M / UM).tolist()}}, indent=2, default=str), encoding="utf-8")
    cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    master = np.random.default_rng(cfg.seed)
    islands: list[MixedIsland] = []
    total = cfg.ordinary_islands + cfg.special_islands
    for iid in range(total):
        family = "ordinary" if iid < cfg.ordinary_islands else "special"
        style = _ordinary_style(iid)
        rng = np.random.default_rng(int(master.integers(2**63)))
        islands.append(MixedIsland(style, rng, _seed_population(iid, family, style, rng, cfg, cache), family))
    history: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    archive: dict[tuple[str, tuple[float, ...]], MixedIndividual] = {}
    best_seen = np.inf
    stall = 0
    burst_remaining = 0

    def update_archive() -> None:
        for isl in islands:
            for item in isl.population:
                if item.valid:
                    archive[(item.family, tuple(np.round(item.genome, 12)))] = item.copy()

    def record(gen: int, mode: str) -> None:
        flat = [i for isl in islands for i in isl.population]
        valid = [i for i in flat if i.valid]
        row = dict(generation=gen, mode=mode, stall_count=stall, burst_remaining=burst_remaining,
                   joint_score=min((_joint(i.result) for i in valid), default=np.inf), valid=len(valid),
                   invalid=len(flat)-len(valid), archive=len(archive), unique_evaluations=len(cache), rejects="")
        if valid:
            z = min(valid, key=lambda i: i.result["dz_peak_m"])
            u = min(valid, key=lambda i: i.result["barrier_ev"])
            j = min(valid, key=lambda i: _joint(i.result))
            row.update(best_dz_um=z.result["dz_peak_m"] / UM, best_U_mev=u.result["barrier_ev"] * 1e3,
                       best_joint_dz_um=j.result["dz_peak_m"] / UM, best_joint_U_mev=j.result["barrier_ev"] * 1e3)
        history.append(row)
        for reason in sorted({i.result.get("reason", "") for i in flat if not i.valid}):
            if reason:
                rejects.append({"generation": gen, "reason": reason, "count": sum(i.result.get("reason") == reason for i in flat)})

    update_archive(); record(0, "NORMAL")
    _save_csv([_serialize(i) for isl in islands for i in isl.population], cfg.output_dir / "population_gen_0000.csv")
    for gen in range(1, cfg.generations + 1):
        burst = burst_remaining > 0
        mode = "BURST" if burst else "NORMAL"
        print(f"Generation {gen} [{mode}]", flush=True)
        next_islands: list[MixedIsland] = []
        for iid, isl in enumerate(islands):
            _rank(isl.population)
            children: list[MixedIndividual] = []
            while len(children) < cfg.offspring:
                parents = sorted(isl.population, key=lambda i: (i.rank, -i.crowding))[:max(2, min(5, len(isl.population)))]
                if burst and isl.rng.random() < .20:
                    if isl.rng.random() < .5:
                        origin, x = special_feature_seed(isl.rng)
                    else:
                        origin, x = "ordinary_restart", w09.conventional_cross()
                elif isl.rng.random() < .45:
                    origin, x = "mutation", parents[int(isl.rng.integers(len(parents)))].genome.copy()
                else:
                    origin = "block_crossover"
                    x = w09.block_crossover(parents[int(isl.rng.integers(len(parents)))].genome,
                                            isl.population[int(isl.rng.integers(len(isl.population)))].genome, isl.rng)
                if isl.family == "special":
                    x = special_07_repair(x)
                    x = mutate_special_07_family(x, isl.rng, gen / cfg.generations, burst, cfg.burst_mutation_multiplier)
                    x = special_07_repair(x)
                else:
                    x = w09.mutate(x, isl.style, isl.rng, gen / cfg.generations, burst, cfg.burst_mutation_multiplier)
                children.append(MixedIndividual(x, _result_for(x, isl.family, cfg, cache), iid, gen, origin, isl.family))
            next_islands.append(MixedIsland(isl.style, isl.rng, _survivors(isl.population + children, cfg.population), isl.family))
        islands = next_islands
        migration_now = len(islands) > 1 and (burst or (cfg.migration_interval > 0 and gen % cfg.migration_interval == 0))
        if migration_now:
            count = min(12, cfg.population) if burst else min(cfg.migrants, cfg.population)
            outgoing = [sorted(isl.population, key=lambda i: (i.rank, -i.crowding))[:count] for isl in islands]
            for src, migrants in enumerate(outgoing):
                dst = (src + 1) % len(islands)
                target = islands[dst]
                transferred: list[MixedIndividual] = []
                for m in migrants:
                    x = special_07_repair(m.genome) if target.family == "special" else w09.repair(m.genome)
                    # Re-evaluate whenever projection can change geometry; cache key includes family/genome.
                    transferred.append(MixedIndividual(x, _result_for(x, target.family, cfg, cache), dst, gen,
                        f"migration_from_{src}", target.family))
                islands[dst] = MixedIsland(target.style, target.rng, _survivors(target.population + transferred, cfg.population), target.family)
        now = min((_joint(i.result) for isl in islands for i in isl.population if i.valid), default=np.inf)
        if now < best_seen * (1.0 - cfg.improvement_fraction):
            best_seen, stall = now, 0
            if burst:
                burst_remaining = 0
        else:
            stall += 1
            if burst:
                burst_remaining -= 1
        if not burst and stall >= cfg.stall_generations:
            burst_remaining, stall = cfg.burst_generations, 0
            print(f"Stagnation burst scheduled: {cfg.burst_generations} generation(s), starting at G{gen + 1:03d}.", flush=True)
        update_archive(); record(gen, mode)
        _save_csv(history, cfg.output_dir / "history.csv")
        _save_csv(rejects, cfg.output_dir / "rejection_summary.csv")
        _save_csv([_serialize(i) for isl in islands for i in isl.population], cfg.output_dir / f"population_gen_{gen:04d}.csv")
        elites = []
        for isl in islands:
            _rank(isl.population)
            elites.extend(_serialize(i) for i in sorted(isl.population, key=lambda i: (i.rank, -i.crowding))[:cfg.retain_per_island])
        _save_csv(elites, cfg.output_dir / "island_elites" / f"elites_gen_{gen:04d}.csv")
    (cfg.output_dir / "best_live_coarse.json").write_text(json.dumps([_serialize(i) for i in archive.values()], indent=2, allow_nan=True), encoding="utf-8")
    # Ensure contract files exist even when no candidate survives strict fine validation.
    _save_csv([], cfg.output_dir / "fine_validation.csv")
    (cfg.output_dir / "top5_lithography_candidates.json").write_text("[]\n", encoding="utf-8")
    _save_csv([], cfg.output_dir / "top5_lithography_candidates.csv")
    (cfg.output_dir / "verified_hits.json").write_text("[]\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    run(settings_from_args(args))


if __name__ == "__main__":
    main()
