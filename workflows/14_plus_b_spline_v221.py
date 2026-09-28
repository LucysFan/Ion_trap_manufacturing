"""
workflows/14_plus_b_spline_v222.py

Massive-scale multi-island quality-diversity оптимизация X-junction.

Ориентирован на машины с 100+ GB RAM и многоядерными CPU.

Отличия от v221:
  * Параллельная BEM-оценка (multiprocessing, spawn context).
  * Persistent-кэш геном->метрики на диске (pickle, ограничение по размеру).
  * Островная модель: N островов NSGA-II + миграция.
  * MAP-Elites с настраиваемым разрешением (по умолчанию 32x32x32).
  * Стартовый пул геномов (init-pool), чтобы быстро заполнить архив.
  * Novelty bias: пустые ячейки архива приоритетны для мутации.
  * Лёгкая real-time отрисовка (без замедления).
  * Полные отчёты: report.json, history.csv, archive.csv, island_stats.csv.

Запуск (пример на 32 ядрах, 250 GB RAM):
  python workflows/14_plus_b_spline_v222.py \
      --islands 8 --island-pop 200 --offspring 40 \
      --init-pool 20000 \
      --archive-bins 32 32 32 \
      --workers 32 \
      --generations 500 --stall-generations 60 \
      --stage1-generations 60 --stage1-mesh coarse --stage2-mesh fine \
      --cache-dir reports/v222/cache --cache-max-gb 120 \
      --plot --plot-every 10 \
      --output-dir reports/v222
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Импорты core (совместимы с v22/v221)
# ---------------------------------------------------------------------------
from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from config.numerical import RF_NULL_MAX_HEIGHT_M, RF_NULL_MIN_HEIGHT_M
from core.analysis.barrier import pseudopotential_profile_ev
from core.analysis.rf_null_trace import RFNullTrace, trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem

log = logging.getLogger("v222")

# ===========================================================================
# 1. ПРОСТРАНСТВО ГЕНОМА
# ===========================================================================

MIN_GAP_M = 3.0e-6
MIN_RAIL_WIDTH_M = 18.0e-6

N_GAPS, N_INNER, N_OUTER = 8, 4, 4
N_LOCK, N_CENTER_IN, N_CENTER_OUT = 2, 2, 2
N_START, N_LENGTH, N_POWER, N_BULGE = 2, 2, 2, 3

IDX_GAPS        = slice(0, N_GAPS)
IDX_INNER       = slice(N_GAPS, N_GAPS + N_INNER)
IDX_OUTER       = slice(IDX_INNER.stop, IDX_INNER.stop + N_OUTER)
IDX_LOCK        = slice(IDX_OUTER.stop, IDX_OUTER.stop + N_LOCK)
IDX_CENTER_IN   = slice(IDX_LOCK.stop, IDX_LOCK.stop + N_CENTER_IN)
IDX_CENTER_OUT  = slice(IDX_CENTER_IN.stop, IDX_CENTER_IN.stop + N_CENTER_OUT)
IDX_START       = slice(IDX_CENTER_OUT.stop, IDX_CENTER_OUT.stop + N_START)
IDX_LENGTH      = slice(IDX_START.stop, IDX_START.stop + N_LENGTH)
IDX_POWER       = slice(IDX_LENGTH.stop, IDX_LENGTH.stop + N_POWER)
IDX_BULGE       = slice(IDX_POWER.stop, IDX_POWER.stop + N_BULGE)
IDX_ARM_TAPER   = IDX_BULGE.stop
IDX_ARM_WIDTH   = IDX_ARM_TAPER + 1
IDX_ARM_GAP     = IDX_ARM_TAPER + 2
N_GENES         = IDX_ARM_TAPER + 3

LOW = np.zeros(N_GENES); HIGH = np.zeros(N_GENES)
LOW[IDX_GAPS] = -4.0;         HIGH[IDX_GAPS] = 4.0
LOW[IDX_INNER] = 0.0;         HIGH[IDX_INNER] = 20e-6
LOW[IDX_OUTER] = 0.0;         HIGH[IDX_OUTER] = 20e-6
LOW[IDX_LOCK] = -4.0;         HIGH[IDX_LOCK] = 4.0
LOW[IDX_CENTER_IN] = 0.0;     HIGH[IDX_CENTER_IN] = 15e-6
LOW[IDX_CENTER_OUT] = 0.0;    HIGH[IDX_CENTER_OUT] = 15e-6
LOW[IDX_START] = 0.05;        HIGH[IDX_START] = 0.45
LOW[IDX_LENGTH] = 0.10;       HIGH[IDX_LENGTH] = 0.90
LOW[IDX_POWER] = 0.5;         HIGH[IDX_POWER] = 3.0
LOW[IDX_BULGE] = 0.0;         HIGH[IDX_BULGE] = 1.0
LOW[IDX_ARM_TAPER] = 2.0;     HIGH[IDX_ARM_TAPER] = 20.0
LOW[IDX_ARM_WIDTH] = 15e-6;   HIGH[IDX_ARM_WIDTH] = 40e-6
LOW[IDX_ARM_GAP] = 2.0e-6;    HIGH[IDX_ARM_GAP] = 6.0e-6

ZONE_WEIGHTS = np.array([1.0, 1.5, 2.0])   # A, B, C
DIVERSITY_MIN_DIST = 0.30


def repair(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float).copy()
    np.clip(g, LOW, HIGH, out=g)
    g[IDX_ARM_GAP] = max(g[IDX_ARM_GAP], MIN_GAP_M)
    g[IDX_ARM_WIDTH] = max(g[IDX_ARM_WIDTH], MIN_RAIL_WIDTH_M)
    return g


def random_genome(rng: np.random.Generator) -> np.ndarray:
    return repair(rng.uniform(LOW, HIGH))


def norm_dist(g1: np.ndarray, g2: np.ndarray) -> float:
    a = (g1 - LOW) / np.maximum(HIGH - LOW, 1e-12)
    b = (g2 - LOW) / np.maximum(HIGH - LOW, 1e-12)
    return float(np.linalg.norm(a - b))


# ===========================================================================
# 2. МЕТРИКИ ЗОН
# ===========================================================================

@dataclass
class ZoneMetrics:
    F_A: float
    F_B: float
    F_C: float
    dz: float
    scalar: float = 0.0

    def finalize(self) -> "ZoneMetrics":
        self.scalar = float(np.dot(ZONE_WEIGHTS, (self.F_A, self.F_B, self.F_C)))
        return self


def zone_metrics(profile_ev, x, arm_len, taper_len, center_half) -> ZoneMetrics:
    x_arm_end = arm_len - taper_len
    x_taper_end = arm_len + taper_len
    x_center_end = arm_len + taper_len + center_half

    mA = x < x_arm_end
    mB = (x >= x_arm_end) & (x < x_taper_end)
    mC = (x >= x_taper_end) & (x < x_center_end)

    def _e(mask):
        if mask.sum() < 3:
            return 0.0
        v = profile_ev[mask]
        return float(np.max(np.abs(v - v.mean())))

    if mC.sum() > 0:
        idx = np.argmin(profile_ev[mC])
        dz = abs(x[mC][idx] - 0.5 * (x_taper_end + x_center_end))
    else:
        dz = float("inf")

    return ZoneMetrics(_e(mA), _e(mB), _e(mC), dz).finalize()


# ===========================================================================
# 3. ПАРАЛЛЕЛЬНАЯ ОЦЕНКА
# ===========================================================================

_WS: Dict = {}


def _worker_init(backend: str, mesh: str):
    _WS["backend"] = backend
    _WS["mesh"] = mesh
    # прогреваем импорт решателя
    _ = available_backend()


def _worker_eval(genome_bytes: bytes) -> Tuple[float, float, float, float]:
    genome = np.frombuffer(genome_bytes, dtype=float).copy()
    geom = build_geometry_aware_quadtree_x_junction_bem(
        genome=genome, mesh_level=_WS["mesh"], backend=_WS["backend"],
    )
    bem = FixedMeshBEM(geom)
    potential = bem.solve()
    trace: RFNullTrace = trace_rf_transverse_minimum(
        potential, height_range=(RF_NULL_MIN_HEIGHT_M, RF_NULL_MAX_HEIGHT_M))
    x = np.asarray(trace.x)
    prof = pseudopotential_profile_ev(trace, RF_ANGULAR_FREQUENCY_RAD_S)

    arm_len = float(TARGET_ION_HEIGHT_M) * 20.0
    taper_len = genome[IDX_ARM_TAPER] * float(TARGET_ION_HEIGHT_M)
    center_half = 0.5 * float(TARGET_ION_HEIGHT_M) * 8.0
    m = zone_metrics(prof, x, arm_len, taper_len, center_half)
    return (m.F_A, m.F_B, m.F_C, m.dz)


class ParallelEvaluator:
    def __init__(self, workers: int, backend: str, mesh: str,
                 cache_path: Path, cache_max_gb: float):
        self.workers = max(1, workers)
        self.backend = backend
        self.mesh = mesh
        self.cache_path = cache_path
        self.cache_max_bytes = int(cache_max_gb * 1e9)
        self.cache: Dict[bytes, Tuple[float, float, float, float]] = {}
        self.n_new = 0
        self._pool: Optional[ProcessPoolExecutor] = None
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                t0 = time.time()
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                log.info("cache loaded: %d entries in %.1fs (%.2f GB)",
                         len(self.cache), time.time() - t0,
                         self._cache_size() / 1e9)
            except Exception as e:
                log.warning("cache load failed: %s", e)
                self.cache = {}

    def save_cache(self):
        try:
            t0 = time.time()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self.cache_path)
            log.info("cache saved: %d entries, %.2f GB, %.1fs",
                     len(self.cache), self._cache_size() / 1e9, time.time() - t0)
        except Exception as e:
            log.warning("cache save failed: %s", e)

    def _cache_size(self) -> int:
        return len(self.cache) * (16 + 4 * 8)

    def _hash(self, g: np.ndarray) -> bytes:
        return hashlib.blake2b(np.ascontiguousarray(g).tobytes(), digest_size=16).digest()

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_worker_init,
                initargs=(self.backend, self.mesh),
            )
        return self._pool

    def evaluate_batch(self, genomes: List[np.ndarray]) -> List[ZoneMetrics]:
        results: List[Optional[Tuple]] = [None] * len(genomes)
        to_eval: List[Tuple[int, bytes, np.ndarray]] = []
        for i, g in enumerate(genomes):
            h = self._hash(g)
            if h in self.cache:
                results[i] = self.cache[h]
            else:
                to_eval.append((i, h, g))

        if to_eval:
            pool = self._get_pool()
            payload = [g.tobytes() for _, _, g in to_eval]
            outputs = list(pool.map(_worker_eval, payload, chunksize=1))
            for (i, h, g), out in zip(to_eval, outputs):
                self.cache[h] = out
                results[i] = out
            self.n_new += len(to_eval)
            # LRU-ish: если превысили бюджет, обрезаем случайные 20%
            if self._cache_size() > self.cache_max_bytes:
                keys = list(self.cache.keys())
                drop = set(keys[: len(keys) // 5])
                for k in drop:
                    del self.cache[k]
                log.info("cache trimmed to %d entries", len(self.cache))

        return [ZoneMetrics(*r).finalize() for r in results]

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


# ===========================================================================
# 4. ИНДИВИД / ПАРЕТО
# ===========================================================================

@dataclass
class Individual:
    genome: np.ndarray
    metrics: ZoneMetrics
    rank: int = 0
    crowding: float = 0.0
    uid: int = 0

    def key(self):
        return (self.metrics.F_A, self.metrics.F_B, self.metrics.F_C)


def dominates(a: Individual, b: Individual) -> bool:
    ka, kb = a.key(), b.key()
    return all(x <= y for x, y in zip(ka, kb)) and any(x < y for x, y in zip(ka, kb))


def non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    S = [[] for _ in pop]
    n = [0] * len(pop)
    fronts: List[List[int]] = [[]]
    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                S[i].append(j)
            elif dominates(q, p):
                n[i] += 1
        if n[i] == 0:
            p.rank = 0
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [[pop[i] for i in f] for f in fronts[:-1]]


def crowding_distance(front: List[Individual]) -> None:
    for ind in front:
        ind.crowding = 0.0
    for k in range(3):
        front.sort(key=lambda i: i.key()[k])
        front[0].crowding = front[-1].crowding = float("inf")
        fmin, fmax = front[0].key()[k], front[-1].key()[k]
        if fmax - fmin < 1e-15:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].key()[k] - front[i - 1].key()[k]) / (fmax - fmin)


def select_next_generation(pool: List[Individual], n_keep: int) -> List[Individual]:
    fronts = non_dominated_sort(pool)
    out: List[Individual] = []
    for f in fronts:
        crowding_distance(f)
        if len(out) + len(f) <= n_keep:
            out.extend(f)
        else:
            f.sort(key=lambda i: i.crowding, reverse=True)
            out.extend(f[: n_keep - len(out)])
            break
    return out


# ===========================================================================
# 5. MAP-ELITES АРХИВ
# ===========================================================================

class MapElitesArchive:
    def __init__(self, bins: Tuple[int, int, int] = (32, 32, 32)):
        self.bins = bins
        self.grid: Dict[Tuple[int, int, int], Individual] = {}

    def _bin(self, g: np.ndarray) -> Tuple[int, int, int]:
        out = []
        for idx, (b, lo, hi) in enumerate(zip(self.bins,
                                             (LOW[IDX_ARM_TAPER], LOW[IDX_ARM_WIDTH], LOW[IDX_ARM_GAP]),
                                             (HIGH[IDX_ARM_TAPER], HIGH[IDX_ARM_WIDTH], HIGH[IDX_ARM_GAP]))):
            t = (g[idx] - lo) / max(hi - lo, 1e-12)
            out.append(min(int(t * b), b - 1))
        # idx берём из genome через фактические индексы
        vals = [g[IDX_ARM_TAPER], g[IDX_ARM_WIDTH], g[IDX_ARM_GAP]]
        los = [LOW[IDX_ARM_TAPER], LOW[IDX_ARM_WIDTH], LOW[IDX_ARM_GAP]]
        his = [HIGH[IDX_ARM_TAPER], HIGH[IDX_ARM_WIDTH], HIGH[IDX_ARM_GAP]]
        return tuple(
            min(int((v - lo) / max(hi - lo, 1e-12) * b), b - 1)
            for v, lo, hi, b in zip(vals, los, his, self.bins)
        )

    def add(self, ind: Individual) -> bool:
        key = self._bin(ind.genome)
        cur = self.grid.get(key)
        if cur is None or ind.metrics.scalar < cur.metrics.scalar:
            self.grid[key] = ind
            return True
        return False

    def sample(self, k: int, rng: np.random.Generator) -> List[Individual]:
        if not self.grid:
            return []
        items = list(self.grid.values())
        idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
        return [items[i] for i in idx]

    def best(self, n: int = 10) -> List[Individual]:
        return sorted(self.grid.values(), key=lambda i: i.metrics.scalar)[:n]

    def is_novel(self, g: np.ndarray) -> bool:
        return self._bin(g) not in self.grid

    def occupancy(self) -> float:
        total = self.bins[0] * self.bins[1] * self.bins[2]
        return len(self.grid) / total


# ===========================================================================
# 6. ОСТРОВ
# ===========================================================================

@dataclass
class Island:
    iid: int
    population: List[Individual] = field(default_factory=list)
    stall: int = 0
    best_scalar: float = float("inf")


# ===========================================================================
# 7. МУТАЦИЯ / КРОССОВЕР
# ===========================================================================

def mutation(g: np.ndarray, rng: np.random.Generator, scale: float,
             emphasis: str = "balanced", stall_factor: float = 1.0) -> np.ndarray:
    g = g.copy()
    sigma = (HIGH - LOW) * scale * stall_factor
    if emphasis == "arm":
        sigma[IDX_ARM_TAPER] *= 2.5; sigma[IDX_ARM_WIDTH] *= 2.0
        sigma[IDX_ARM_GAP] *= 2.0; sigma[IDX_START] *= 2.0
        sigma[IDX_LENGTH] *= 2.5; sigma[IDX_POWER] *= 2.0
    elif emphasis == "center":
        sigma[IDX_LOCK] *= 2.5; sigma[IDX_CENTER_IN] *= 2.5
        sigma[IDX_CENTER_OUT] *= 2.5; sigma[IDX_BULGE] *= 2.0
        sigma[IDX_GAPS] *= 1.5
    mask = rng.random(N_GENES) < 0.55
    g[mask] += rng.normal(0.0, sigma[mask])
    return repair(g)


def crossover(g1: np.ndarray, g2: np.ndarray, rng: np.random.Generator):
    m = rng.random(N_GENES) < 0.5
    return repair(np.where(m, g1, g2)), repair(np.where(m, g2, g1))


# ===========================================================================
# 8. LIVE PLOT
# ===========================================================================

class LivePlotter:
    def __init__(self, enabled: bool, out_dir: Path):
        self.enabled = enabled and _HAS_MPL
        self.out_dir = out_dir
        self.history: List[Dict] = []
        if self.enabled:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))

    def update(self, gen, best, pop, archive: MapElitesArchive, save: bool):
        self.history.append(dict(
            gen=gen, scalar=best.metrics.scalar,
            F_A=best.metrics.F_A, F_B=best.metrics.F_B, F_C=best.metrics.F_C,
            occ=archive.occupancy(), cache=len(archive.grid),
        ))
        if not self.enabled:
            return
        for a in self.ax.flat:
            a.clear()
        h = self.history
        G = [d["gen"] for d in h]
        self.ax[0, 0].plot(G, [d["scalar"] for d in h], lw=2, label="scalar")
        self.ax[0, 0].plot(G, [d["F_A"] for d in h], label="F_A arm")
        self.ax[0, 0].plot(G, [d["F_B"] for d in h], label="F_B taper")
        self.ax[0, 0].plot(G, [d["F_C"] for d in h], label="F_C center")
        self.ax[0, 0].set_yscale("log"); self.ax[0, 0].legend(fontsize=8)
        self.ax[0, 0].set_title("metrics (eV)"); self.ax[0, 0].grid(True, alpha=0.3)

        FA = [i.metrics.F_A for i in pop]; FC = [i.metrics.F_C for i in pop]
        self.ax[0, 1].scatter(FA, FC, s=15, alpha=0.5)
        self.ax[0, 1].set_xlabel("F_A"); self.ax[0, 1].set_ylabel("F_C")
        self.ax[0, 1].set_title("Pareto projection"); self.ax[0, 1].grid(True, alpha=0.3)

        self.ax[1, 0].plot(G, [d["occ"] * 100 for d in h], color="tab:green")
        self.ax[1, 0].set_title("Archive occupancy (%)")
        self.ax[1, 0].grid(True, alpha=0.3)

        d = []
        for i in range(min(60, len(pop))):
            for j in range(i + 1, min(60, len(pop))):
                d.append(norm_dist(pop[i].genome, pop[j].genome))
        if d:
            self.ax[1, 1].hist(d, bins=25, alpha=0.7, color="tab:orange")
            self.ax[1, 1].axvline(DIVERSITY_MIN_DIST, color="r", ls="--")
        self.ax[1, 1].set_title("genome distance"); self.ax[1, 1].grid(True, alpha=0.3)

        self.fig.tight_layout()
        if save:
            self.fig.savefig(self.out_dir / f"live_gen{gen:05d}.png", dpi=90)
        plt.pause(0.005)

    def close(self):
        if self.enabled:
            self.fig.savefig(self.out_dir / "live_final.png", dpi=120)
            plt.ioff(); plt.close(self.fig)


# ===========================================================================
# 9. ОСНОВНОЙ ЦИКЛ
# ===========================================================================

@dataclass
class RunConfig:
    islands: int = 8
    island_pop: int = 200
    offspring: int = 40
    init_pool: int = 20000
    archive_bins: Tuple[int, int, int] = (32, 32, 32)
    workers: int = 32
    generations: int = 500
    stage1_generations: int = 60
    stage1_mesh: str = "coarse"
    stage2_mesh: str = "fine"
    backend: str = "auto"
    seed: int = 20260928
    plot: bool = True
    plot_every: int = 10
    stall_generations: int = 60
    migration_every: int = 20
    migration_size: int = 10
    cache_dir: Path = field(default_factory=lambda: Path("reports/v222/cache"))
    cache_max_gb: float = 120.0
    output_dir: Path = field(default_factory=lambda: Path("reports/v222"))


def diversity_filter(pool: List[Individual], k: int,
                     min_dist: float = DIVERSITY_MIN_DIST) -> List[Individual]:
    pool = sorted(pool, key=lambda i: i.metrics.scalar)
    out: List[Individual] = []
    for ind in pool:
        if all(norm_dist(ind.genome, c.genome) >= min_dist for c in out):
            out.append(ind)
        if len(out) >= k:
            break
    return out


def make_individual(genome: np.ndarray, evaluator: ParallelEvaluator) -> Individual:
    m = evaluator.evaluate_batch([genome])[0]
    return Individual(genome=genome, metrics=m)


def eval_many(genomes: List[np.ndarray], evaluator: ParallelEvaluator) -> List[Individual]:
    ms = evaluator.evaluate_batch(genomes)
    return [Individual(genome=g, metrics=m) for g, m in zip(genomes, ms)]


def init_pool_fill(n: int, rng: np.random.Generator, evaluator: ParallelEvaluator,
                   archive: MapElitesArchive, chunk: int = 512) -> List[Individual]:
    log.info("init pool: generating %d random genomes", n)
    all_ind: List[Individual] = []
    t0 = time.time()
    for start in range(0, n, chunk):
        k = min(chunk, n - start)
        gs = [random_genome(rng) for _ in range(k)]
        inds = eval_many(gs, evaluator)
        for ind in inds:
            archive.add(ind)
        all_ind.extend(inds)
        if (start // chunk) % 5 == 0:
            log.info("init pool: %d/%d, archive occ %.2f%%, %.1fs",
                     start + k, n, archive.occupancy() * 100, time.time() - t0)
    return all_ind


def run(cfg: RunConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.FileHandler(cfg.output_dir / "run.log"),
                  logging.StreamHandler()],
        force=True,
    )
    log.info("=== v222 start ===")
    log.info("config: %s", {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in asdict(cfg).items()})

    backend = cfg.backend if cfg.backend != "auto" else available_backend()
    log.info("backend: %s", backend)

    evaluator = ParallelEvaluator(
        workers=cfg.workers, backend=backend,
        mesh=cfg.stage1_mesh,
        cache_path=cfg.cache_dir / "eval_cache.pkl",
        cache_max_gb=cfg.cache_max_gb,
    )
    archive = MapElitesArchive(bins=cfg.archive_bins)
    plotter = LivePlotter(enabled=cfg.plot, out_dir=cfg.output_dir)

    # --- Стартовый пул ------------------------------------------------------
    init_inds = init_pool_fill(cfg.init_pool, rng, evaluator, archive)
    log.info("init pool done: %d evals, archive %d cells (%.2f%%)",
             evaluator.n_new, len(archive.grid), archive.occupancy() * 100)

    # --- Формирование островов ---------------------------------------------
    islands: List[Island] = []
    for iid in range(cfg.islands):
        # каждая особь = стартовый геном + мутация, чтобы острова различались
        base = rng.choice(len(init_inds))
        g = mutation(init_inds[base].genome, rng, scale=0.15,
                     emphasis=rng.choice(["arm", "center", "balanced"]))
        seed_ind = make_individual(g, evaluator)
        islands.append(Island(iid=iid, population=[seed_ind]))

    # добьём острова до island_pop, используя разнообразие
    all_seed = init_inds[:]
    for isl in islands:
        while len(isl.population) < cfg.island_pop:
            idx = rng.integers(len(all_seed))
            g = mutation(all_seed[idx].genome, rng, scale=0.05,
                         emphasis=rng.choice(["arm", "center", "balanced"]))
            if all(norm_dist(g, m.genome) >= DIVERSITY_MIN_DIST for m in isl.population):
                isl.population.append(make_individual(g, evaluator))
        isl.best_scalar = min(i.metrics.scalar for i in isl.population)

    log.info("islands seeded: %d x %d", cfg.islands, cfg.island_pop)
    hist: List[Dict] = []
    global_best = min(
        (i for isl in islands for i in isl.population),
        key=lambda i: i.metrics.scalar)

    # --- Основной цикл ------------------------------------------------------
    for gen in range(cfg.generations):
        t0 = time.time()

        # переключение mesh на stage2
        if gen == cfg.stage1_generations and evaluator.mesh != cfg.stage2_mesh:
            evaluator.mesh = cfg.stage2_mesh
            if evaluator._pool is not None:
                evaluator._pool.shutdown(wait=True)
                evaluator._pool = None
            log.info("switch to stage2 mesh=%s at gen=%d", cfg.stage2_mesh, gen)

        # ---- генерация потомков для всех островов ------------------------
        all_children: List[Individual] = []
        for isl in islands:
            children_g: List[np.ndarray] = []
            n_from_archive = max(1, cfg.offspring // 3)
            arch_parents = archive.sample(n_from_archive, rng)
            n_cross = cfg.offspring - n_from_archive

            for _ in range(n_cross):
                a, b = rng.choice(len(isl.population), size=2, replace=False)
                p1, p2 = isl.population[a], isl.population[b]
                if norm_dist(p1.genome, p2.genome) < DIVERSITY_MIN_DIST:
                    p2 = rng.choice(isl.population)
                c1, c2 = crossover(p1.genome, p2.genome, rng)
                pick = c1 if rng.random() < 0.5 else c2
                pick = mutation(pick, rng, scale=0.05,
                                emphasis=rng.choice(["arm", "center", "balanced"]),
                                stall_factor=1.0 + 0.3 * min(isl.stall, 10))
                children_g.append(pick)

            for p in arch_parents:
                # Novelty bias: если геном из архива даёт пустую ячейку — приоритет
                g = mutation(p.genome, rng, scale=0.08,
                             emphasis=rng.choice(["arm", "center", "balanced"]))
                children_g.append(g)

            # приоритет пустым ячейкам
            children_g.sort(key=lambda g: (0 if archive.is_novel(g) else 1))
            all_children.extend(eval_many(children_g, evaluator))

        # ---- распределяем потомков по островам ---------------------------
        ptr = 0
        for isl in islands:
            chunk = all_children[ptr:ptr + cfg.offspring]
            ptr += cfg.offspring
            pool = isl.population + chunk
            isl.population = select_next_generation(pool, cfg.island_pop)
            isl.population = diversity_filter(isl.population, cfg.island_pop)
            while len(isl.population) < cfg.island_pop:
                g = random_genome(rng)
                isl.population.append(make_individual(g, evaluator))

            best = min(isl.population, key=lambda i: i.metrics.scalar)
            if best.metrics.scalar < isl.best_scalar - 1e-10:
                isl.best_scalar = best.metrics.scalar
                isl.stall = 0
            else:
                isl.stall += 1

            for ind in isl.population:
                archive.add(ind)

        # ---- миграция ----------------------------------------------------
        if (gen + 1) % cfg.migration_every == 0 and cfg.islands > 1:
            for k in range(cfg.islands):
                src = islands[k]
                dst = islands[(k + 1) % cfg.islands]
                emigrants = sorted(src.population,
                                   key=lambda i: i.metrics.scalar)[: cfg.migration_size]
                combined = dst.population + emigrants
                dst.population = select_next_generation(combined, cfg.island_pop)
            log.info("migration at gen %d", gen)

        # ---- глобальная статистика ---------------------------------------
        allp = [i for isl in islands for i in isl.population]
        best = min(allp, key=lambda i: i.metrics.scalar)
        if best.metrics.scalar < global_best.metrics.scalar:
            global_best = best

        hist.append(dict(
            gen=gen, stage=(1 if gen < cfg.stage1_generations else 2),
            best_scalar=best.metrics.scalar,
            best_F_A=best.metrics.F_A, best_F_B=best.metrics.F_B,
            best_F_C=best.metrics.F_C, best_dz=best.metrics.dz,
            global_best=global_best.metrics.scalar,
            archive_cells=len(archive.grid),
            archive_occ=archive.occupancy(),
            n_new_evals=evaluator.n_new,
            n_cache=len(evaluator.cache),
            t=time.time() - t0,
        ))

        if gen % max(1, cfg.plot_every // 2) == 0:
            plotter.update(gen, best, allp, archive,
                           save=(gen % cfg.plot_every == 0))
        if gen % 5 == 0:
            log.info("gen %4d | best=%.4e (global=%.4e) | F_A=%.3e F_B=%.3e F_C=%.3e | occ=%.1f%% | cache=%d | %.1fs",
                     gen, best.metrics.scalar, global_best.metrics.scalar,
                     best.metrics.F_A, best.metrics.F_B, best.metrics.F_C,
                     archive.occupancy() * 100, len(evaluator.cache),
                     time.time() - t0)

        # ---- стоп по стагнации ------------------------------------------
        if all(isl.stall >= cfg.stall_generations for isl in islands):
            log.info("all islands stalled at gen %d — early stop", gen)
            break

        # ---- периодический сейв кэша ------------------------------------
        if gen % 10 == 9:
            evaluator.save_cache()

    plotter.close()
    evaluator.save_cache()
    evaluator.shutdown()

    # --- Отчётность ---------------------------------------------------------
    allp = [i for isl in islands for i in isl.population]
    pareto = non_dominated_sort(allp)[0] if allp else []
    finalists = diversity_filter(pareto, min(20, len(pareto)))
    best = min(allp, key=lambda i: i.metrics.scalar)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(cfg).items()},
        "backend": backend,
        "n_new_evals": evaluator.n_new,
        "n_cache_entries": len(evaluator.cache),
        "best": dict(
            scalar=best.metrics.scalar, F_A=best.metrics.F_A,
            F_B=best.metrics.F_B, F_C=best.metrics.F_C, dz=best.metrics.dz,
            genome=best.genome.tolist(),
        ),
        "pareto_size": len(pareto),
        "archive_size": len(archive.grid),
        "archive_occupancy": archive.occupancy(),
        "finalists": [
            dict(scalar=i.metrics.scalar, F_A=i.metrics.F_A, F_B=i.metrics.F_B,
                 F_C=i.metrics.F_C, dz=i.metrics.dz, genome=i.genome.tolist())
            for i in finalists
        ],
    }
    with open(cfg.output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(cfg.output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)

    with open(cfg.output_dir / "archive.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_t", "bin_w", "bin_g", "scalar", "F_A", "F_B", "F_C", "dz", "genome"])
        for k, ind in archive.grid.items():
            w.writerow([k[0], k[1], k[2], ind.metrics.scalar,
                        ind.metrics.F_A, ind.metrics.F_B, ind.metrics.F_C,
                        ind.metrics.dz, json.dumps(ind.genome.tolist())])

    with open(cfg.output_dir / "island_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["island", "pop_size", "best_scalar", "stall"])
        for isl in islands:
            w.writerow([isl.iid, len(isl.population), isl.best_scalar, isl.stall])

    log.info("=== v222 done ===")
    log.info("best scalar=%.4e  F_A=%.3e F_B=%.3e F_C=%.3e  pareto=%d  archive=%d (%.1f%%)  new_evals=%d",
             best.metrics.scalar, best.metrics.F_A, best.metrics.F_B, best.metrics.F_C,
             len(pareto), len(archive.grid), archive.occupancy() * 100, evaluator.n_new)
    return report


# ===========================================================================
# 10. CLI
# ===========================================================================

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--islands", type=int, default=8)
    p.add_argument("--island-pop", type=int, default=200)
    p.add_argument("--offspring", type=int, default=40)
    p.add_argument("--init-pool", type=int, default=20000)
    p.add_argument("--archive-bins", type=int, nargs=3, default=(32, 32, 32))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--stage1-generations", type=int, default=60)
    p.add_argument("--stage1-mesh", type=str, default="coarse")
    p.add_argument("--stage2-mesh", type=str, default="fine")
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--stall-generations", type=int, default=60)
    p.add_argument("--migration-every", type=int, default=20)
    p.add_argument("--migration-size", type=int, default=10)
    p.add_argument("--cache-dir", type=Path, default=Path("reports/v222/cache"))
    p.add_argument("--cache-max-gb", type=float, default=120.0)
    p.add_argument("--output-dir", type=Path, default=Path("reports/v222"))
    a = p.parse_args()
    return RunConfig(
        islands=a.islands, island_pop=a.island_pop, offspring=a.offspring,
        init_pool=a.init_pool, archive_bins=tuple(a.archive_bins),
        workers=a.workers, generations=a.generations,
        stage1_generations=a.stage1_generations,
        stage1_mesh=a.stage1_mesh, stage2_mesh=a.stage2_mesh,
        backend=a.backend, seed=a.seed, plot=a.plot,
        plot_every=a.plot_every, stall_generations=a.stall_generations,
        migration_every=a.migration_every, migration_size=a.migration_size,
        cache_dir=a.cache_dir, cache_max_gb=a.cache_max_gb,
        output_dir=a.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
