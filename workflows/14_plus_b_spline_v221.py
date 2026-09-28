"""
workflows/14_plus_b_spline_v223.py

Stage-1 (fast) + real-time ideal-mesh rebuild for top-1 per island.

Запуск (быстрый smoke, 5-10 мин):
  python workflows/14_plus_b_spline_v223.py \
      --islands 2 --island-pop 30 --offspring 10 \
      --init-pool 400 --archive-bins 10 10 10 \
      --workers 8 --generations 25 --stage1-generations 25 \
      --stage1-mesh coarse --ideal-mesh ideal \
      --fine-every 3 \
      --plot --plot-every 1 \
      --cache-dir /tmp/v223_smoke/cache \
      --output-dir /tmp/v223_smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    # Если нужна живая анимация — замените на TkAgg / Qt5Agg / macosx.
    # Agg = только PNG-снапшоты, нулевое замедление.
    matplotlib.use(os.environ.get("MPL_BACKEND", "Agg"))
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# --- core (совместимо с v22/v221/v222) -------------------------------------
from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from config.numerical import RF_NULL_MAX_HEIGHT_M, RF_NULL_MIN_HEIGHT_M
from core.analysis.barrier import pseudopotential_profile_ev
from core.analysis.rf_null_trace import RFNullTrace, trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem

log = logging.getLogger("v223")

# ===========================================================================
# 1. ГЕНОМ (как в v222)
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

ZONE_WEIGHTS = np.array([1.0, 1.5, 2.0])
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
# 2. ЗОННЫЕ МЕТРИКИ
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
# 3. РАБОЧИЙ ПРОЦЕСС (mesh передаётся в аргументе, а не в initializer)
# ===========================================================================
_WS: Dict = {}


def _worker_init(backend: str):
    _WS["backend"] = backend
    _ = available_backend()


def _downsample(arr: np.ndarray, max_side: int = 128) -> np.ndarray:
    if arr.ndim != 2:
        return arr
    h, w = arr.shape
    if max(h, w) <= max_side:
        return arr
    step = max(h // max_side, w // max_side, 1)
    return arr[::step, ::step]


def _worker_eval(payload):
    """
    payload = (mesh: str, genome_bytes: bytes, want_arrays: bool)
    Возвращает dict.
    """
    mesh, genome_bytes, want_arrays = payload
    genome = np.frombuffer(genome_bytes, dtype=float).copy()
    t0 = time.time()

    geom = build_geometry_aware_quadtree_x_junction_bem(
        genome=genome, mesh_level=mesh, backend=_WS["backend"])
    bem = FixedMeshBEM(geom)
    potential = bem.solve()
    trace: RFNullTrace = trace_rf_transverse_minimum(
        potential, height_range=(RF_NULL_MIN_HEIGHT_M, RF_NULL_MAX_HEIGHT_M))
    x = np.asarray(trace.x)
    prof = np.asarray(pseudopotential_profile_ev(trace, RF_ANGULAR_FREQUENCY_RAD_S))

    arm_len = float(TARGET_ION_HEIGHT_M) * 20.0
    taper_len = genome[IDX_ARM_TAPER] * float(TARGET_ION_HEIGHT_M)
    center_half = 0.5 * float(TARGET_ION_HEIGHT_M) * 8.0
    m = zone_metrics(prof, x, arm_len, taper_len, center_half)

    out = dict(
        F_A=m.F_A, F_B=m.F_B, F_C=m.F_C, dz=m.dz, scalar=m.scalar,
        mesh=mesh, solve_time_s=time.time() - t0,
    )
    if want_arrays:
        out["x"] = x.astype(float)
        out["profile_ev"] = prof.astype(float)
        # Пытаемся достать маску электродов для инсета (может отсутствовать).
        try:
            mask = getattr(geom, "mask", None) or getattr(geom, "electrode_mask", None)
            if mask is not None:
                out["mask"] = _downsample(np.asarray(mask), 128).astype(np.float32)
        except Exception:
            pass
    return out


# ===========================================================================
# 4. ПАРАЛЛЕЛЬНЫЙ EVALUATOR (кэш по (mesh, hash))
# ===========================================================================
class ParallelEvaluator:
    def __init__(self, workers: int, backend: str,
                 cache_path: Path, cache_max_gb: float):
        self.workers = max(1, workers)
        self.backend = backend
        self.cache_path = cache_path
        self.cache_max_bytes = int(cache_max_gb * 1e9)
        self.cache: Dict[bytes, Dict] = {}
        self.n_new = 0
        self._pool: Optional[ProcessPoolExecutor] = None
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                t0 = time.time()
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                log.info("cache loaded: %d entries (%.2f GB) in %.1fs",
                         len(self.cache), self._cache_size() / 1e9, time.time() - t0)
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
            log.info("cache saved: %d entries (%.2f GB) in %.1fs",
                     len(self.cache), self._cache_size() / 1e9, time.time() - t0)
        except Exception as e:
            log.warning("cache save failed: %s", e)

    def _cache_size(self) -> int:
        return len(self.cache) * 96

    def _hash(self, mesh: str, g: np.ndarray) -> bytes:
        h = hashlib.blake2b(np.ascontiguousarray(g).tobytes(), digest_size=16).digest()
        return mesh.encode("ascii") + b"|" + h

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_worker_init,
                initargs=(self.backend,),
            )
        return self._pool

    def evaluate_multi(self, jobs: List[Tuple[str, np.ndarray, bool]]) -> List[Dict]:
        """
        jobs: [(mesh, genome, want_arrays), ...]
        Возвращает list[dict] той же длины и порядка.
        """
        results: List[Optional[Dict]] = [None] * len(jobs)
        to_run: List[Tuple[int, bytes, str, np.ndarray, bool]] = []

        for i, (mesh, g, wa) in enumerate(jobs):
            h = self._hash(mesh, g)
            if (not wa) and h in self.cache:
                results[i] = dict(self.cache[h], mesh=mesh)
                continue
            to_run.append((i, h, mesh, g, wa))

        if to_run:
            pool = self._get_pool()
            calls = [(mesh, g.tobytes(), wa) for _, _, mesh, g, wa in to_run]
            outs = list(pool.map(_worker_eval, calls, chunksize=1))
            for (i, h, mesh, g, wa), out in zip(to_run, outs):
                # кэшируем только метрики
                self.cache[h] = {k: out[k] for k in
                                 ("F_A", "F_B", "F_C", "dz", "scalar")}
                results[i] = out
            self.n_new += len(to_run)

            if self._cache_size() > self.cache_max_bytes:
                keys = list(self.cache.keys())
                for k in keys[: len(keys) // 5]:
                    del self.cache[k]
                log.info("cache trimmed to %d entries", len(self.cache))

        return results  # type: ignore

    def evaluate_batch(self, genomes: List[np.ndarray], mesh: str) -> List[ZoneMetrics]:
        outs = self.evaluate_multi([(mesh, g, False) for g in genomes])
        return [ZoneMetrics(o["F_A"], o["F_B"], o["F_C"], o["dz"]).finalize()
                for o in outs]

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


# ===========================================================================
# 5. ИНДИВИД / ПАРЕТО
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
# 6. MAP-ELITES АРХИВ
# ===========================================================================
class MapElitesArchive:
    def __init__(self, bins: Tuple[int, int, int] = (32, 32, 32)):
        self.bins = bins
        self.grid: Dict[Tuple[int, int, int], Individual] = {}

    def _bin(self, g: np.ndarray) -> Tuple[int, int, int]:
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
# 7. МУТАЦИЯ / КРОССОВЕР
# ===========================================================================
def mutation(g, rng, scale, emphasis="balanced", stall_factor=1.0):
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


def crossover(g1, g2, rng):
    m = rng.random(N_GENES) < 0.5
    return repair(np.where(m, g1, g2)), repair(np.where(m, g2, g1))


# ===========================================================================
# 8. LIVE PLOTTER (потоковый, две фигуры: metrics + islands)
# ===========================================================================
class LivePlotter:
    def __init__(self, enabled: bool, out_dir: Path, n_islands: int):
        self.enabled = enabled and _HAS_MPL
        self.out_dir = out_dir
        self.n_islands = n_islands
        self.history: List[Dict] = []
        self._q: "queue.Queue" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def update(self, gen, best, pop, archive, island_profiles, save):
        self.history.append(dict(
            gen=gen, scalar=best.metrics.scalar,
            F_A=best.metrics.F_A, F_B=best.metrics.F_B, F_C=best.metrics.F_C,
            occ=archive.occupancy(),
        ))
        if not self.enabled:
            return
        snap = dict(
            gen=gen,
            history=list(self.history),
            FA=[i.metrics.F_A for i in pop],
            FC=[i.metrics.F_C for i in pop],
            FB=[i.metrics.F_B for i in pop],
            occ=archive.occupancy(),
            island_profiles=island_profiles,   # dict iid -> {coarse, ideal, F_A...}
            save=save,
        )
        try:
            self._q.put_nowait(snap)
        except queue.Full:
            pass

    def _run(self):
        plt.ion()
        ncols = min(4, max(1, self.n_islands))
        nrows = math.ceil(self.n_islands / ncols)
        fig1, ax1 = plt.subplots(2, 2, figsize=(12, 8))
        fig2, ax2 = plt.subplots(nrows, ncols,
                                 figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        last_save = -1

        while not self._stop.is_set():
            try:
                snap = self._q.get(timeout=0.5)
            except queue.Empty:
                plt.pause(0.01)
                continue

            h = snap["history"]; G = [d["gen"] for d in h]

            # ---- fig1: метрики ----
            for a in ax1.flat:
                a.clear()
            ax1[0, 0].plot(G, [d["scalar"] for d in h], lw=2, label="scalar")
            ax1[0, 0].plot(G, [d["F_A"] for d in h], label="F_A arm")
            ax1[0, 0].plot(G, [d["F_B"] for d in h], label="F_B taper")
            ax1[0, 0].plot(G, [d["F_C"] for d in h], label="F_C center")
            ax1[0, 0].set_yscale("log"); ax1[0, 0].legend(fontsize=8)
            ax1[0, 0].set_title("metrics (eV)"); ax1[0, 0].grid(True, alpha=0.3)

            ax1[0, 1].scatter(snap["FA"], snap["FC"], s=12, alpha=0.5)
            ax1[0, 1].set_xlabel("F_A"); ax1[0, 1].set_ylabel("F_C")
            ax1[0, 1].set_title("Pareto F_A vs F_C")
            ax1[0, 1].grid(True, alpha=0.3)

            ax1[1, 0].plot(G, [d["occ"] * 100 for d in h], color="tab:green")
            ax1[1, 0].set_title(f"Archive occupancy {snap['occ']*100:.2f}%")
            ax1[1, 0].grid(True, alpha=0.3)

            ax1[1, 1].hist(snap["FA"], bins=20, alpha=0.7, color="tab:orange")
            ax1[1, 1].set_title("F_A distribution")
            ax1[1, 1].grid(True, alpha=0.3)

            fig1.tight_layout()

            # ---- fig2: острова ----
            for iid in range(self.n_islands):
                r, c = divmod(iid, ncols)
                ax = ax2[r, c]; ax.clear()
                data = snap["island_profiles"].get(iid)
                if data is None:
                    ax.set_title(f"Island {iid}: no ideal eval yet", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    continue
                coarse = data["coarse"]; ideal = data["ideal"]
                ax.plot(coarse["x"] * 1e6, coarse["profile_ev"] * 1e3, "--",
                        lw=1.2, color="tab:gray",
                        label=f"coarse {coarse['solve_time_s']:.1f}s")
                ax.plot(ideal["x"] * 1e6, ideal["profile_ev"] * 1e3, "-",
                        lw=1.6, color="tab:blue",
                        label=f"ideal  {ideal['solve_time_s']:.1f}s")
                ax.set_xlabel("x (µm)", fontsize=8)
                ax.set_ylabel("U_ps (meV)", fontsize=8)
                ax.set_title(
                    f"Isl {iid} | F_A={data['F_A']*1e3:.1f} "
                    f"F_B={data['F_B']*1e3:.1f} F_C={data['F_C']*1e3:.1f} meV",
                    fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

                # инсет маски идеального решения (если есть)
                if "mask" in ideal:
                    try:
                        ins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
                        ins.imshow(ideal["mask"], cmap="gray", origin="lower",
                                   aspect="auto")
                        ins.set_xticks([]); ins.set_yticks([])
                        ins.set_title("mesh", fontsize=6)
                    except Exception:
                        pass

            fig2.tight_layout()

            if snap["save"] and snap["gen"] != last_save:
                fig1.savefig(self.out_dir / f"live_gen{snap['gen']:05d}_metrics.png", dpi=85)
                fig2.savefig(self.out_dir / f"live_gen{snap['gen']:05d}_islands.png", dpi=85)
                last_save = snap["gen"]

            plt.pause(0.005)

        fig1.savefig(self.out_dir / "live_final_metrics.png", dpi=120)
        fig2.savefig(self.out_dir / "live_final_islands.png", dpi=120)
        plt.ioff()
        plt.close(fig1); plt.close(fig2)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


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
    ideal_mesh: str = "ideal"
    stage2_mesh: str = "fine"
    fine_every: int = 15
    backend: str = "auto"
    seed: int = 20260928
    plot: bool = True
    plot_every: int = 5
    stall_generations: int = 60
    migration_every: int = 20
    migration_size: int = 10
    cache_dir: Path = field(default_factory=lambda: Path("reports/v223/cache"))
    cache_max_gb: float = 120.0
    output_dir: Path = field(default_factory=lambda: Path("reports/v223"))


def diversity_filter(pool, k, min_dist=DIVERSITY_MIN_DIST):
    pool = sorted(pool, key=lambda i: i.metrics.scalar)
    out: List[Individual] = []
    for ind in pool:
        if all(norm_dist(ind.genome, c.genome) >= min_dist for c in out):
            out.append(ind)
        if len(out) >= k:
            break
    return out


def eval_many(genomes: List[np.ndarray], evaluator: ParallelEvaluator,
              mesh: str) -> List[Individual]:
    ms = evaluator.evaluate_batch(genomes, mesh)
    return [Individual(genome=g, metrics=m) for g, m in zip(genomes, ms)]


def run_ideal_pass(
    islands: List["Island"],
    evaluator: ParallelEvaluator,
    ideal_mesh: str,
    stage1_mesh: str,
) -> Dict[int, Dict]:
    """
    Для top-1 каждого острова: параллельно coarse + ideal eval с массивами.
    Возвращает dict[iid] -> {'coarse': ..., 'ideal': ..., 'F_A'/'F_B'/'F_C': ...}
    """
    jobs: List[Tuple[str, np.ndarray, bool]] = []
    order: List[Tuple[int, str]] = []
    for isl in islands:
        top1 = min(isl.population, key=lambda i: i.metrics.scalar)
        jobs.append((stage1_mesh, top1.genome, True))
        order.append((isl.iid, "coarse"))
        jobs.append((ideal_mesh, top1.genome, True))
        order.append((isl.iid, "ideal"))

    outs = evaluator.evaluate_multi(jobs)

    result: Dict[int, Dict] = {}
    for (iid, kind), out in zip(order, outs):
        slot = result.setdefault(iid, {})
        slot[kind] = out
        # метрики берём с ideal (если есть), иначе с coarse
        if kind == "ideal":
            slot["F_A"] = out["F_A"]; slot["F_B"] = out["F_B"]
            slot["F_C"] = out["F_C"]; slot["dz"] = out["dz"]
    # для островов, где ideal не посчитался — фолбэк на coarse
    for iid, slot in result.items():
        if "F_A" not in slot and "coarse" in slot:
            slot["F_A"] = slot["coarse"]["F_A"]
            slot["F_B"] = slot["coarse"]["F_B"]
            slot["F_C"] = slot["coarse"]["F_C"]
            slot["dz"] = slot["coarse"]["dz"]
    return result


def init_pool_fill(n, rng, evaluator, archive, mesh, chunk=512):
    log.info("init pool: %d random genomes on mesh=%s", n, mesh)
    all_ind = []
    t0 = time.time()
    for start in range(0, n, chunk):
        k = min(chunk, n - start)
        gs = [random_genome(rng) for _ in range(k)]
        inds = eval_many(gs, evaluator, mesh)
        for ind in inds:
            archive.add(ind)
        all_ind.extend(inds)
        if (start // chunk) % 5 == 0:
            log.info("init pool: %d/%d (%.1f%%) archive occ %.2f%% %.1fs",
                     start + k, n, 100 * (start + k) / n,
                     archive.occupancy() * 100, time.time() - t0)
    return all_ind


@dataclass
class Island:
    iid: int
    population: List[Individual] = field(default_factory=list)
    stall: int = 0
    best_scalar: float = float("inf")


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
    log.info("=== v223 start ===")
    log.info("config: %s", {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in asdict(cfg).items()})

    backend = cfg.backend if cfg.backend != "auto" else available_backend()
    log.info("backend: %s", backend)

    evaluator = ParallelEvaluator(
        workers=cfg.workers, backend=backend,
        cache_path=cfg.cache_dir / "eval_cache.pkl",
        cache_max_gb=cfg.cache_max_gb,
    )
    archive = MapElitesArchive(bins=cfg.archive_bins)
    plotter = LivePlotter(enabled=cfg.plot, out_dir=cfg.output_dir,
                          n_islands=cfg.islands)

    # --- init pool на coarse -------------------------------------------------
    init_inds = init_pool_fill(cfg.init_pool, rng, evaluator, archive,
                               cfg.stage1_mesh)
    log.info("init pool done: archive %d cells (%.2f%%)",
             len(archive.grid), archive.occupancy() * 100)

    # --- острова -------------------------------------------------------------
    islands: List[Island] = []
    for iid in range(cfg.islands):
        base = rng.choice(len(init_inds))
        g = mutation(init_inds[base].genome, rng, scale=0.15,
                     emphasis=rng.choice(["arm", "center", "balanced"]))
        seed_ind = eval_many([g], evaluator, cfg.stage1_mesh)[0]
        islands.append(Island(iid=iid, population=[seed_ind]))

    for isl in islands:
        while len(isl.population) < cfg.island_pop:
            idx = rng.integers(len(init_inds))
            g = mutation(init_inds[idx].genome, rng, scale=0.05,
                         emphasis=rng.choice(["arm", "center", "balanced"]))
            if all(norm_dist(g, m.genome) >= DIVERSITY_MIN_DIST
                   for m in isl.population):
                isl.population.append(eval_many([g], evaluator, cfg.stage1_mesh)[0])
        isl.best_scalar = min(i.metrics.scalar for i in isl.population)

    log.info("islands seeded: %d x %d", cfg.islands, cfg.island_pop)
    hist: List[Dict] = []
    island_profiles: Dict[int, Dict] = {}
    global_best = min((i for isl in islands for i in isl.population),
                      key=lambda i: i.metrics.scalar)

    # --- первый ideal-проход сразу (чтобы фигура островов не пустовала) ----
    log.info("initial ideal-mesh pass on top-1 per island (mesh=%s)",
             cfg.ideal_mesh)
    t0 = time.time()
    island_profiles = run_ideal_pass(islands, evaluator, cfg.ideal_mesh,
                                     cfg.stage1_mesh)
    log.info("initial ideal pass done in %.1fs", time.time() - t0)

    # --- основной цикл -------------------------------------------------------
    for gen in range(cfg.generations):
        t0 = time.time()
        cur_mesh = cfg.stage1_mesh if gen < cfg.stage1_generations else cfg.stage2_mesh

        # ---- потомки ----
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
                children_g.append(mutation(
                    p.genome, rng, scale=0.08,
                    emphasis=rng.choice(["arm", "center", "balanced"])))

            # novelty bias: сначала пустые ячейки
            children_g.sort(key=lambda g: (0 if archive.is_novel(g) else 1))
            all_children.extend(eval_many(children_g, evaluator, cur_mesh))

        # ---- распределяем по островам ----
        ptr = 0
        for isl in islands:
            chunk = all_children[ptr:ptr + cfg.offspring]
            ptr += cfg.offspring
            pool = isl.population + chunk
            isl.population = select_next_generation(pool, cfg.island_pop)
            isl.population = diversity_filter(isl.population, cfg.island_pop)
            while len(isl.population) < cfg.island_pop:
                g = random_genome(rng)
                isl.population.append(eval_many([g], evaluator, cur_mesh)[0])

            best = min(isl.population, key=lambda i: i.metrics.scalar)
            if best.metrics.scalar < isl.best_scalar - 1e-10:
                isl.best_scalar = best.metrics.scalar
                isl.stall = 0
            else:
                isl.stall += 1

            for ind in isl.population:
                archive.add(ind)

        # ---- миграция ----
        if (gen + 1) % cfg.migration_every == 0 and cfg.islands > 1:
            for k in range(cfg.islands):
                src = islands[k]; dst = islands[(k + 1) % cfg.islands]
                emigrants = sorted(src.population,
                                   key=lambda i: i.metrics.scalar)[: cfg.migration_size]
                combined = dst.population + emigrants
                dst.population = select_next_generation(combined, cfg.island_pop)
            log.info("migration at gen %d", gen)

        # ---- ideal-mesh pass для top-1 каждого острова ---------------------
        if gen > 0 and gen % cfg.fine_every == 0:
            t_ideal = time.time()
            island_profiles = run_ideal_pass(
                islands, evaluator, cfg.ideal_mesh, cur_mesh)
            log.info("ideal pass gen %d: %d islands, %.1fs",
                     gen, len(island_profiles), time.time() - t_ideal)

        # ---- глобальная статистика ----
        allp = [i for isl in islands for i in isl.population]
        best = min(allp, key=lambda i: i.metrics.scalar)
        if best.metrics.scalar < global_best.metrics.scalar:
            global_best = best

        hist.append(dict(
            gen=gen, mesh=cur_mesh,
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

        plotter.update(gen, best, allp, archive, island_profiles,
                       save=(gen % cfg.plot_every == 0))
        if gen % 5 == 0:
            log.info("gen %4d mesh=%s | best=%.4e (g=%.4e) | F_A=%.3e F_B=%.3e F_C=%.3e | occ=%.1f%% | %.1fs",
                     gen, cur_mesh, best.metrics.scalar,
                     global_best.metrics.scalar,
                     best.metrics.F_A, best.metrics.F_B, best.metrics.F_C,
                     archive.occupancy() * 100, time.time() - t0)

        if all(isl.stall >= cfg.stall_generations for isl in islands):
            log.info("all islands stalled at gen %d — early stop", gen)
            break

        if gen % 10 == 9:
            evaluator.save_cache()

    plotter.close()
    evaluator.save_cache()
    evaluator.shutdown()

    # ---- отчёты -------------------------------------------------------------
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
        "best": dict(scalar=best.metrics.scalar, F_A=best.metrics.F_A,
                     F_B=best.metrics.F_B, F_C=best.metrics.F_C,
                     dz=best.metrics.dz, genome=best.genome.tolist()),
        "pareto_size": len(pareto),
        "archive_size": len(archive.grid),
        "archive_occupancy": archive.occupancy(),
        "finalists": [
            dict(scalar=i.metrics.scalar, F_A=i.metrics.F_A, F_B=i.metrics.F_B,
                 F_C=i.metrics.F_C, dz=i.metrics.dz, genome=i.genome.tolist())
            for i in finalists
        ],
        "last_island_profiles": {
            iid: {
                "F_A": d.get("F_A"), "F_B": d.get("F_B"),
                "F_C": d.get("F_C"), "dz": d.get("dz"),
                "coarse_time_s": d["coarse"]["solve_time_s"] if "coarse" in d else None,
                "ideal_time_s":  d["ideal"]["solve_time_s"]  if "ideal"  in d else None,
            }
            for iid, d in island_profiles.items()
        },
    }
    with open(cfg.output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(cfg.output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)

    with open(cfg.output_dir / "archive.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_t", "bin_w", "bin_g", "scalar",
                    "F_A", "F_B", "F_C", "dz", "genome"])
        for k, ind in archive.grid.items():
            w.writerow([k[0], k[1], k[2], ind.metrics.scalar,
                        ind.metrics.F_A, ind.metrics.F_B, ind.metrics.F_C,
                        ind.metrics.dz, json.dumps(ind.genome.tolist())])

    with open(cfg.output_dir / "island_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["island", "pop_size", "best_scalar", "stall"])
        for isl in islands:
            w.writerow([isl.iid, len(isl.population), isl.best_scalar, isl.stall])

    log.info("=== v223 done ===")
    log.info("best scalar=%.4e F_A=%.3e F_B=%.3e F_C=%.3e | pareto=%d archive=%d (%.1f%%) new_evals=%d",
             best.metrics.scalar, best.metrics.F_A, best.metrics.F_B,
             best.metrics.F_C, len(pareto), len(archive.grid),
             archive.occupancy() * 100, evaluator.n_new)
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
    p.add_argument("--ideal-mesh", type=str, default="ideal",
                   help="mesh level, который вы считаете 'ideal'/'fine'")
    p.add_argument("--stage2-mesh", type=str, default="fine")
    p.add_argument("--fine-every", type=int, default=15,
                   help="каждые N поколений строить ideal mesh для top-1 каждого острова")
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--plot-every", type=int, default=5)
    p.add_argument("--stall-generations", type=int, default=60)
    p.add_argument("--migration-every", type=int, default=20)
    p.add_argument("--migration-size", type=int, default=10)
    p.add_argument("--cache-dir", type=Path, default=Path("reports/v223/cache"))
    p.add_argument("--cache-max-gb", type=float, default=120.0)
    p.add_argument("--output-dir", type=Path, default=Path("reports/v223"))
    a = p.parse_args()
    return RunConfig(
        islands=a.islands, island_pop=a.island_pop, offspring=a.offspring,
        init_pool=a.init_pool, archive_bins=tuple(a.archive_bins),
        workers=a.workers, generations=a.generations,
        stage1_generations=a.stage1_generations,
        stage1_mesh=a.stage1_mesh, ideal_mesh=a.ideal_mesh,
        stage2_mesh=a.stage2_mesh, fine_every=a.fine_every,
        backend=a.backend, seed=a.seed, plot=a.plot,
        plot_every=a.plot_every, stall_generations=a.stall_generations,
        migration_every=a.migration_every, migration_size=a.migration_size,
        cache_dir=a.cache_dir, cache_max_gb=a.cache_max_gb,
        output_dir=a.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
