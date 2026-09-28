"""
14_plus_b_spline_v223.py

Wrapper над 14_plus_b_spline_v22.py. Не заменяет, а расширяет.

Что делает:
  Stage 1 (--stage1-generations, default 5):
    * Multi-objective NSGA-II: objectives = [dz, U_centre, U_transition, U_arm]
      (вместо v22 [dz, max_zones(U)]).
    * Diversity threshold (--diversity-threshold, default 0.30) при инициализации.
    * Rebuild ideal mesh для top-1 КАЖДОГО острова каждые --fine-every поколений,
      с overlaid профилями coarse vs ideal.
    * Live plot в отдельном потоке, не блокирует основной цикл.
    * Live виджет островов: по одному субплоту на остров с двумя кривыми.

  Stage 2 (остальные --generations):
    * Возврат к v22 minimax objectives.
    * Всё v22 логирование без изменений.
    * Ideal mesh продолжает строиться для top-1 каждого острова.

Новые файлы в output_dir:
  * stage1_diverse_pool.csv  — top-200 разных кандидатов stage 1
  * diversity_history.csv    — mean/min/max genome distance по поколениям
  * island_ideal_history/    — идеальные профили top-1 per island per trigger
  * live_final_metrics.png / live_final_islands.png

Запуск:
  python workflows/14_plus_b_spline_v223.py \
      --smoke --stage1-generations 2 --diversity-threshold 0.30 \
      --live-plot --plot-every 1 --fine-every 1 \
      --output-dir reports/v223_smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import sys
import threading
import time
from dataclasses import asdict as _asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Импорт v22
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
V22_PATH = HERE / "14_plus_b_spline_v22.py"
if not V22_PATH.exists():
    raise FileNotFoundError(f"v22 not found: {V22_PATH}")

_spec = importlib.util.spec_from_file_location("v22mod", V22_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["v22mod"] = base
_spec.loader.exec_module(base)

# Псевдонимы
Individual = base.Individual
Island = base.Island
Settings = base.Settings
repair = base.repair
genome_distance = base.genome_distance
local_mutation = base.local_mutation
bspline_genome_operator = base.bspline_genome_operator
memetic_candidates = base.memetic_candidates
two_parent_block_mix = base.two_parent_block_mix
blended_block_mix = base.blended_block_mix
seed_from_kind = base.seed_from_kind
evaluate = base.evaluate
rejected = base.rejected
serialize = base.serialize
save_csv = base.save_csv
elite_rows = base.elite_rows
make_island_styles = base.make_island_styles
anneal_scale = base.anneal_scale
discover_valid_baseline = base.discover_valid_baseline
project_to_valid = base.project_to_valid
elite_seeds = base.elite_seeds
zonal_score = base.zonal_score
tournament = base.tournament
plot_convergence = base.plot_convergence
plot_pareto = base.plot_pareto
fine_plots = base.fine_plots

# ---------------------------------------------------------------------------
# 2. Multi-objective режим
# ---------------------------------------------------------------------------
base.MULTI_OBJECTIVE_MODE = False

_ORIG_OBJECTIVES = base.Individual.objectives.fget


def _objectives_v223(self) -> np.ndarray:
    if not self.valid:
        n = 4 if base.MULTI_OBJECTIVE_MODE else 2
        return np.full(n, np.inf, dtype=float)
    if base.MULTI_OBJECTIVE_MODE:
        return np.array(
            [
                self.result["dz_global_ratio_search"],
                self.result["excursion_centre_ev"] / base.SEARCH_EXCURSION_LIMIT_EV,
                self.result["excursion_transition_ev"] / base.SEARCH_EXCURSION_LIMIT_EV,
                self.result["excursion_arm_ev"] / base.SEARCH_EXCURSION_LIMIT_EV,
            ],
            dtype=float,
        )
    return _ORIG_OBJECTIVES(self)


base.Individual.objectives = property(_objectives_v223)


def rank_and_crowding_v223(population: list) -> list:
    if not population:
        return []
    n_obj = len(population[0].objectives)
    count = len(population)
    defeated = [[] for _ in population]
    losses = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(count):
        for j in range(i + 1, count):
            left, right = population[i], population[j]
            if base.dominates(left, right):
                defeated[i].append(j)
                losses[j] += 1
            elif base.dominates(right, left):
                defeated[j].append(i)
                losses[i] += 1

    fronts[0] = [i for i in range(count) if losses[i] == 0]
    while fronts[-1]:
        nxt = []
        for w in fronts[-1]:
            for l in defeated[w]:
                losses[l] -= 1
                if losses[l] == 0:
                    nxt.append(l)
        fronts.append(nxt)
    fronts.pop()

    for rank, front in enumerate(fronts):
        for idx in front:
            population[idx].rank = rank
            population[idx].crowding = 0.0
        feasible = [i for i in front if population[i].search_feasible]
        if len(feasible) <= 2:
            for i in feasible:
                population[i].crowding = np.inf
            continue
        for axis in range(n_obj):
            ordered = sorted(feasible, key=lambda k: population[k].objectives[axis])
            low = population[ordered[0]].objectives[axis]
            high = population[ordered[-1]].objectives[axis]
            if high <= low:
                continue
            population[ordered[0]].crowding = np.inf
            population[ordered[-1]].crowding = np.inf
            for pos in range(1, len(ordered) - 1):
                k = ordered[pos]
                if np.isfinite(population[k].crowding):
                    a = population[ordered[pos - 1]].objectives[axis]
                    b = population[ordered[pos + 1]].objectives[axis]
                    population[k].crowding += (b - a) / (high - low)
    return fronts


base.rank_and_crowding = rank_and_crowding_v223


def survivors_v223(pool, size, emphasis):
    selected = []
    for front in base.rank_and_crowding(pool):
        ordered = sorted(
            front,
            key=lambda idx: (
                not pool[idx].search_feasible,
                pool[idx].violation,
                -pool[idx].crowding,
                base.zonal_score(pool[idx].result, emphasis),
            ),
        )
        selected.extend(pool[idx].copy() for idx in ordered[: size - len(selected)])
        if len(selected) == size:
            break
    return selected


base.survivors = survivors_v223


# ---------------------------------------------------------------------------
# 3. Ideal-mesh rebuild для top-1 каждого острова
# ---------------------------------------------------------------------------
def build_ideal_geometry():
    """
    Вызов вашей же build_mesh(parameters, fast=False) — то, что v22 использует
    для fine-валидации. Возвращает функцию, применяемую к геному.
    """
    def evaluate_ideal(genome: np.ndarray, cfg) -> dict:
        try:
            values = repair(genome)
            param_obj = base.parameters(values)
            report = base.check_x_junction_manufacturability(param_obj)
            if not report.valid:
                return {"valid": False, "reason": "geometry"}

            # Это ровно то, что v22 делает при stage="fine"
            result, data = evaluate(values, cfg, stage="fine", retain=True)
            result = dict(result)
            if data is not None:
                model, trace, pseudo_ev, field, _, zone_data = data
                result["_profile_x_m"] = np.asarray(trace.x_m).tolist()
                result["_profile_u_ev"] = np.asarray(pseudo_ev).tolist()
                result["_u_reference_ev"] = float(zone_data["u_reference_ev"])
                result["_panels"] = int(model.n_panels)
            return result
        except Exception as e:
            return {"valid": False, "reason": f"{type(e).__name__}: {e}"}

    return evaluate_ideal


IDEAL_EVAL = build_ideal_geometry()


# ---------------------------------------------------------------------------
# 4. Live plot в потоке
# ---------------------------------------------------------------------------
class LivePlotter:
    """2 figure: metrics (2x2) + islands (профили top-1 per island)."""

    def __init__(self, enabled: bool, out_dir: Path, n_islands: int):
        self.enabled = enabled
        self.out_dir = out_dir
        self.n_islands = n_islands
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interactive = False

        if not self.enabled:
            return

        try:
            import matplotlib
            for b in ("TkAgg", "Qt5Agg", "Agg"):
                try:
                    matplotlib.use(b, force=True)
                    import matplotlib.pyplot as plt
                    break
                except Exception:
                    continue
            self._plt = plt
            self._interactive = matplotlib.get_backend().lower() != "agg"
        except Exception as e:
            print(f"[v223] matplotlib unavailable: {e}")
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, gen: int, best, pop: list, island_profiles: dict, save: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latest = {
                "gen": gen, "best": best, "pop": pop,
                "island_profiles": island_profiles, "save": save,
            }

    def _run(self) -> None:
        plt = self._plt
        if not self._interactive:
            plt.ioff()

        ncols = min(4, max(1, self.n_islands))
        nrows = math.ceil(self.n_islands / ncols)

        fig1, ax1 = plt.subplots(2, 2, figsize=(13, 8))
        fig1.suptitle("v223 metrics")

        fig2, ax2 = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
            squeeze=False,
        )
        fig2.suptitle("v223 islands — top-1 per island, coarse vs ideal")

        last_drawn = -1
        while not self._stop.is_set():
            with self._lock:
                payload = self._latest
            if payload is None or payload["gen"] == last_drawn:
                time.sleep(0.15)
                continue

            gen = payload["gen"]
            self._history.append({
                "generation": gen,
                "best_U_arm_mev": payload["best"].result.get("excursion_arm_ev", np.nan) * 1e3
                if payload["best"] is not None else np.nan,
                "best_U_transition_mev": payload["best"].result.get("excursion_transition_ev", np.nan) * 1e3
                if payload["best"] is not None else np.nan,
                "best_U_centre_mev": payload["best"].result.get("excursion_centre_ev", np.nan) * 1e3
                if payload["best"] is not None else np.nan,
                "best_dz_um": payload["best"].result.get("dz_peak_m", np.nan) * 1e6
                if payload["best"] is not None else np.nan,
            })
            last_drawn = gen

            try:
                self._draw_metrics(fig1, ax1, gen, payload["pop"], payload["best"])
                self._draw_islands(fig2, ax2, gen, payload["island_profiles"])
            except Exception as e:
                print(f"[v223 live] draw error at gen {gen}: {e}")
                continue

            if not self._interactive:
                if payload["save"]:
                    fig1.savefig(self.out_dir / f"live_gen{gen:04d}_metrics.png", dpi=85)
                    fig2.savefig(self.out_dir / f"live_gen{gen:04d}_islands.png", dpi=85)
            else:
                try:
                    plt.pause(0.005)
                except Exception:
                    pass
            time.sleep(0.03)

        try:
            fig1.savefig(self.out_dir / "live_final_metrics.png", dpi=110)
            fig2.savefig(self.out_dir / "live_final_islands.png", dpi=110)
        except Exception:
            pass
        plt.close(fig1)
        plt.close(fig2)

    def _draw_metrics(self, fig, axes, gen: int, pop: list, best) -> None:
        for ax in axes.flat:
            ax.clear()
        h = self._history
        G = [r["generation"] for r in h]

        ax = axes[0, 0]
        for key, col in (("best_U_arm_mev", "tab:red"),
                         ("best_U_transition_mev", "tab:orange"),
                         ("best_U_centre_mev", "tab:green"),
                         ("best_dz_um", "tab:blue")):
            ax.plot(G, [r[key] for r in h], color=col, label=key.replace("best_", ""))
        ax.set_yscale("log")
        ax.axhline(base.SEARCH_EXCURSION_LIMIT_EV * 1e3, color="gray", ls=":", label="search target")
        ax.set_title("Best metrics over time")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        Uc = [i.result.get("excursion_centre_ev", np.nan) * 1e3 for i in pop if i.valid]
        Ua = [i.result.get("excursion_arm_ev", np.nan) * 1e3 for i in pop if i.valid]
        Ut = [i.result.get("excursion_transition_ev", np.nan) * 1e3 for i in pop if i.valid]
        ax.scatter(Uc, Ua, s=15, alpha=0.55, c="tab:red", label="arm vs centre")
        ax.scatter(Uc, Ut, s=15, alpha=0.35, c="tab:orange", label="transition vs centre")
        ax.set_xlabel("U_centre (meV)")
        ax.set_ylabel("U_arm / U_transition (meV)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("Pareto projection")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        d = []
        sample = pop[: min(40, len(pop))]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d.append(genome_distance(sample[i].genome, sample[j].genome))
        if d:
            ax.hist(d, bins=20, color="tab:purple", alpha=0.7)
            ax.axvline(base._V223_DIVERSITY_THRESHOLD, color="red", ls="--",
                       label=f"target {base._V223_DIVERSITY_THRESHOLD}")
            ax.legend(fontsize=7)
        ax.set_title(f"Genome distance (n={len(d)})")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        # counts
        n_valid = sum(1 for i in pop if i.valid)
        n_feas = sum(1 for i in pop if i.valid and i.search_feasible)
        ax.bar(["valid", "feasible"], [n_valid, n_feas],
               color=["tab:blue", "tab:green"], alpha=0.7)
        ax.set_title(f"Gen {gen}  |  islands={self.n_islands}")
        ax.grid(alpha=0.3)

        fig.tight_layout()

    def _draw_islands(self, fig, axes, gen: int, island_profiles: dict) -> None:
        ncols = axes.shape[1]
        for iid in range(self.n_islands):
            r, c = divmod(iid, ncols)
            ax = axes[r, c]
            ax.clear()
            data = island_profiles.get(iid)
            if data is None:
                ax.set_title(f"Island {iid}: no ideal eval yet", fontsize=8)
                ax.grid(alpha=0.3)
                continue

            coarse = data.get("coarse")
            ideal = data.get("ideal")

            if coarse and "_profile_x_m" in coarse:
                ax.plot(np.asarray(coarse["_profile_x_m"]) * 1e6,
                        (np.asarray(coarse["_profile_u_ev"]) - coarse["_u_reference_ev"]) * 1e3,
                        "--", lw=1.0, color="tab:gray", label="coarse")
            if ideal and "_profile_x_m" in ideal:
                ax.plot(np.asarray(ideal["_profile_x_m"]) * 1e6,
                        (np.asarray(ideal["_profile_u_ev"]) - ideal["_u_reference_ev"]) * 1e3,
                        "-", lw=1.6, color="tab:blue", label="ideal")

            # зоны
            for a, b, c_ in ((-320, -160, "#DDEBFF"), (-160, -45, "#FFF2CC"),
                             (-45, 45, "#FCE4D6"), (45, 160, "#FFF2CC"),
                             (160, 320, "#DDEBFF")):
                ax.axvspan(a, b, color=c_, alpha=0.3, zorder=0)

            ax.axhline(0, color="gray", lw=0.4)
            ax.set_xlabel("x (µm)", fontsize=7)
            ax.set_ylabel("U - U_ref (meV)", fontsize=7)
            ax.tick_params(labelsize=7)
            if ideal:
                ax.set_title(
                    f"Isl {iid} | F_A={ideal.get('excursion_arm_ev', np.nan)*1e3:.1f} "
                    f"F_T={ideal.get('excursion_transition_ev', np.nan)*1e3:.1f} "
                    f"F_C={ideal.get('excursion_centre_ev', np.nan)*1e3:.1f} meV",
                    fontsize=8,
                )
            else:
                ax.set_title(f"Island {iid} | ideal failed", fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.tight_layout()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# 5. Diversity init
# ---------------------------------------------------------------------------
base._V223_DIVERSITY_THRESHOLD = 0.30


def diverse_initial_population(factory, baseline, target_size, rng,
                                min_dist, max_attempts=200):
    pool: list[np.ndarray] = []
    for _ in range(target_size):
        best_cand = None
        best_min_d = -1.0
        for _ in range(max_attempts):
            try:
                cand = factory()
            except Exception:
                continue
            cand, _ = project_to_valid(baseline, cand)
            if pool:
                d_min = min(genome_distance(cand, p) for p in pool)
            else:
                d_min = np.inf
            if d_min >= min_dist:
                best_cand = cand
                break
            if d_min > best_min_d:
                best_min_d = d_min
                best_cand = cand
        pool.append(best_cand if best_cand is not None else baseline.copy())
    return pool


# ---------------------------------------------------------------------------
# 6. Ideal-pass: top-1 per island
# ---------------------------------------------------------------------------
def run_ideal_pass(islands, cfg, coarse_eval):
    """
    Для top-1 каждого острова: coarse-профиль + ideal-профиль.
    Возвращает dict[iid] -> {"coarse": {...}, "ideal": {...}}.
    """
    result = {}
    for isl in islands:
        top1 = min(
            (it for it in isl.population if it.valid),
            key=lambda it: base.zonal_score(it.result, "balanced"),
            default=None,
        )
        if top1 is None:
            continue

        # coarse с retained данными — перезапрашиваем с retain=True
        coarse_res, coarse_data = evaluate(
            top1.genome, cfg, stage="coarse", retain=True)

        coarse_block = None
        if coarse_data is not None:
            _m, trace, pseudo_ev, _f, _p, zone_data = coarse_data
            coarse_block = {
                "valid": True,
                "_profile_x_m": np.asarray(trace.x_m).tolist(),
                "_profile_u_ev": np.asarray(pseudo_ev).tolist(),
                "_u_reference_ev": float(zone_data["u_reference_ev"]),
                "excursion_arm_ev": coarse_res["excursion_arm_ev"],
                "excursion_transition_ev": coarse_res["excursion_transition_ev"],
                "excursion_centre_ev": coarse_res["excursion_centre_ev"],
            }

        ideal_res = IDEAL_EVAL(top1.genome, cfg)

        result[isl.style.name] = {
            "island_id": islands.index(isl),
            "coarse": coarse_block,
            "ideal": ideal_res if ideal_res.get("valid", False) else None,
        }
    return result


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = base.parser()
    p.add_argument("--stage1-generations", type=int, default=5,
                   help="Сколько поколений multi-objective stage 1")
    p.add_argument("--diversity-threshold", type=float, default=0.30,
                   help="Min genome distance при init stage 1")
    p.add_argument("--live-plot", action="store_true",
                   help="Открыть живое окно (TkAgg/Qt5Agg). Иначе PNG-снапшоты.")
    p.add_argument("--plot-every", type=int, default=1,
                   help="Обновлять live-график каждые N поколений")
    p.add_argument("--fine-every", type=int, default=3,
                   help="Ideal mesh для top-1 per island каждые N поколений")
    p.add_argument("--no-stage-split", action="store_true",
                   help="Отключить stage 1 (чистое v22)")
    return p


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if getattr(args, "self_test", False):
        base.run_self_test(args.seed)
        return

    if getattr(args, "smoke", False):
        args.population = 8
        args.offspring = 2
        args.generations = 4
        args.migration_interval = 1
        args.migrants = 1
        args.retain_per_island = 3
        args.coarse_points = 31
        args.fine_points = 61
        args.fine_top = 4
        args.max_panels = 5000
        args.fast = True
        args.animation_stride = 1
        args.surface_points = 61
        args.stall_generations = 2
        args.burst_generations = 1
        args.stage1_generations = 2

    if args.population < 8 or args.offspring < 1 or args.generations < 1:
        raise ValueError("population>=8, offspring>=1, generations>=1")

    backend = base.available_backend(args.backend)
    if not backend.available:
        raise RuntimeError(backend.reason)

    cfg = Settings(
        backend=backend.selected,
        population=args.population,
        offspring=args.offspring,
        generations=args.generations,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
        retain_per_island=args.retain_per_island,
        seed=args.seed,
        coarse_points=args.coarse_points,
        fine_points=args.fine_points,
        fine_top=args.fine_top,
        max_panels=args.max_panels,
        fast=args.fast,
        animation_stride=args.animation_stride,
        surface_points=args.surface_points,
        stall_generations=args.stall_generations,
        burst_generations=args.burst_generations,
        improvement_fraction=args.improvement_fraction,
        burst_mutation_multiplier=args.burst_mutation_multiplier,
        output_dir=args.output_dir,
        memetic_interval=args.memetic_interval,
        memetic_steps=args.memetic_steps,
        memetic_candidates=args.memetic_candidates,
        spline_trace=not args.no_spline_trace,
        spline_trace_anchors=args.spline_trace_anchors,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base._V223_DIVERSITY_THRESHOLD = float(args.diversity_threshold)

    styles = make_island_styles()

    # settings.json
    (cfg.output_dir / "settings.json").write_text(
        json.dumps({
            "settings": _asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg.__dict__,
            "backend": {"selected": backend.selected, "available": backend.available},
            "islands": [_asdict(s) if hasattr(s, "__dataclass_fields__") else s.__dict__
                        for s in styles],
            "v223_flags": {
                "stage1_generations": args.stage1_generations,
                "diversity_threshold": args.diversity_threshold,
                "live_plot": args.live_plot,
                "plot_every": args.plot_every,
                "fine_every": args.fine_every,
            },
        }, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"v223 | backend={backend.selected} | islands={len(styles)}")
    print(f"Stage 1: {args.stage1_generations} gens multi-objective, "
          f"diversity={args.diversity_threshold}, live_plot={args.live_plot}")
    print(f"Stage 2: {args.generations - args.stage1_generations} gens v22 minimax")
    print(f"Ideal mesh for top-1 per island every {args.fine_every} gens")

    master_rng = np.random.default_rng(cfg.seed)
    baseline = discover_valid_baseline(master_rng)

    raw_seeds = elite_seeds()
    seeds: dict[str, np.ndarray] = {}
    for name, raw in raw_seeds.items():
        seeds[name], alpha = project_to_valid(baseline, raw)
        print(f"  Seed {name}: alpha={alpha:.6f}")

    # ---- общий кэш coarse-eval ------------------------------------------------
    cache: dict[tuple, dict] = {}
    cache_lock = threading.Lock()

    def coarse(genome: np.ndarray) -> dict:
        values = repair(genome)
        key = tuple(np.round(values, 13))
        with cache_lock:
            if key in cache:
                return dict(cache[key])
        ok, reason = base.geometry_precheck(values)
        if not ok:
            res = rejected(reason, "coarse", cfg)
        else:
            res, _ = evaluate(values, cfg, stage="coarse", retain=False)
        with cache_lock:
            cache[key] = res
        return dict(res)

    # ---- инициализация -------------------------------------------------------
    islands: list[Island] = []
    init_rng = np.random.default_rng(cfg.seed + 1)

    div = float(args.diversity_threshold) if args.stage1_generations > 0 else base.MIN_GENOME_DISTANCE

    for island_id, style in enumerate(styles):
        rng = np.random.default_rng(int(init_rng.integers(0, 2**32 - 1)))
        label, genome = seed_from_kind(style.seed_kind, seeds, rng)
        genome, _ = project_to_valid(baseline, genome)

        pop = [Individual(
            genome=genome.copy(), result=coarse(genome),
            island=island_id, generation=0, origin=label,
            style_name=style.name, seed_kind=style.seed_kind,
            emphasis=style.emphasis,
        )]

        def factory(style=style, genome=genome, rng=rng):
            if style.operator == "bspline":
                return bspline_genome_operator(genome, rng, style.explore_scale, 0.85)
            if style.operator == "memetic":
                return memetic_candidates(genome, rng, 0.1, 1)[0]
            return local_mutation(genome, style.emphasis, rng,
                                   scale=style.explore_scale, probability=0.85)

        extra = diverse_initial_population(
            factory, baseline, max(0, cfg.population - 1), rng, min_dist=div)

        for g in extra:
            pop.append(Individual(
                genome=g, result=coarse(g),
                island=island_id, generation=0,
                origin=f"initial_diverse_{label}",
                style_name=style.name, seed_kind=style.seed_kind,
                emphasis=style.emphasis,
            ))

        base.rank_and_crowding(pop)
        islands.append(Island(style=style, rng=rng, population=pop))
        if island_id % 4 == 0:
            print(f"  island {island_id}/{len(styles)} seeded ({len(pop)} members)")

    # ---- live plotter --------------------------------------------------------
    plotter = LivePlotter(enabled=bool(args.live_plot),
                          out_dir=cfg.output_dir,
                          n_islands=len(islands))

    # ---- архив ---------------------------------------------------------------
    archive: list[Individual] = []

    def update_archive(items):
        nonlocal archive
        unique = {tuple(np.round(it.genome, 12)): it.copy()
                  for it in archive + list(items) if it.valid}
        archive = list(unique.values())
        base.rank_and_crowding(archive)
        archive.sort(key=lambda it: (
            not it.search_feasible,
            it.violation,
            it.rank,
            -it.crowding,
            zonal_score(it.result, "balanced"),
        ))
        del archive[2000:]

    update_archive([it for isl in islands for it in isl.population])

    history: list[dict] = []
    diversity_rows: list[dict] = []

    def record_diversity(gen: int) -> None:
        allp = [it for isl in islands for it in isl.population]
        if len(allp) < 2:
            return
        sample = allp[: min(60, len(allp))]
        ds = [genome_distance(sample[i].genome, sample[j].genome)
              for i in range(len(sample)) for j in range(i + 1, len(sample))]
        diversity_rows.append({
            "generation": gen,
            "mean_distance": float(np.mean(ds)) if ds else 0.0,
            "min_distance": float(np.min(ds)) if ds else 0.0,
            "max_distance": float(np.max(ds)) if ds else 0.0,
            "threshold": float(args.diversity_threshold),
        })

    def summary(gen: int) -> dict:
        allp = [it for isl in islands for it in isl.population]
        valid = [it for it in allp if it.valid]
        feas = [it for it in valid if it.search_feasible]
        if not valid:
            return {"generation": gen, "valid": 0, "mode": "NA"}
        best = min(valid, key=lambda it: (
            not it.search_feasible, it.violation,
            zonal_score(it.result, "balanced")))
        return {
            "generation": gen,
            "mode": "STAGE1_MULTIOBJ" if base.MULTI_OBJECTIVE_MODE else "STAGE2_MINIMAX",
            "valid": len(valid),
            "search_feasible": len(feas),
            "joint_dz_um": best.result["dz_peak_m"] * 1e6,
            "joint_excursion_mev": best.result["excursion_ev"] * 1e3,
            "joint_dz_centre_um": best.result["dz_centre_peak_m"] * 1e6,
            "joint_dz_transition_um": best.result["dz_transition_peak_m"] * 1e6,
            "joint_dz_arm_um": best.result["dz_arm_peak_m"] * 1e6,
            "joint_U_centre_mev": best.result["excursion_centre_ev"] * 1e3,
            "joint_U_transition_mev": best.result["excursion_transition_ev"] * 1e3,
            "joint_U_arm_mev": best.result["excursion_arm_ev"] * 1e3,
            "joint_score": zonal_score(best.result, "balanced"),
        }

    # ---- главный цикл --------------------------------------------------------
    stage1_end = args.stage1_generations if not args.no_stage_split else 0
    best_global = np.inf
    stall = 0
    burst = 0

    island_profiles: dict = {}

    # первый ideal-pass сразу — чтобы Islands-панель не пустовала
    print("Initial ideal pass (mesh=ideal) on top-1 per island...")
    t0 = time.time()
    island_profiles = run_ideal_pass(islands, cfg, coarse)
    print(f"  done in {time.time()-t0:.1f}s")

    for generation in range(1, cfg.generations + 1):
        t_gen = time.time()
        in_stage1 = generation <= stage1_end
        base.MULTI_OBJECTIVE_MODE = in_stage1
        mode = "STAGE1" if in_stage1 else ("BURST" if burst > 0 else "STAGE2")

        if generation == stage1_end + 1 and stage1_end > 0:
            print(f"--- switch to stage 2 (minimax, v22 logging) at gen {generation} ---")

        # ---- потомки ----
        next_islands: list[Island] = []
        memetic_active = (cfg.memetic_interval > 0
                          and generation % cfg.memetic_interval == 0)

        for island_id, island in enumerate(islands):
            base.rank_and_crowding(island.population)
            elite = sorted(island.population, key=lambda it: (
                not it.search_feasible, it.violation, it.rank, -it.crowding,
                zonal_score(it.result, island.style.emphasis),
            ))[: max(2, min(5, len(island.population)))]

            children: list[Individual] = []
            scale = anneal_scale(island.style, generation / cfg.generations)

            if memetic_active and island.style.operator == "memetic":
                center = elite[0].genome.copy()
                for step in range(cfg.memetic_steps):
                    proposals = memetic_candidates(center, island.rng, 0.5,
                                                    cfg.memetic_candidates)
                    step_children = []
                    for p in proposals:
                        p, _ = project_to_valid(baseline, p)
                        step_children.append(Individual(
                            genome=p, result=coarse(p),
                            island=island_id, generation=generation,
                            origin=f"memetic_s{step}",
                            style_name=island.style.name,
                            seed_kind=island.style.seed_kind,
                            emphasis=island.style.emphasis,
                        ))
                    valid_step = [c for c in step_children if c.valid]
                    if valid_step:
                        w = min(valid_step, key=lambda c: zonal_score(c.result, "balanced"))
                        if zonal_score(w.result, "balanced") < zonal_score(coarse(center), "balanced"):
                            center = w.genome.copy()
                    children.extend(step_children)

            while len(children) < cfg.offspring:
                v = island.rng.random()
                if v < 0.55:
                    parent = elite[0 if island.rng.random() < 0.8 else -1]
                    g = parent.genome.copy(); origin = "elite_local"
                elif v < 0.78:
                    l = tournament(island.population, island.style.emphasis, island.rng)
                    r = tournament(island.population, island.style.emphasis, island.rng)
                    g = blended_block_mix(l.genome, r.genome, island.rng); origin = "blend"
                elif v < 0.92:
                    l = tournament(island.population, island.style.emphasis, island.rng)
                    r = tournament(island.population, island.style.emphasis, island.rng)
                    g = two_parent_block_mix(l.genome, r.genome, island.rng); origin = "block"
                else:
                    lbl, g = seed_from_kind(island.style.seed_kind, seeds, island.rng)
                    origin = f"restart_{lbl}"

                prob = 0.9 if burst > 0 else 0.55
                mult = cfg.burst_mutation_multiplier if burst > 0 else 1.0
                if island.style.operator == "bspline":
                    g = bspline_genome_operator(g, island.rng, scale * mult, prob)
                    origin = "bspline_" + origin
                else:
                    g = local_mutation(g, island.style.emphasis, island.rng,
                                        scale=scale * mult, probability=prob)
                g, _ = project_to_valid(baseline, g)
                children.append(Individual(
                    genome=g, result=coarse(g),
                    island=island_id, generation=generation, origin=origin,
                    style_name=island.style.name,
                    seed_kind=island.style.seed_kind,
                    emphasis=island.style.emphasis,
                ))

            next_islands.append(Island(
                style=island.style, rng=island.rng,
                population=base.survivors(island.population + children,
                                           cfg.population, island.style.emphasis),
            ))

        islands = next_islands

        # ---- миграция ----
        if len(islands) > 1 and (burst > 0 or
                                  (cfg.migration_interval > 0
                                   and generation % cfg.migration_interval == 0)):
            n_mig = min(8 if burst > 0 else cfg.migrants, cfg.population)
            outgoing = []
            for isl in islands:
                base.rank_and_crowding(isl.population)
                outgoing.append(sorted(isl.population, key=lambda it: (
                    not it.search_feasible, it.violation, it.rank, -it.crowding,
                    zonal_score(it.result, isl.style.emphasis),
                ))[:n_mig])
            for src, ms in enumerate(outgoing):
                dst = (src + 1) % len(islands)
                target = islands[dst]
                incoming = [Individual(
                    genome=m.genome.copy(), result=dict(m.result),
                    island=dst, generation=generation,
                    origin=f"mig_from_{src}",
                    style_name=target.style.name,
                    seed_kind=target.style.seed_kind,
                    emphasis=target.style.emphasis,
                ) for m in ms]
                islands[dst] = Island(
                    style=target.style, rng=target.rng,
                    population=base.survivors(
                        target.population + incoming,
                        cfg.population, target.style.emphasis))

        # ---- stall/burst ----
        valid_now = [it for isl in islands for it in isl.population if it.valid]
        cur = min((zonal_score(it.result, "balanced") for it in valid_now),
                  default=np.inf)
        if cur < best_global * (1 - cfg.improvement_fraction):
            best_global = cur; stall = 0
            if burst > 0: burst = 0
        else:
            stall += 1
        if burst > 0:
            burst -= 1
        if burst == 0 and stall >= cfg.stall_generations:
            burst = cfg.burst_generations; stall = 0
            print(f"burst scheduled at gen {generation}")

        # ---- ideal-pass для top-1 per island ----
        if generation % max(1, args.fine_every) == 0:
            t_ideal = time.time()
            island_profiles = run_ideal_pass(islands, cfg, coarse)
            print(f"  ideal pass gen {generation}: {len(island_profiles)} islands "
                  f"in {time.time()-t_ideal:.1f}s")

        # ---- архив, история ----
        update_archive([it for isl in islands for it in isl.population])
        hrow = summary(generation)
        history.append(hrow)
        record_diversity(generation)

        # ---- live plot ----
        allp = [it for isl in islands for it in isl.population if it.valid]
        best_ind = min(allp, key=lambda it: zonal_score(it.result, "balanced")) if allp else None
        if generation % max(1, args.plot_every) == 0:
            plotter.push(generation, best_ind, allp, island_profiles,
                         save=(generation % max(1, args.plot_every) == 0))

        # ---- сохранения CSV (стиль v22 + новое) ----
        save_csv(history, cfg.output_dir / "history.csv")
        save_csv(diversity_rows, cfg.output_dir / "diversity_history.csv")
        save_csv(elite_rows(islands, cfg),
                 cfg.output_dir / "island_elites" / f"elites_gen_{generation:04d}.csv")
        save_csv([serialize(it) for isl in islands for it in isl.population],
                 cfg.output_dir / f"population_gen_{generation:04d}.csv")

        # ---- сохранение ideal-профилей per island ----
        if generation % max(1, args.fine_every) == 0:
            ideal_dir = cfg.output_dir / "island_ideal_history" / f"gen_{generation:04d}"
            ideal_dir.mkdir(parents=True, exist_ok=True)
            for name, data in island_profiles.items():
                (ideal_dir / f"{name}.json").write_text(
                    json.dumps(data, default=float), encoding="utf-8")

        # ---- лог ----
        dt = time.time() - t_gen
        print(
            f"G{generation:03d} [{mode}] "
            f"valid={hrow.get('valid',0)} feas={hrow.get('search_feasible',0)} "
            f"U_c={hrow.get('joint_U_centre_mev', float('nan')):.2f} "
            f"U_t={hrow.get('joint_U_transition_mev', float('nan')):.2f} "
            f"U_a={hrow.get('joint_U_arm_mev', float('nan')):.2f} meV  "
            f"dz={hrow.get('joint_dz_um', float('nan')):.2f} um  [{dt:.1f}s]",
            flush=True,
        )

    plotter.close()

    # ---- stage1 pool ----
    if stage1_end > 0:
        stage1_pool = [it for it in archive if it.generation <= stage1_end]
        stage1_pool.sort(key=lambda it: zonal_score(it.result, "balanced"))
        save_csv([serialize(it) for it in stage1_pool[:200]],
                 cfg.output_dir / "stage1_diverse_pool.csv")
        print(f"Stage-1 pool saved: {len(stage1_pool)} items")

    # ---- оригинальные графики v22 ----
    try:
        plot_convergence(history, cfg.output_dir, cfg)
        plot_pareto(archive, cfg.output_dir, cfg)
    except Exception as e:
        print(f"[v223] plot error: {e}")

    # ---- fine validation (как в v22) ----
    candidates = [it for it in archive if it.valid]
    candidates.sort(key=lambda it: (
        not it.search_feasible, it.violation,
        zonal_score(it.result, "balanced"),
        it.result["dz_peak_m"], it.result["excursion_ev"],
    ))
    selected = candidates[: cfg.fine_top]

    fine_rows = []
    for num, it in enumerate(selected, start=1):
        try:
            result, data = evaluate(it.genome, cfg, stage="fine", retain=True)
        except Exception as e:
            print(f"[v223] fine eval error: {e}")
            continue
        row = {"candidate": num, **serialize(it),
               **{f"fine_{k}": v for k, v in result.items()}}
        fine_rows.append(row)
        if data is not None:
            directory = cfg.output_dir / "fine" / f"candidate_{num:03d}"
            try:
                fine_plots(data, result, directory,
                            f"Fine candidate {num} | {it.style_name}", cfg)
            except Exception as e:
                print(f"[v223] fine_plots error: {e}")

    save_csv(fine_rows, cfg.output_dir / "fine_validation.csv")
    verified = [r for r in fine_rows if r.get("fine_verified", False)]
    search_hits = [r for r in fine_rows if r.get("fine_search_feasible", False)]
    (cfg.output_dir / "verified_hits.json").write_text(
        json.dumps(verified, indent=2, default=float), encoding="utf-8")
    (cfg.output_dir / "search_hits.json").write_text(
        json.dumps(search_hits, indent=2, default=float), encoding="utf-8")

    print(f"\n=== v223 done ===")
    print(f"Archive: {len(archive)} valid")
    print(f"Search-feasible fine hits: {len(search_hits)}")
    print(f"Strict final fine hits: {len(verified)}")


if __name__ == "__main__":
    main()
