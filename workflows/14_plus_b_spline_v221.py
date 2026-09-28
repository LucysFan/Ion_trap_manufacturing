"""
14_plus_b_spline_v223.py

Двухстадийная многоцелевая оптимизация X-junction на базе v22.

Stage 1 (fast exploration, --stage1-generations):
  * MULTI_OBJECTIVE_MODE = True → objectives = [dz, U_centre, U_transition, U_arm]
    Это даёт Парето-фронт, размазанный по трём зонам (никакая зона не доминирует).
  * Diversity threshold (--diversity-threshold, default 0.30) принудительно
    разносит стартовую популяцию в геном-пространстве.
  * Live plot в отдельном потоке (--live-plot) — не тормозит основной цикл.
  * Результат: пул из ~N разных, но не идеальных кандидатов.

Stage 2 (refinement, остальные --generations):
  * MULTI_OBJECTIVE_MODE = False → возврат к оригинальному v22 поведению
    с 2 целями и minimax по зонам.
  * Всё логирование v22 сохраняется без изменений.
  * Плюс новое: история diversity и stage1_diverse_pool.csv.

Требует: 14_plus_b_spline_v22.py в той же директории.

Запуск:
  python workflows/14_plus_b_spline_v223.py \\
      --stage1-generations 5 \\
      --stage1-population-mult 1.5 \\
      --diversity-threshold 0.30 \\
      --live-plot --plot-every 1 \\
      --population 12 --offspring 4 --generations 20 \\
      --fast --output-dir reports/v223_probe
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Импорт v22 как модуля
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
V22_PATH = HERE / "14_plus_b_spline_v22.py"
if not V22_PATH.exists():
    raise FileNotFoundError(f"14_plus_b_spline_v22.py not found at {V22_PATH}")

_spec = importlib.util.spec_from_file_location("v22mod", V22_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["v22mod"] = base
_spec.loader.exec_module(base)

# Псевдонимы для удобства
Individual = base.Individual
Island = base.Island
IslandStyle = base.IslandStyle
Settings = base.Settings
repair = base.repair
decode = base.decode
genome_distance = base.genome_distance
local_mutation = base.local_mutation
bspline_genome_operator = base.bspline_genome_operator
memetic_candidates = base.memetic_candidates
two_parent_block_mix = base.two_parent_block_mix
blended_block_mix = base.blended_block_mix
three_parent_block_mix = base.three_parent_block_mix
seed_from_kind = base.seed_from_kind
unique_valid_candidate = base.unique_valid_candidate
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
plot_convergence = base.plot_convergence
plot_pareto = base.plot_pareto
animate_surfaces = base.animate_surfaces
fine_plots = base.fine_plots
tournament = base.tournament

# ---------------------------------------------------------------------------
# 1. МНОГОЦЕЛЕВОЙ РЕЖИМ
# ---------------------------------------------------------------------------
# base.MULTI_OBJECTIVE_MODE переключается между стадиями.
base.MULTI_OBJECTIVE_MODE = False

_ORIG_OBJECTIVES = base.Individual.objectives.fget


def _objectives_v223(self) -> np.ndarray:
    if not self.valid:
        n = 4 if base.MULTI_OBJECTIVE_MODE else 2
        return np.full(n, np.inf, dtype=float)

    if base.MULTI_OBJECTIVE_MODE:
        # 4 цели: [dz, U_centre, U_transition, U_arm] — все нормированы на search-limit
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


# ---------------------------------------------------------------------------
# 2. RANK_AND_CROWDING с N целями
# ---------------------------------------------------------------------------
def rank_and_crowding_v223(population: list) -> list:
    """Копия v22, но range(n_obj) вместо range(2)."""
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


# ---------------------------------------------------------------------------
# 3. SURVIVORS с N целями (копия v22; rank_and_crowding уже заменён)
# ---------------------------------------------------------------------------
def survivors_v223(pool: list, size: int, emphasis) -> list:
    selected: list = []
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
# 4. LIVE PLOTTER (в отдельном потоке, не тормозит)
# ---------------------------------------------------------------------------
class LivePlotter:
    """4 панели: лучший профиль U, Парето, зональные excursion, diversity."""

    def __init__(self, enabled: bool, out_dir: Path):
        self.enabled = enabled
        self.out_dir = out_dir
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if not self.enabled:
            return

        try:
            import matplotlib
            # Попробуем интерактивный backend; если нет — Agg (только PNG)
            for backend in ("TkAgg", "Qt5Agg", "Agg"):
                try:
                    matplotlib.use(backend, force=True)
                    import matplotlib.pyplot as plt
                    break
                except Exception:
                    continue
            self._plt = plt
            self._interactive = matplotlib.get_backend().lower() not in ("agg",)
        except Exception as e:
            print(f"[v223] matplotlib unavailable: {e}")
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(
        self,
        generation: int,
        best_individual,
        population: list,
        history_row: dict | None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "generation": generation,
            "best": best_individual,
            "population": population,
            "history_row": history_row,
        }
        with self._lock:
            self._latest = payload

    def _run(self) -> None:
        plt = self._plt
        if not self._interactive:
            plt.ioff()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("v223 live — stage 1 (multi-objective)")

        last_drawn = -1
        while not self._stop.is_set():
            with self._lock:
                payload = self._latest
            if payload is None or payload["generation"] == last_drawn:
                time.sleep(0.15)
                continue

            gen = payload["generation"]
            best = payload["best"]
            pop = payload["population"]
            row = payload["history_row"]
            last_drawn = gen

            if row is not None:
                self._history.append(row)

            try:
                self._draw(fig, axes, gen, best, pop)
            except Exception as e:
                print(f"[v223 live] draw error at gen {gen}: {e}")
                continue

            if not self._interactive:
                # сохраняем снапшот раз в 5 поколений
                if gen % 5 == 0:
                    fig.savefig(self.out_dir / f"live_gen{gen:04d}.png", dpi=85)
            else:
                try:
                    plt.pause(0.01)
                except Exception:
                    pass
            time.sleep(0.05)

        try:
            fig.savefig(self.out_dir / "live_final.png", dpi=110)
        except Exception:
            pass
        plt.close(fig)

    def _draw(self, fig, axes, gen: int, best, pop: list) -> None:
        for ax in axes.flat:
            ax.clear()

        # ---- panel 1: лучший профиль U ----
        ax = axes[0, 0]
        self._draw_profile(ax, best)
        ax.set_title(f"Best profile | gen {gen} | "
                     f"F_A={best.result.get('excursion_arm_ev', np.nan)*1e3:.2f} "
                     f"F_T={best.result.get('excursion_transition_ev', np.nan)*1e3:.2f} "
                     f"F_C={best.result.get('excursion_centre_ev', np.nan)*1e3:.2f} meV",
                     fontsize=9)

        # ---- panel 2: Парето в (dz, excursion) ----
        ax = axes[0, 1]
        FAs = [i.result.get("excursion_arm_ev", np.nan) * 1e3 for i in pop if i.valid]
        FCs = [i.result.get("excursion_centre_ev", np.nan) * 1e3 for i in pop if i.valid]
        ax.scatter(FCs, FAs, s=18, alpha=0.55, c="tab:blue")
        ax.set_xlabel("U_centre (meV)")
        ax.set_ylabel("U_arm (meV)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("Pareto: arm vs centre")
        ax.grid(alpha=0.3)

        # ---- panel 3: зональные excursion по поколениям ----
        ax = axes[1, 0]
        if self._history:
            gens = [h["generation"] for h in self._history]
            for key, color in (("joint_U_arm_mev", "tab:red"),
                               ("joint_U_transition_mev", "tab:orange"),
                               ("joint_U_centre_mev", "tab:green")):
                ys = [h.get(key, np.nan) for h in self._history]
                ax.plot(gens, ys, color=color, label=key.replace("joint_U_", ""))
            ax.set_yscale("log")
            ax.axhline(base.SEARCH_EXCURSION_LIMIT_EV * 1e3,
                       color="gray", ls=":", label="search target")
            ax.set_xlabel("generation"); ax.set_ylabel("U (meV)")
            ax.set_title("Zone excursions over time")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ---- panel 4: diversity ----
        ax = axes[1, 1]
        d = []
        sample = pop[: min(40, len(pop))]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d.append(genome_distance(sample[i].genome, sample[j].genome))
        if d:
            ax.hist(d, bins=20, color="tab:purple", alpha=0.7)
            ax.axvline(base._V223_DIVERSITY_THRESHOLD, color="red", ls="--",
                       label=f"target {base._V223_DIVERSITY_THRESHOLD}")
            ax.legend(fontsize=8)
        ax.set_title(f"Genome distance (n={len(d)})")
        ax.grid(alpha=0.3)

        fig.tight_layout()

    def _draw_profile(self, ax, ind) -> None:
        # у Individual нет сохранённого профиля, но результат содержит x_m, y_m, z_m
        # если не сохранены — просто показываем сводные числа
        try:
            x_m = ind.result.get("trace_x_m")
            u_ev = ind.result.get("profile_ev")
            if x_m is None or u_ev is None:
                raise KeyError("no profile")
            x_um = np.asarray(x_m) * 1e6
            u_mev = (np.asarray(u_ev) - ind.result["u_reference_ev"]) * 1e3
            ax.plot(x_um, u_mev, color="tab:red", lw=1.4)
            for a, b, c in ((-320, -160, "#DDEBFF"),
                            (-160, -45, "#FFF2CC"),
                            (-45, 45, "#FCE4D6"),
                            (45, 160, "#FFF2CC"),
                            (160, 320, "#DDEBFF")):
                ax.axvspan(a, b, color=c, alpha=0.35, zorder=0)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("U - U_ref (meV)")
            ax.grid(alpha=0.3)
        except Exception:
            ax.text(0.5, 0.5, "profile not retained\n(retain=True on evaluate)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# 5. ИНИЦИАЛИЗАЦИЯ С DIVERSITY
# ---------------------------------------------------------------------------
base._V223_DIVERSITY_THRESHOLD = 0.3


def diverse_initial_population(
    factory,
    baseline: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
    min_dist: float,
    max_attempts_per_slot: int = 200,
) -> list[np.ndarray]:
    """
    Жадно набирает геномы с попарным genome_distance >= min_dist.
    factory() -> кандидат; затем project_to_valid.
    """
    pool: list[np.ndarray] = []

    for _slot in range(target_size):
        best_candidate = None
        best_min_d = -1.0

        for _attempt in range(max_attempts_per_slot):
            try:
                cand = factory()
            except Exception:
                continue
            cand, _alpha = project_to_valid(baseline, cand)

            if pool:
                d_min = min(genome_distance(cand, prev) for prev in pool)
            else:
                d_min = np.inf

            if d_min >= min_dist:
                best_candidate = cand
                break

            if d_min > best_min_d:
                best_min_d = d_min
                best_candidate = cand

        if best_candidate is None:
            best_candidate = baseline.copy()
        pool.append(best_candidate)

    return pool


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = base.parser()

    # Новые флаги v223
    p.add_argument("--stage1-generations", type=int, default=5,
                   help="Сколько поколений быстро (multi-obj + live plot).")
    p.add_argument("--stage1-population-mult", type=float, default=1.5,
                   help="Множитель начальной популяции в stage 1.")
    p.add_argument("--diversity-threshold", type=float, default=0.30,
                   help="Минимальное нормированное genome distance при init.")
    p.add_argument("--live-plot", action="store_true",
                   help="Открыть окно с live-графиком (TkAgg / Qt5Agg).")
    p.add_argument("--plot-every", type=int, default=1,
                   help="Обновлять live-график каждые N поколений.")
    p.add_argument("--no-stage-split", action="store_true",
                   help="Отключить две стадии (чистое v22 поведение).")
    return p


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
        raise ValueError("population >= 8, offspring >= 1, generations >= 1")

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

    backend_payload = (
        backend.as_dict() if hasattr(backend, "as_dict")
        else {"selected": backend.selected, "available": backend.available}
    )
    (cfg.output_dir / "settings.json").write_text(
        json.dumps(
            {
                "settings": base.asdict(cfg) if hasattr(base, "asdict") else cfg.__dict__,
                "backend": backend_payload,
                "islands": [style.__dict__ for style in styles],
                "v223_flags": {
                    "stage1_generations": args.stage1_generations,
                    "diversity_threshold": args.diversity_threshold,
                    "live_plot": args.live_plot,
                    "plot_every": args.plot_every,
                },
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    print(f"v223 | backend={backend.selected} | islands={len(styles)}")
    print(f"Stage 1: {args.stage1_generations} gens, multi-objective, live_plot={args.live_plot}")
    print(f"Stage 2: {args.generations - args.stage1_generations} gens, v22 minimax mode")

    master_rng = np.random.default_rng(cfg.seed)
    baseline = discover_valid_baseline(master_rng)

    raw_seeds = elite_seeds()
    seeds: dict[str, np.ndarray] = {}
    for name, raw in raw_seeds.items():
        seeds[name], alpha = project_to_valid(baseline, raw)
        print(f"Seed {name}: alpha={alpha:.6f}")

    # --- общий кэш coarse-eval ------------------------------------------------
    cache: dict[tuple, dict] = {}

    def coarse(genome: np.ndarray) -> dict:
        values = repair(genome)
        key = tuple(np.round(values, 13))
        if key not in cache:
            ok, reason = base.geometry_precheck(values)
            if ok:
                cache[key], _ = evaluate(values, cfg, stage="coarse", retain=False)
            else:
                cache[key] = rejected(reason, "coarse", cfg)
        return dict(cache[key])

    # --- инициализация островов ----------------------------------------------
    islands: list[Island] = []
    master_rng_init = np.random.default_rng(cfg.seed + 1)

    for island_id, style in enumerate(styles):
        rng = np.random.default_rng(int(master_rng_init.integers(0, 2**32 - 1)))

        # Первый = точный seed / блок-микс
        label, genome = seed_from_kind(style.seed_kind, seeds, rng)
        genome, _ = project_to_valid(baseline, genome)

        pop: list[Individual] = [
            Individual(
                genome=genome.copy(),
                result=coarse(genome),
                island=island_id,
                generation=0,
                origin=label,
                style_name=style.name,
                seed_kind=style.seed_kind,
                emphasis=style.emphasis,
            )
        ]

        # Остальные — через diverse_initial_population
        def factory(style=style, genome=genome, rng=rng):
            if style.operator == "bspline":
                return bspline_genome_operator(genome, rng, style.explore_scale, 0.85)
            if style.operator == "memetic":
                props = memetic_candidates(genome, rng, 0.1, 1)
                return props[0]
            return local_mutation(genome, style.emphasis, rng,
                                  scale=style.explore_scale, probability=0.85)

        # В stage 1 усиливаем diversity
        div = float(args.diversity_threshold) if args.stage1_generations > 0 else base.MIN_GENOME_DISTANCE
        extra = diverse_initial_population(
            factory, baseline,
            target_size=max(0, cfg.population - 1),
            rng=rng,
            min_dist=div,
        )

        for g in extra:
            pop.append(Individual(
                genome=g,
                result=coarse(g),
                island=island_id,
                generation=0,
                origin=f"initial_diverse_{label}",
                style_name=style.name,
                seed_kind=style.seed_kind,
                emphasis=style.emphasis,
            ))

        base.rank_and_crowding(pop)
        islands.append(Island(style=style, rng=rng, population=pop))

        if island_id % 4 == 0:
            print(f"  island {island_id}/{len(styles)} seeded ({len(pop)} members)")

    # --- live plotter ---------------------------------------------------------
    plotter = LivePlotter(
        enabled=bool(args.live_plot),
        out_dir=cfg.output_dir,
    )

    # --- архив + история ------------------------------------------------------
    archive: list[Individual] = []

    def update_archive(items: list[Individual]) -> None:
        nonlocal archive
        unique = {tuple(np.round(it.genome, 12)): it.copy()
                  for it in archive + items if it.valid}
        archive = list(unique.values())
        base.rank_and_crowding(archive)
        archive.sort(key=lambda it: (
            not it.search_feasible,
            it.violation,
            it.rank,
            -it.crowding,
            # в stage1 хотим баланс по зонам; zonal_score balanced это даёт
            zonal_score(it.result, "balanced"),
        ))
        archive = archive[:2000]

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

    # --- главный цикл --------------------------------------------------------
    stage1_end = args.stage1_generations if not args.no_stage_split else 0
    best_global_score = np.inf
    stall = 0
    burst = 0
    search_mode = "NORMAL"

    def _summary(gen: int) -> dict:
        allp = [it for isl in islands for it in isl.population]
        valid = [it for it in allp if it.valid]
        feas = [it for it in valid if it.search_feasible]
        if not valid:
            return {"generation": gen, "valid": 0}

        def joint_key(it):
            return (
                not it.search_feasible,
                it.violation,
                zonal_score(it.result, "balanced"),
            )

        best_joint = min(valid, key=joint_key)
        return {
            "generation": gen,
            "mode": search_mode,
            "valid": len(valid),
            "search_feasible": len(feas),
            "joint_dz_um": best_joint.result["dz_peak_m"] * 1e6,
            "joint_excursion_mev": best_joint.result["excursion_ev"] * 1e3,
            "joint_dz_centre_um": best_joint.result["dz_centre_peak_m"] * 1e6,
            "joint_dz_transition_um": best_joint.result["dz_transition_peak_m"] * 1e6,
            "joint_dz_arm_um": best_joint.result["dz_arm_peak_m"] * 1e6,
            "joint_U_centre_mev": best_joint.result["excursion_centre_ev"] * 1e3,
            "joint_U_transition_mev": best_joint.result["excursion_transition_ev"] * 1e3,
            "joint_U_arm_mev": best_joint.result["excursion_arm_ev"] * 1e3,
            "joint_score": zonal_score(best_joint.result, "balanced"),
        }

    for generation in range(1, cfg.generations + 1):
        t_gen = time.time()

        in_stage1 = generation <= stage1_end
        # Переключение режима
        base.MULTI_OBJECTIVE_MODE = in_stage1
        search_mode = "STAGE1_MULTIOBJ" if in_stage1 else (
            "BURST" if burst > 0 else "NORMAL")

        if generation == stage1_end + 1 and stage1_end > 0:
            print(f"--- switching to stage 2 (minimax, all v22 logging) at gen {generation} ---")

        # --- потомки ---
        next_islands: list[Island] = []
        memetic_active = (cfg.memetic_interval > 0
                          and generation % cfg.memetic_interval == 0)

        for island_id, island in enumerate(islands):
            base.rank_and_crowding(island.population)

            elite = sorted(
                island.population,
                key=lambda it: (
                    not it.search_feasible, it.violation, it.rank, -it.crowding,
                    zonal_score(it.result, island.style.emphasis),
                ),
            )[: max(2, min(5, len(island.population)))]

            children: list[Individual] = []
            scale = anneal_scale(island.style, generation / cfg.generations)

            # --- memetic branch ---
            if memetic_active and island.style.operator == "memetic":
                center = elite[0].genome.copy()
                for step in range(cfg.memetic_steps):
                    proposals = memetic_candidates(center, island.rng, 0.5, cfg.memetic_candidates)
                    step_children = []
                    for p in proposals:
                        p, _ = project_to_valid(baseline, p)
                        step_children.append(Individual(
                            genome=p,
                            result=coarse(p),
                            island=island_id,
                            generation=generation,
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

            # --- обычные потомки ---
            while len(children) < cfg.offspring:
                v = island.rng.random()
                if v < 0.55:
                    parent = elite[0 if island.rng.random() < 0.8 else -1]
                    g = parent.genome.copy()
                    origin = "elite_local"
                elif v < 0.78:
                    l = tournament(island.population, island.style.emphasis, island.rng)
                    r = tournament(island.population, island.style.emphasis, island.rng)
                    g = blended_block_mix(l.genome, r.genome, island.rng)
                    origin = "blend"
                elif v < 0.92:
                    l = tournament(island.population, island.style.emphasis, island.rng)
                    r = tournament(island.population, island.style.emphasis, island.rng)
                    g = two_parent_block_mix(l.genome, r.genome, island.rng)
                    origin = "block"
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
                    genome=g,
                    result=coarse(g),
                    island=island_id,
                    generation=generation,
                    origin=origin,
                    style_name=island.style.name,
                    seed_kind=island.style.seed_kind,
                    emphasis=island.style.emphasis,
                ))

            next_islands.append(Island(
                style=island.style,
                rng=island.rng,
                population=base.survivors(island.population + children,
                                          cfg.population, island.style.emphasis),
            ))

        islands = next_islands

        # --- миграция ---
        if len(islands) > 1 and (burst > 0 or
                                  (cfg.migration_interval > 0
                                   and generation % cfg.migration_interval == 0)):
            n_mig = min(8 if burst > 0 else cfg.migrants, cfg.population)
            outgoing = []
            for isl in islands:
                base.rank_and_crowding(isl.population)
                ms = sorted(isl.population, key=lambda it: (
                    not it.search_feasible, it.violation, it.rank, -it.crowding,
                    zonal_score(it.result, isl.style.emphasis),
                ))[:n_mig]
                outgoing.append(ms)
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
                        cfg.population, target.style.emphasis),
                )

        # --- stall / burst ---
        valid_now = [it for isl in islands for it in isl.population if it.valid]
        cur_score = min((zonal_score(it.result, "balanced") for it in valid_now),
                        default=np.inf)
        if cur_score < best_global_score * (1 - cfg.improvement_fraction):
            best_global_score = cur_score
            stall = 0
            if burst > 0:
                burst = 0
        else:
            stall += 1
        if burst > 0:
            burst -= 1
        if burst == 0 and stall >= cfg.stall_generations:
            burst = cfg.burst_generations
            stall = 0
            print(f"burst scheduled at gen {generation}")

        # --- архив, история, логи ---
        update_archive([it for isl in islands for it in isl.population])

        hrow = _summary(generation)
        history.append(hrow)
        record_diversity(generation)

        # --- live plot ---
        allp = [it for isl in islands for it in isl.population if it.valid]
        best_ind = min(allp, key=lambda it: zonal_score(it.result, "balanced")) if allp else None
        if best_ind is not None and generation % max(1, args.plot_every) == 0:
            plotter.push(generation, best_ind, allp, hrow)

        # --- сохранение CSV (в стиле v22) ---
        save_csv(history, cfg.output_dir / "history.csv")
        save_csv(diversity_rows, cfg.output_dir / "diversity_history.csv")
        save_csv(
            elite_rows(islands, cfg),
            cfg.output_dir / "island_elites" / f"elites_gen_{generation:04d}.csv",
        )
        save_csv(
            [serialize(it) for isl in islands for it in isl.population],
            cfg.output_dir / f"population_gen_{generation:04d}.csv",
        )

        # --- лог ---
        dt = time.time() - t_gen
        print(
            f"G{generation:03d} [{search_mode}] "
            f"valid={hrow.get('valid', 0)} feas={hrow.get('search_feasible', 0)} "
            f"U_c={hrow.get('joint_U_centre_mev', np.nan):.2f} "
            f"U_t={hrow.get('joint_U_transition_mev', np.nan):.2f} "
            f"U_a={hrow.get('joint_U_arm_mev', np.nan):.2f} meV  "
            f"dz={hrow.get('joint_dz_um', np.nan):.2f} um  "
            f"[{dt:.1f}s]",
            flush=True,
        )

    # --- после цикла ---
    plotter.close()

    # Сохраняем пул stage-1 отдельно (для аудита)
    if stage1_end > 0:
        stage1_pool = [it for it in archive if it.generation <= stage1_end]
        stage1_pool.sort(key=lambda it: zonal_score(it.result, "balanced"))
        save_csv(
            [serialize(it) for it in stage1_pool[:200]],
            cfg.output_dir / "stage1_diverse_pool.csv",
        )
        print(f"Stage-1 pool saved: {len(stage1_pool)} items")

    # Оригинальные графики v22
    try:
        plot_convergence(history, cfg.output_dir, cfg)
        plot_pareto(archive, cfg.output_dir, cfg)
    except Exception as e:
        print(f"[v223] plot_convergence/plot_pareto error: {e}")

    # --- fine validation (как в v22) ---
    candidates = [it for it in archive if it.valid]
    candidates.sort(key=lambda it: (
        not it.search_feasible, it.violation,
        zonal_score(it.result, "balanced"),
        it.result["dz_peak_m"], it.result["excursion_ev"],
    ))
    selected = candidates[: cfg.fine_top]

    fine_rows: list[dict] = []
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
    print(f"Outputs: {cfg.output_dir}")


if __name__ == "__main__":
    main()
