"""Step 08: NSGA-II optimization over the complete static RF genome.

This workflow optimizes only the static RF electrode surface of the planar
junction. It does not include ion shuttling dynamics or time-dependent DC
waveforms. The output is a Pareto set of static RF geometries suitable for the
later single-ion shuttling stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.electrostatics.fixed_mesh import build_fixed_mesh
from core.optimization.genome import Genome, random_genome
from core.optimization.junction_evaluator import (
    EvaluatorConfig,
    JunctionEvaluation,
    JunctionEvaluator,
)
from core.optimization.nsga2 import (
    Nsga2Config,
    initialize_population,
    make_offspring,
    objective_matrix,
    rank_and_crowding,
    select_survivors,
)
from core.optimization.rf_experiment import (
    ExperimentConfig,
    generation_statistics,
    write_history_csv,
    write_results,
)

try:
    from core.ga.fixed_bem import FixedMeshBEM, available_backend
except ImportError as error:
    raise ImportError(
        "workflows/08_full_genom_ga.py expects core.ga.fixed_bem "
        "to provide FixedMeshBEM and available_backend()."
    ) from error


class MeshConfig:
    outer_extent_m: float = 650e-6
    central_half_extent_m: float = 140e-6
    central_max_cell_m: float = 8e-6
    boundary_max_cell_m: float = 8e-6
    outer_max_cell_m: float = 12e-6
    min_cell_m: float = 8e-6
    max_panels: int = 30000

    base_inner_m: float = 37.35e-6
    base_outer_m: float = 216.45e-6


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NSGA-II optimization of the complete static RF genome."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="BEM backend. 'auto' prefers CUDA when available.",
    )
    parser.add_argument("--population", type=int, default=256, help="Population size.")
    parser.add_argument("--offspring", type=int, default=256, help="Offspring per generation.")
    parser.add_argument("--generations", type=int, default=20, help="NSGA-II generations.")
    parser.add_argument("--seed", type=int, default=20260924, help="Random seed.")
    parser.add_argument("--route-extent-um", type=float, default=350.0, help="Route extent in um.")
    parser.add_argument("--path-points", type=int, default=41, help="Points along route.")
    parser.add_argument("--arm-length-um", type=float, default=350.0, help="Mask arm half-length in um.")
    parser.add_argument("--rf-peak-v", type=float, default=100.0, help="RF peak voltage.")
    parser.add_argument("--target-frequency-mhz", type=float, default=1.5, help="Target transverse frequency in MHz.")
    parser.add_argument("--gpu-candidate-chunk", type=int, default=64, help="Chunk size for field_batch.")
    parser.add_argument("--mesh-cell-um", type=float, default=8.0, help="Legacy square cell size for center-based fixed mesh.")
    parser.add_argument("--broad-init", action="store_true", help="Use broad random initialization.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint cadence.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures/08_full_genom_ga",
        help="Output directory.",
    )
    return parser


def resolve_backend(requested: str) -> tuple[str, dict[str, object]]:
    status = available_backend(requested)
    if not status.available:
        raise RuntimeError(status.reason)
    return status.selected, status.as_dict()


def build_fixed_bem(backend: str, *, legacy_cell_size_m: float) -> Any:
    centers = build_fixed_mesh(MeshConfig())
    bem = FixedMeshBEM(
        centers,
        backend=backend,
        dtype="float64",
        legacy_cell_size_m=legacy_cell_size_m,
    )
    return bem


def extract_mesh_centers(bem: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(bem, "x_centers_m") and hasattr(bem, "y_centers_m"):
        return (
            np.asarray(bem.x_centers_m, dtype=float),
            np.asarray(bem.y_centers_m, dtype=float),
        )

    if hasattr(bem, "panel_centers_m"):
        centers = np.asarray(bem.panel_centers_m, dtype=float)
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError("panel_centers_m has unexpected shape")
        return centers[:, 0], centers[:, 1]

    if hasattr(bem, "centers_m"):
        centers = np.asarray(bem.centers_m, dtype=float)
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError("centers_m has unexpected shape")
        return centers[:, 0], centers[:, 1]

    if hasattr(bem, "panels_cpu"):
        panels = np.asarray(bem.panels_cpu, dtype=float)
        return 0.5 * (panels[:, 0] + panels[:, 1]), 0.5 * (panels[:, 2] + panels[:, 3])

    raise AttributeError("Could not extract panel centers from FixedMeshBEM.")


def choose_best_valid_index(evaluations: list[JunctionEvaluation]) -> int:
    valid_indices = [i for i, ev in enumerate(evaluations) if ev.valid]
    if not valid_indices:
        objectives = objective_matrix(evaluations)
        return int(np.argmin(np.sum(objectives, axis=1)))

    valid_objectives = objective_matrix([evaluations[i] for i in valid_indices])
    return int(valid_indices[int(np.argmin(np.sum(valid_objectives, axis=1)))])


def choose_best_height_index(evaluations: list[JunctionEvaluation]) -> int:
    heights = np.asarray(
        [ev.metrics.get("height_peak_m", np.inf) for ev in evaluations],
        dtype=float,
    )
    return int(np.argmin(heights))


def save_generation_summary(
    output_dir: Path,
    generation: int,
    evaluations: list[JunctionEvaluation],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    objectives = objective_matrix(evaluations)
    _, ranks, crowding = rank_and_crowding(objectives)

    path = output_dir / f"generation_{generation:04d}_summary.csv"
    fieldnames = [
        "index",
        "rank",
        "crowding",
        "valid",
        "failure_reason",
        "height_peak_m",
        "height_rms_m",
        "lateral_peak_m",
        "barrier_ev",
        "frequency_error_hz",
        "anisotropy_max",
        "field_residual_max_v_m",
        "frequency_min_hz",
        "frequency_max_hz",
        "geometry_penalty",
        "objectives",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, evaluation in enumerate(evaluations):
            writer.writerow(
                {
                    "index": index,
                    "rank": int(ranks[index]),
                    "crowding": float(crowding[index]),
                    "valid": bool(evaluation.valid),
                    "failure_reason": evaluation.failure_reason,
                    "height_peak_m": evaluation.metrics.get("height_peak_m"),
                    "height_rms_m": evaluation.metrics.get("height_rms_m"),
                    "lateral_peak_m": evaluation.metrics.get("lateral_peak_m"),
                    "barrier_ev": evaluation.metrics.get("barrier_ev"),
                    "frequency_error_hz": evaluation.metrics.get("frequency_error_hz"),
                    "anisotropy_max": evaluation.metrics.get("anisotropy_max"),
                    "field_residual_max_v_m": evaluation.metrics.get("field_residual_max_v_m"),
                    "frequency_min_hz": evaluation.metrics.get("frequency_min_hz"),
                    "frequency_max_hz": evaluation.metrics.get("frequency_max_hz"),
                    "geometry_penalty": evaluation.metrics.get("geometry_penalty"),
                    "objectives": json.dumps(evaluation.objectives.tolist()),
                }
            )


def print_generation_log(
    generation: int,
    total_generations: int,
    elapsed_s: float,
    evaluations: list[JunctionEvaluation],
) -> None:
    stats = generation_statistics(generation, evaluations, elapsed_s=elapsed_s)
    best_height_index = choose_best_height_index(evaluations)
    best_height = evaluations[best_height_index].metrics.get("height_peak_m", np.nan)
    print(
        f"[gen {generation:03d}/{total_generations:03d}] "
        f"valid={stats.valid_count:4d}/{stats.population_size:4d} "
        f"pareto={stats.pareto_size:4d} "
        f"best_height={best_height * 1e6:8.3f} um "
        f"best_barrier={stats.best_barrier_ev * 1e3:8.4f} meV "
        f"best_df={stats.best_frequency_error_hz / 1e3:8.2f} kHz "
        f"dt={elapsed_s:8.2f} s"
    )


def run_experiment(
    *,
    backend: str,
    backend_info: dict[str, object],
    experiment_config: ExperimentConfig,
    evaluator_config: EvaluatorConfig,
    nsga2_config: Nsga2Config,
    arm_length_m: float,
    broad_init: bool,
    output_dir: Path,
    legacy_cell_size_m: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(experiment_config.seed)

    print("Backend info:")
    print(json.dumps(backend_info, indent=2))

    print("Building fixed BEM ...")
    bem = build_fixed_bem(backend, legacy_cell_size_m=legacy_cell_size_m)

    print(f"Panels: {bem.n}")
    if hasattr(bem, "assemble"):
        print("Assembling BEM matrix ...")
        t_assemble = time.perf_counter()
        bem.assemble(block_rows=128)
        bem.factorize()
        bem.release_matrix()
        bem.synchronize()
        print(f"BEM ready in {time.perf_counter() - t_assemble:.2f} s")

    x_centers_m, y_centers_m = extract_mesh_centers(bem)

    evaluator = JunctionEvaluator(
        bem=bem,
        x_centers_m=x_centers_m,
        y_centers_m=y_centers_m,
        l_arm_m=arm_length_m,
        config=evaluator_config,
    )

    initializer = lambda local_rng: random_genome(local_rng, broad=broad_init)

    print("Initializing population ...")
    population = initialize_population(
        rng,
        size=experiment_config.population_size,
        initializer=initializer,
        config=nsga2_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    generations_dir = output_dir / "generations"

    print("Evaluating initial population ...")
    t0 = time.perf_counter()
    evaluations = evaluator.evaluate_population(population)
    initial_elapsed = time.perf_counter() - t0

    history = [
        generation_statistics(
            generation=0,
            evaluations=evaluations,
            elapsed_s=initial_elapsed,
        )
    ]
    write_history_csv(output_dir / "history.csv", history)
    save_generation_summary(generations_dir, 0, evaluations)
    write_results(
        checkpoints_dir / "generation_0000",
        population,
        evaluations,
        0,
        experiment_config=experiment_config,
        evaluator_config=evaluator_config,
    )
    print_generation_log(0, experiment_config.generations, initial_elapsed, evaluations)

    for generation in range(1, experiment_config.generations + 1):
        t_gen = time.perf_counter()

        offspring = make_offspring(
            population,
            evaluations,
            rng,
            offspring_size=experiment_config.offspring_size,
            config=nsga2_config,
        )
        offspring_evaluations = evaluator.evaluate_population(offspring)

        population, evaluations, _ = select_survivors(
            population + offspring,
            evaluations + offspring_evaluations,
            experiment_config.population_size,
        )

        elapsed_s = time.perf_counter() - t_gen
        history.append(
            generation_statistics(
                generation=generation,
                evaluations=evaluations,
                elapsed_s=elapsed_s,
            )
        )

        write_history_csv(output_dir / "history.csv", history)
        save_generation_summary(generations_dir, generation, evaluations)
        print_generation_log(generation, experiment_config.generations, elapsed_s, evaluations)

        if generation % experiment_config.checkpoint_every == 0:
            write_results(
                checkpoints_dir / f"generation_{generation:04d}",
                population,
                evaluations,
                generation,
                experiment_config=experiment_config,
                evaluator_config=evaluator_config,
            )

    best_valid_index = choose_best_valid_index(evaluations)
    best_height_index = choose_best_height_index(evaluations)

    write_results(
        output_dir / "final",
        population,
        evaluations,
        experiment_config.generations,
        experiment_config=experiment_config,
        evaluator_config=evaluator_config,
    )

    best_valid_genome = population[best_valid_index]
    best_valid_eval = evaluations[best_valid_index]

    best_height_genome = population[best_height_index]
    best_height_eval = evaluations[best_height_index]

    (output_dir / "final" / "best_valid_genome.json").write_text(
        json.dumps(
            {
                "index": int(best_valid_index),
                "summary": best_valid_genome.summary(),
                "metrics": best_valid_eval.metrics,
                "objectives": best_valid_eval.objectives.tolist(),
                "valid": bool(best_valid_eval.valid),
                "failure_reason": best_valid_eval.failure_reason,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    (output_dir / "final" / "best_height_genome.json").write_text(
        json.dumps(
            {
                "index": int(best_height_index),
                "summary": best_height_genome.summary(),
                "metrics": best_height_eval.metrics,
                "objectives": best_height_eval.objectives.tolist(),
                "valid": bool(best_height_eval.valid),
                "failure_reason": best_height_eval.failure_reason,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    return {
        "population": population,
        "evaluations": evaluations,
        "history": history,
        "best_valid_index": best_valid_index,
        "best_valid_genome": best_valid_genome,
        "best_valid_evaluation": best_valid_eval,
        "best_height_index": best_height_index,
        "best_height_genome": best_height_genome,
        "best_height_evaluation": best_height_eval,
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    backend, backend_info = resolve_backend(args.backend)

    experiment_config = ExperimentConfig(
        seed=int(args.seed),
        population_size=int(args.population),
        offspring_size=int(args.offspring),
        generations=int(args.generations),
        checkpoint_every=int(args.checkpoint_every),
        plot_every=1,
    )

    evaluator_config = EvaluatorConfig(
        route_extent_m=float(args.route_extent_um) * 1e-6,
        path_points=int(args.path_points),
        target_frequency_hz=float(args.target_frequency_mhz) * 1e6,
        v_rf_peak_v=float(args.rf_peak_v),
        gpu_candidate_chunk=int(args.gpu_candidate_chunk),
    )

    nsga2_config = Nsga2Config(
        min_rf_width_m=evaluator_config.min_rf_width_m,
    )

    output_dir = Path(args.output_dir)

    result = run_experiment(
        backend=backend,
        backend_info=backend_info,
        experiment_config=experiment_config,
        evaluator_config=evaluator_config,
        nsga2_config=nsga2_config,
        arm_length_m=float(args.arm_length_um) * 1e-6,
        broad_init=bool(args.broad_init),
        output_dir=output_dir,
        legacy_cell_size_m=float(args.mesh_cell_um) * 1e-6,
    )

    print()
    print("Best valid genome summary:")
    print(json.dumps(result["best_valid_genome"].summary(), indent=2))
    print("Best valid metrics:")
    print(json.dumps(result["best_valid_evaluation"].metrics, indent=2, allow_nan=True))
    print()
    print("Best height genome summary:")
    print(json.dumps(result["best_height_genome"].summary(), indent=2))
    print("Best height metrics:")
    print(json.dumps(result["best_height_evaluation"].metrics, indent=2, allow_nan=True))
    print()
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()