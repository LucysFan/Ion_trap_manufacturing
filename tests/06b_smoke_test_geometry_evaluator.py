"""Step 6b: smoke test for genome-to-geometry-aware-junction evaluation.

This workflow deliberately reuses the physical calculation sequence from 05h.
It does not run a GA. Its only purpose is to verify that a JunctionGenome can
be converted into a valid, evaluated RF-junction candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.targets import (
    RF_ANGULAR_FREQUENCY_RAD_S,
    TARGET_ION_HEIGHT_M,
)
from core.analysis.barrier import (
    compute_barrier_metrics,
    pseudopotential_profile_ev,
)
from core.analysis.path_metrics import compute_path_metrics
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.analysis.validation import validate_rf_transport_path
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import (
    check_x_junction_manufacturability,
)
from core.geometry.mask_builder import (
    build_geometry_aware_quadtree_x_junction_bem,
)
from core.optimization.genome import JunctionGenome

from core.optimization.junction_evaluator import evaluate_junction_genome

def evaluate_genome(
    genome: JunctionGenome,
    *,
    label: str,
    x_values_m: np.ndarray,
    candidate_directory: Path,
) -> dict[str, object]:
    """Evaluate one physical JunctionGenome using the resolved 05h pipeline."""
    parameters = make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M,
        arm_length_m=600e-6,
        outer_extent_m=900e-6,
        taper_length_m=genome.taper_length_um * 1e-6,
        inner_edge_shift_at_centre_m=genome.inner_shift_um * 1e-6,
        outer_edge_shift_at_centre_m=genome.outer_shift_um * 1e-6,
        taper_power=genome.taper_power,
        rf_start_radius_override_m=genome.start_radius_um * 1e-6,
    )

    try:
        manufacturability = check_x_junction_manufacturability(parameters)
    except ValueError as error:
        return {
            **genome.as_dict(),
            "label": label,
            "trace_valid": False,
            "path_valid": False,
            "height_peak_m": np.inf,
            "height_rms_m": np.inf,
            "barrier_ev": np.inf,
            "maximum_y_m": np.inf,
            "n_panels": 0,
            "reason": f"invalid geometry: {error}",
        }

    if not manufacturability.valid:
        return {
            **genome.as_dict(),
            "label": label,
            "trace_valid": False,
            "path_valid": False,
            "height_peak_m": np.inf,
            "height_rms_m": np.inf,
            "barrier_ev": np.inf,
            "maximum_y_m": np.inf,
            "n_panels": 0,
            "reason": "; ".join(manufacturability.messages),
        }

    model = build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )

    model.bem.assemble(show_progress=True)
    model.bem.solve(show_progress=False)

    trace = trace_rf_transverse_minimum(
        model.bem,
        x_values_m,
        initial_y_m=0.0,
        initial_z_m=TARGET_ION_HEIGHT_M,
        residual_tolerance_v_m=1e-3,
        max_transverse_shift_m=25e-6,
    )

    metrics = compute_path_metrics(trace)

    energies_ev = pseudopotential_profile_ev(
        model.bem,
        trace,
        rf_voltage_peak_v=100.0,
        rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S,
    )
    barrier = compute_barrier_metrics(energies_ev, trace)
    validation = validate_rf_transport_path(metrics)

    candidate_directory.mkdir(parents=True, exist_ok=True)

    title = (
        f"{label}: start={genome.start_radius_um:.1f} um, "
        f"din={genome.inner_shift_um:.1f} um, "
        f"dout={genome.outer_shift_um:.1f} um, "
        f"L={genome.taper_length_um:.1f} um, "
        f"p={genome.taper_power:.2f}"
    )

    plot_candidate_layout(
        model,
        trace,
        output_path=candidate_directory / "layout_and_trace.png",
        title=title,
    )
    plot_candidate_diagnostics(
        trace,
        metrics,
        barrier,
        output_path=candidate_directory / "path_diagnostics.png",
        title=title,
    )

    return {
        **genome.as_dict(),
        "label": label,
        "trace_valid": bool(trace.valid),
        "path_valid": bool(validation.valid),
        "height_peak_m": float(metrics.height_peak_deviation_m),
        "height_rms_m": float(metrics.height_rms_deviation_m),
        "barrier_ev": float(barrier.barrier_height_ev),
        "maximum_y_m": float(metrics.maximum_lateral_offset_m),
        "n_panels": int(model.n_panels),
        "reason": "; ".join(validation.messages),
        "candidate_directory": str(candidate_directory),
    }


def print_result(row: dict[str, object]) -> None:
    print()
    print(f"=== {row['label']} ===")
    print(
        "geometry: "
        f"start={float(row['start_radius_um']):.1f} um, "
        f"inner={float(row['inner_shift_um']):+.1f} um, "
        f"outer={float(row['outer_shift_um']):+.1f} um, "
        f"L={float(row['taper_length_um']):.1f} um, "
        f"p={float(row['taper_power']):.2f}"
    )
    print(f"trace valid: {bool(row['trace_valid'])}")
    print(f"path valid:  {bool(row['path_valid'])}")
    print(f"peak dz:     {float(row['height_peak_m']) * 1e6:.4f} um")
    print(f"rms dz:      {float(row['height_rms_m']) * 1e6:.4f} um")
    print(f"barrier:     {float(row['barrier_ev']) * 1e3:.5f} meV")
    print(f"max |y|:     {float(row['maximum_y_m']) * 1e6:.4f} um")
    print(f"panels:      {int(row['n_panels'])}")
    print(f"reason:      {row['reason']}")


def save_results_csv(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    keys = (
        "label",
        "start_radius_um",
        "inner_shift_um",
        "outer_shift_um",
        "taper_length_um",
        "taper_power",
        "trace_valid",
        "path_valid",
        "height_peak_m",
        "height_rms_m",
        "barrier_ev",
        "maximum_y_m",
        "n_panels",
        "reason",
        "candidate_directory",
    )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(",".join(keys) + "\n")
        for row in rows:
            values = [
                str(row.get(key, "")).replace(",", ";")
                for key in keys
            ]
            file.write(",".join(values) + "\n")


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "06b_smoke_test_geometry_evaluator"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Same resolved-backend order as 05h. For the smoke test, 41 points are
    # enough; later selected finalists can use 71 or more.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    candidates = {
        "baseline_05h": JunctionGenome(
            start_radius_um=37.35,
            inner_shift_um=0.0,
            outer_shift_um=0.0,
            taper_length_um=150.0,
            taper_power=2.0,
        ),
        "wide_10_20_05h": JunctionGenome(
            start_radius_um=30.0,
            inner_shift_um=-10.0,
            outer_shift_um=20.0,
            taper_length_um=150.0,
            taper_power=2.0,
        ),
        "long_wide_05h": JunctionGenome(
            start_radius_um=30.0,
            inner_shift_um=-10.0,
            outer_shift_um=20.0,
            taper_length_um=250.0,
            taper_power=2.0,
        ),
    }

    rows: list[dict[str, object]] = []

    print("Step 06b: JunctionGenome -> resolved 05h physical evaluator")
    print("=" * 78)

    for index, (label, genome) in enumerate(candidates.items(), start=1):
        print(f"[{index}/{len(candidates)}] Evaluating {label} ...")

        row = evaluate_junction_genome(
        genome,
        label=label,
        x_values_m=x_values_m,
        candidate_directory=output_directory / "candidates" / label,
        save_diagnostics=True,
        show_bem_progress=True,
    )
        rows.append(row)
        print_result(row)

    save_results_csv(
        rows,
        output_path=output_directory / "geometry_evaluator_smoke_test.csv",
    )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()