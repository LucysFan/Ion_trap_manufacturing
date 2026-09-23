"""Step 6c: local one-parameter RF-junction sensitivity scan.

The scan uses the resolved geometry-aware evaluator validated in 06b.
Each sweep changes exactly one JunctionGenome parameter around a nominal
candidate, leaving the other four parameters unchanged.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.optimization.genome import JunctionGenome
from core.optimization.junction_evaluator import evaluate_junction_genome


PARAMETERS = (
    "start_radius_um",
    "inner_shift_um",
    "outer_shift_um",
    "taper_length_um",
    "taper_power",
)

PARAMETER_LABELS = {
    "start_radius_um": r"$r_{\rm start}$ [um]",
    "inner_shift_um": r"$\Delta r_{\rm in}$ [um]",
    "outer_shift_um": r"$\Delta r_{\rm out}$ [um]",
    "taper_length_um": r"$L_{\rm taper}$ [um]",
    "taper_power": r"$p$",
}


def make_scan_plan(
    nominal: JunctionGenome,
) -> list[tuple[str, float, JunctionGenome]]:
    """Create one-at-a-time perturbations around the nominal geometry."""
    values_by_parameter = {
        "start_radius_um": (15.0, 45.0, 60.0),
        "inner_shift_um": (-25.0, 0.0, 15.0),
        "outer_shift_um": (0.0, 35.0, 50.0),
        "taper_length_um": (80.0, 220.0),
        "taper_power": (0.9, 1.4, 2.8, 3.5),
    }

    plan: list[tuple[str, float, JunctionGenome]] = []

    for parameter in PARAMETERS:
        for value in values_by_parameter[parameter]:
            genome = replace(nominal, **{parameter: value})
            plan.append((parameter, value, genome))

    return plan


def valid_metric(
    row: dict[str, object],
    key: str,
    scale: float,
) -> float:
    """Return a plottable finite metric; invalid candidates become NaN."""
    value = float(row.get(key, np.nan))

    if not bool(row.get("trace_valid", False)):
        return np.nan
    if not np.isfinite(value):
        return np.nan

    return value * scale


def plot_sensitivity(
    rows: list[dict[str, object]],
    nominal: JunctionGenome,
    *,
    output_path: Path,
) -> None:
    """Create a five-column response map for height and RF barrier."""
    figure, axes = plt.subplots(
        2,
        len(PARAMETERS),
        figsize=(4.1 * len(PARAMETERS), 7.0),
        squeeze=False,
        sharey="row",
    )

    for column, parameter in enumerate(PARAMETERS):
        parameter_rows = [
            row
            for row in rows
            if row.get("swept_parameter") == parameter
        ]

        parameter_rows.sort(
            key=lambda row: float(row["swept_value"])
        )

        x = np.asarray(
            [float(row["swept_value"]) for row in parameter_rows],
            dtype=float,
        )
        dz_um = np.asarray(
            [valid_metric(row, "height_peak_m", 1e6) for row in parameter_rows],
            dtype=float,
        )
        barrier_mev = np.asarray(
            [valid_metric(row, "barrier_ev", 1e3) for row in parameter_rows],
            dtype=float,
        )

        nominal_x = float(getattr(nominal, parameter))

        axes[0, column].plot(
            x,
            dz_um,
            "o-",
            color="tab:blue",
            linewidth=1.8,
            markersize=5.5,
        )
        axes[0, column].axvline(
            nominal_x,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axes[0, column].axhline(
            3.0,
            color="tab:red",
            linestyle=":",
            linewidth=1.1,
        )
        axes[0, column].set_title(PARAMETER_LABELS[parameter])
        axes[0, column].grid(alpha=0.3)

        axes[1, column].plot(
            x,
            barrier_mev,
            "o-",
            color="tab:purple",
            linewidth=1.8,
            markersize=5.5,
        )
        axes[1, column].axvline(
            nominal_x,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1, column].grid(alpha=0.3)
        axes[1, column].set_xlabel(PARAMETER_LABELS[parameter])

    axes[0, 0].set_ylabel(r"Peak $\Delta z$ [um]")
    axes[1, 0].set_ylabel(r"RF barrier [meV]")

    figure.suptitle(
        "Step 06c: one-parameter sensitivity around "
        r"$(30, -10, +20, 150, 2.0)$",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
) -> None:
    keys = (
        "label",
        "swept_parameter",
        "swept_value",
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
        writer = csv.DictWriter(
            file,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def score_compromise(row: dict[str, object]) -> float:
    """Ranking only for selecting a few visualization candidates."""
    if not bool(row.get("trace_valid", False)):
        return np.inf

    height_um = float(row["height_peak_m"]) * 1e6
    barrier_mev = float(row["barrier_ev"]) * 1e3

    if not np.isfinite(height_um) or not np.isfinite(barrier_mev):
        return np.inf

    return (height_um / 30.0) ** 2 + (barrier_mev / 130.0) ** 2


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "06c_local_sensitivity_scan"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Keep the 05h/06b resolved path convention.
    x_values_m = np.linspace(-350e-6, 350e-6, 41)

    nominal = JunctionGenome(
        start_radius_um=30.0,
        inner_shift_um=-10.0,
        outer_shift_um=20.0,
        taper_length_um=150.0,
        taper_power=2.0,
    )

    scan_plan = make_scan_plan(nominal)

    print("Step 06c: local resolved sensitivity scan")
    print("=" * 112)
    print(
        f"Nominal: {nominal.as_dict()}"
    )
    print(
        f"{'index':>5} {'parameter':<18} {'value':>10} "
        f"{'trace':>7} {'peak dz [um]':>15} "
        f"{'barrier [meV]':>16} {'panels':>8}"
    )

    rows: list[dict[str, object]] = []

    nominal_row = evaluate_junction_genome(
        nominal,
        label="nominal",
        x_values_m=x_values_m,
        candidate_directory=None,
        save_diagnostics=False,
        show_bem_progress=True,
    )
    nominal_row["swept_parameter"] = "nominal"
    nominal_row["swept_value"] = 0.0
    rows.append(nominal_row)

    print(
        f"{0:5d} {'nominal':<18} {'-':>10} "
        f"{str(bool(nominal_row['trace_valid'])):>7} "
        f"{float(nominal_row['height_peak_m']) * 1e6:15.4f} "
        f"{float(nominal_row['barrier_ev']) * 1e3:16.5f} "
        f"{int(nominal_row['n_panels']):8d}"
    )

    for index, (parameter, value, genome) in enumerate(scan_plan, start=1):
        label = f"{parameter}_{value:+.3f}"

        row = evaluate_junction_genome(
            genome,
            label=label,
            x_values_m=x_values_m,
            candidate_directory=None,
            save_diagnostics=False,
            show_bem_progress=True,
        )
        row["swept_parameter"] = parameter
        row["swept_value"] = value
        rows.append(row)

        print(
            f"{index:5d} {parameter:<18} {value:10.3f} "
            f"{str(bool(row['trace_valid'])):>7} "
            f"{float(row['height_peak_m']) * 1e6:15.4f} "
            f"{float(row['barrier_ev']) * 1e3:16.5f} "
            f"{int(row['n_panels']):8d}"
        )

    write_csv(
        rows,
        output_path=output_directory / "local_sensitivity_results.csv",
    )

    plot_sensitivity(
        rows,
        nominal,
        output_path=output_directory / "local_sensitivity.png",
    )

    valid_rows = [
        row
        for row in rows
        if bool(row.get("trace_valid", False))
        and np.isfinite(float(row.get("height_peak_m", np.inf)))
        and np.isfinite(float(row.get("barrier_ev", np.inf)))
    ]

    best_height = min(
        valid_rows,
        key=lambda row: float(row["height_peak_m"]),
    )
    best_barrier = min(
        valid_rows,
        key=lambda row: float(row["barrier_ev"]),
    )
    best_compromise = min(valid_rows, key=score_compromise)

    selected = {
        "nominal": nominal_row,
        "best_height": best_height,
        "best_barrier": best_barrier,
        "best_compromise": best_compromise,
    }

    print()
    print("Selected diagnostic reruns")
    print("=" * 112)

    for category, row in selected.items():
        genome = JunctionGenome(
            start_radius_um=float(row["start_radius_um"]),
            inner_shift_um=float(row["inner_shift_um"]),
            outer_shift_um=float(row["outer_shift_um"]),
            taper_length_um=float(row["taper_length_um"]),
            taper_power=float(row["taper_power"]),
        )

        print(
            f"{category:<16} {row['label']}: "
            f"dz={float(row['height_peak_m']) * 1e6:.4f} um, "
            f"barrier={float(row['barrier_ev']) * 1e3:.5f} meV"
        )

        evaluate_junction_genome(
            genome,
            label=f"{category}_{row['label']}",
            x_values_m=x_values_m,
            candidate_directory=output_directory / "selected" / category,
            save_diagnostics=True,
            show_bem_progress=True,
        )

    print()
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()