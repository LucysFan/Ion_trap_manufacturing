"""Resolved RF-junction evaluator shared by optimization workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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


if TYPE_CHECKING:
    from core.analysis.barrier import BarrierMetrics
    from core.analysis.path_metrics import PathMetrics
    from core.analysis.rf_null_trace import RFNullTrace
    from core.geometry.mask_builder import XJunctionBEMModel


def plot_candidate_layout(
    model: "XJunctionBEMModel",
    trace: "RFNullTrace",
    *,
    output_path: Path,
    title: str,
) -> None:
    """Draw quadtree electrode panels and traced transverse RF minimum."""
    panels_m = model.bem.panels_m
    rf = model.bem.electrode_voltages_v > 0.5

    figure, axis = plt.subplots(figsize=(8.0, 7.5))

    patches = [
        Rectangle(
            (panel[0] * 1e6, panel[2] * 1e6),
            (panel[1] - panel[0]) * 1e6,
            (panel[3] - panel[2]) * 1e6,
        )
        for panel in panels_m
    ]

    collection = PatchCollection(
        patches,
        array=rf.astype(float),
        cmap="RdYlBu_r",
        edgecolor=(0.25, 0.25, 0.25, 0.18),
        linewidth=0.22,
    )
    collection.set_clim(0.0, 1.0)
    axis.add_collection(collection)

    good = trace.converged
    axis.plot(
        trace.x_m[good] * 1e6,
        trace.y_m[good] * 1e6,
        color="lime",
        linewidth=2.5,
        label="Traced transverse RF minimum",
    )

    if np.any(~good):
        axis.scatter(
            trace.x_m[~good] * 1e6,
            np.zeros(np.sum(~good)),
            marker="x",
            color="cyan",
            s=45,
            label="Trace failure",
        )

    axis.set_aspect("equal")
    axis.set_xlim(-350.0, 350.0)
    axis.set_ylim(-350.0, 350.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_diagnostics(
    trace: "RFNullTrace",
    metrics: "PathMetrics",
    barrier: "BarrierMetrics",
    *,
    output_path: Path,
    title: str,
) -> None:
    """Plot height, lateral shift, residuals, and pseudopotential profile."""
    good = trace.converged
    x_um = trace.x_m[good] * 1e6

    figure, axes = plt.subplots(4, 1, figsize=(10.0, 11.0), sharex=True)

    axes[0].plot(x_um, trace.z_m[good] * 1e6, color="tab:blue", linewidth=2.0)
    axes[0].axhline(
        metrics.arm_reference_height_m * 1e6,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Arm reference",
    )
    axes[0].set_ylabel(r"$z_{\rm min}$ [um]")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_um, trace.y_m[good] * 1e6, color="tab:green", linewidth=2.0)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"$y_{\rm min}$ [um]")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(
        x_um,
        trace.transverse_residual_v_m[good],
        color="tab:red",
        linewidth=1.8,
        label=r"$\sqrt{E_y^2+E_z^2}$",
    )
    axes[2].semilogy(
        x_um,
        trace.full_field_norm_v_m[good],
        color="tab:orange",
        linestyle="--",
        linewidth=1.3,
        label=r"$|E|$",
    )
    axes[2].set_ylabel("Field [V/m]")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3, which="both")

    axes[3].plot(
        x_um,
        barrier.energies_ev[good] * 1e3,
        color="tab:purple",
        linewidth=2.0,
    )
    axes[3].axhline(
        barrier.reference_energy_ev * 1e3,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Arm reference",
    )
    axes[3].set_xlabel("x [um]")
    axes[3].set_ylabel(r"$U_{\rm ps}$ [meV]")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_junction_genome(
    genome: JunctionGenome,
    *,
    label: str,
    x_values_m: np.ndarray,
    candidate_directory: Path | None = None,
    save_diagnostics: bool = False,
    show_bem_progress: bool = False,
) -> dict[str, object]:
    """
    Evaluate one JunctionGenome with the geometry-aware resolved BEM backend.

    The returned row remains finite on invalid geometry where possible, so it
    can later be used as a GA objective source.
    """
    base_row: dict[str, object] = {
        **genome.as_dict(),
        "label": label,
        "trace_valid": False,
        "path_valid": False,
        "height_peak_m": np.inf,
        "height_rms_m": np.inf,
        "barrier_ev": np.inf,
        "maximum_y_m": np.inf,
        "n_panels": 0,
        "reason": "",
    }

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
        base_row["reason"] = f"invalid geometry: {error}"
        return base_row

    if not manufacturability.valid:
        base_row["reason"] = "; ".join(manufacturability.messages)
        return base_row

    model = build_geometry_aware_quadtree_x_junction_bem(
        parameters,
        central_half_extent_m=180e-6,
        central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6,
        outer_max_cell_m=180e-6,
        min_cell_m=5e-6,
    )

    model.bem.assemble(show_progress=show_bem_progress)
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

    if save_diagnostics:
        if candidate_directory is None:
            raise ValueError(
                "candidate_directory is required when save_diagnostics=True."
            )

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
        **base_row,
        "trace_valid": bool(trace.valid),
        "path_valid": bool(validation.valid),
        "height_peak_m": float(metrics.height_peak_deviation_m),
        "height_rms_m": float(metrics.height_rms_deviation_m),
        "barrier_ev": float(barrier.barrier_height_ev),
        "maximum_y_m": float(metrics.maximum_lateral_offset_m),
        "n_panels": int(model.n_panels),
        "reason": "; ".join(validation.messages),
        "candidate_directory": (
            str(candidate_directory)
            if candidate_directory is not None
            else ""
        ),
    }