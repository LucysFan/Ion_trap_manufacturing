"""Plots for RF transverse-minimum traces through an X-junction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.analysis.barrier import BarrierMetrics
from core.analysis.path_metrics import PathMetrics
from core.analysis.rf_null_trace import RFNullTrace
from core.geometry.mask_builder import XJunctionBEMModel


def _prepare_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_trace_on_geometry(
    model: XJunctionBEMModel,
    trace: RFNullTrace,
    *,
    output_path: str | Path,
) -> None:
    """Show projected transverse RF-minimum trace over electrode mask."""
    path = _prepare_path(output_path)

    fig, axis = plt.subplots(figsize=(7.5, 7.0))

    image = axis.pcolormesh(
        model.x_edges_m * 1e6,
        model.y_edges_m * 1e6,
        model.rf_mask.astype(float),
        cmap="RdYlBu_r",
        shading="flat",
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = fig.colorbar(image, ax=axis, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["Ground", "RF"])

    good = trace.converged
    bad = ~good

    axis.plot(
        trace.x_m[good] * 1e6,
        trace.y_m[good] * 1e6,
        color="lime",
        linewidth=2.5,
        label="Traced transverse RF minimum",
    )

    if np.any(bad):
        axis.scatter(
            trace.x_m[bad] * 1e6,
            np.zeros(np.sum(bad)),
            color="cyan",
            marker="x",
            s=45,
            label="Trace failure",
        )

    axis.set_aspect("equal")
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title("RF transverse-minimum trace over X-junction geometry")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_path_metrics(
    trace: RFNullTrace,
    metrics: PathMetrics,
    barrier: BarrierMetrics,
    *,
    output_path: str | Path,
) -> None:
    """Plot height, lateral displacement, residual field, and RF proxy."""
    path = _prepare_path(output_path)

    x_um = trace.x_m * 1e6
    good = trace.converged

    fig, axes = plt.subplots(4, 1, figsize=(10.0, 11.0), sharex=True)

    axes[0].plot(x_um[good], trace.z_m[good] * 1e6, color="tab:blue", lw=2)
    axes[0].axhline(
        metrics.arm_reference_height_m * 1e6,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Arm reference height",
    )
    axes[0].set_ylabel(r"$z_{\rm min}$ [um]")
    axes[0].set_title(
        "Transport-path diagnostics: height, lateral offset, residual, RF proxy"
    )
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_um[good], trace.y_m[good] * 1e6, color="tab:green", lw=2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel(r"$y_{\rm min}$ [um]")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(
        x_um[good],
        trace.transverse_residual_v_m[good],
        color="tab:red",
        lw=1.8,
        label=r"$\sqrt{E_y^2+E_z^2}$",
    )
    axes[2].semilogy(
        x_um[good],
        trace.full_field_norm_v_m[good],
        color="tab:orange",
        lw=1.2,
        linestyle="--",
        label=r"$|E|$",
    )
    axes[2].set_ylabel("Field [V/m]")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3, which="both")

    axes[3].plot(
        x_um[good],
        barrier.energies_ev[good],
        color="tab:purple",
        lw=2,
    )
    axes[3].axhline(
        barrier.reference_energy_ev,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Arm reference",
    )
    axes[3].set_xlabel("x [um]")
    axes[3].set_ylabel(r"$U_{\rm ps}$ [eV]")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_height_deviation(
    trace: RFNullTrace,
    metrics: PathMetrics,
    *,
    output_path: str | Path,
) -> None:
    """Plot z(x) relative to arm reference height."""
    path = _prepare_path(output_path)

    good = trace.converged
    x_um = trace.x_m[good] * 1e6
    deviation_um = (
        trace.z_m[good] - metrics.arm_reference_height_m
    ) * 1e6

    fig, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.plot(x_um, deviation_um, color="tab:blue", lw=2)
    axis.axhline(0.0, color="black", linewidth=1)

    axis.fill_between(
        x_um,
        -metrics.height_peak_deviation_m * 1e6,
        metrics.height_peak_deviation_m * 1e6,
        color="tab:blue",
        alpha=0.12,
        label=(
            "Peak deviation = "
            f"{metrics.height_peak_deviation_m * 1e6:.3f} um"
        ),
    )

    axis.set_xlabel("x [um]")
    axis.set_ylabel(r"$z(x)-z_{\rm arm}$ [um]")
    axis.set_title("RF-null height deviation through X-junction")
    axis.legend(loc="best")
    axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)