"""Plots for controlled X-junction geometry scans."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path) -> Path:
    """Create output parent directory and return normalized Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_island_scan_summary(
    results: list[dict[str, object]],
    *,
    output_path: str | Path,
) -> None:
    """Plot volcano and RF-proxy barrier versus central-island size."""
    path = _prepare_path(output_path)

    kinds = sorted({str(row["kind"]) for row in results})

    colors = {
        "none": "black",
        "square": "tab:blue",
        "circle": "tab:orange",
    }
    markers = {
        "none": "o",
        "square": "s",
        "circle": "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for kind in kinds:
        subset = [row for row in results if row["kind"] == kind]

        radii_um = np.asarray(
            [float(row["radius_m"]) * 1e6 for row in subset],
            dtype=float,
        )
        height_um = np.asarray(
            [float(row["height_peak_m"]) * 1e6 for row in subset],
            dtype=float,
        )
        barrier_mev = np.asarray(
            [float(row["barrier_ev"]) * 1e3 for row in subset],
            dtype=float,
        )
        valid = np.asarray([bool(row["trace_valid"]) for row in subset])

        label = kind.replace("_", " ").title()
        style = {
            "color": colors.get(kind, "tab:gray"),
            "marker": markers.get(kind, "o"),
            "linewidth": 1.8,
            "markersize": 6,
        }

        axes[0].plot(radii_um, height_um, label=label, **style)
        axes[1].plot(radii_um, barrier_mev, label=label, **style)

        if np.any(~valid):
            axes[0].scatter(
                radii_um[~valid],
                height_um[~valid],
                marker="x",
                s=70,
                color="red",
                zorder=5,
            )
            axes[1].scatter(
                radii_um[~valid],
                barrier_mev[~valid],
                marker="x",
                s=70,
                color="red",
                zorder=5,
            )

    axes[0].axhline(
        3.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="Nominal target: 3 um",
    )
    axes[0].set_xlabel("Central island radius / half-side [um]")
    axes[0].set_ylabel(r"Peak $|z-z_{\rm arm}|$ [um]")
    axes[0].set_title("RF-null volcano versus island size")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("Central island radius / half-side [um]")
    axes[1].set_ylabel("RF proxy barrier at 100 V [meV]")
    axes[1].set_title("Longitudinal RF barrier proxy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_height_profiles(
    profiles: list[dict[str, object]],
    *,
    output_path: str | Path,
) -> None:
    """Compare height-deviation profiles for selected candidates."""
    path = _prepare_path(output_path)

    fig, axis = plt.subplots(figsize=(10.0, 5.0))

    for profile in profiles:
        x_um = np.asarray(profile["x_m"], dtype=float) * 1e6
        deviation_um = np.asarray(profile["deviation_m"], dtype=float) * 1e6
        label = str(profile["label"])

        axis.plot(x_um, deviation_um, linewidth=2.0, label=label)

    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axhline(
        3.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label="Nominal target +3 um",
    )
    axis.axhline(
        -3.0,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel("x [um]")
    axis.set_ylabel(r"$z(x)-z_{\rm arm}$ [um]")
    axis.set_title("Selected RF-null height profiles")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_transition_scan_summary(
    results: list[dict[str, object]],
    *,
    output_path: str | Path,
) -> None:
    """Plot controlled taper candidates in volcano-barrier space."""
    path = _prepare_path(output_path)

    peak_um = np.asarray(
        [float(row["height_peak_m"]) * 1e6 for row in results],
        dtype=float,
    )
    barrier_mev = np.asarray(
        [float(row["barrier_ev"]) * 1e3 for row in results],
        dtype=float,
    )
    labels = [str(row["label"]) for row in results]
    valid = np.asarray([bool(row["trace_valid"]) for row in results])

    fig, axis = plt.subplots(figsize=(9.0, 6.5))

    axis.scatter(
        peak_um[valid],
        barrier_mev[valid],
        s=75,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
        label="Complete trace",
    )

    if np.any(~valid):
        finite_invalid = (~valid) & np.isfinite(peak_um) & np.isfinite(barrier_mev)
        if np.any(finite_invalid):
            axis.scatter(
                peak_um[finite_invalid],
                barrier_mev[finite_invalid],
                s=90,
                color="crimson",
                marker="x",
                linewidth=2.0,
                label="Incomplete trace",
            )

    for index, label in enumerate(labels):
        if np.isfinite(peak_um[index]) and np.isfinite(barrier_mev[index]):
            axis.annotate(
                label,
                (peak_um[index], barrier_mev[index]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    axis.axvline(
        3.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="Target peak dz = 3 um",
    )

    axis.set_xlabel(r"Peak $|z-z_{\rm arm}|$ [um]")
    axis.set_ylabel("RF proxy barrier at 100 V [meV]")
    axis.set_title("Controlled taper candidates: volcano-barrier trade-off")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)