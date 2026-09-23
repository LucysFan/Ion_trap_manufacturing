"""Step 07b: visual smoke test of the polygonal/chamfered X-junction family."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.targets import TARGET_ION_HEIGHT_M
from core.geometry.chamfered_junction import (
    chamfered_x_junction_rf_mask,
    make_chamfered_house_x_junction,
)


def plot_mask(
    mask: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Plot a high-resolution logical RF mask, not a quadtree panel view."""
    figure, axis = plt.subplots(figsize=(8.0, 8.0))

    cmap = ListedColormap(["#c7d9f1", "#203b9a"])

    axis.pcolormesh(
        x_m * 1e6,
        y_m * 1e6,
        mask.astype(float),
        cmap=cmap,
        shading="auto",
    )

    axis.set_aspect("equal")
    axis.set_xlim(-220.0, 220.0)
    axis.set_ylim(-220.0, 220.0)
    axis.set_xlabel("x [um]")
    axis.set_ylabel("y [um]")
    axis.set_title(title)
    axis.grid(alpha=0.18)

    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_boundaries(
    parameter_sets: list[tuple[str, object]],
    *,
    output_path: Path,
) -> None:
    """Plot inner and outer rail boundaries over the transition."""
    figure, axis = plt.subplots(figsize=(9.0, 5.0))

    s_m = np.linspace(0.0, 230e-6, 801)

    for label, parameters in parameter_sets:
        inner_m, outer_m = parameters.rail_boundaries_m(s_m)

        axis.plot(
            s_m * 1e6,
            inner_m * 1e6,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )
        axis.plot(
            s_m * 1e6,
            outer_m * 1e6,
            linewidth=2.2,
            label=label,
        )

    axis.set_xlabel(r"$s=|x|$ or $|y|$ [um]")
    axis.set_ylabel("Boundary distance from arm axis [um]")
    axis.set_title(
        "Chamfered central transition: solid outer edge, dashed inner edge"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_directory = (
        Path("reports")
        / "figures"
        / "07b_chamfered_geometry_smoke_test"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Logical high-resolution grid. This deliberately shows the intended
    # geometry without quadtree/panel stair-stepping.
    x_edges_m = np.linspace(-230e-6, 230e-6, 1001)
    y_edges_m = np.linspace(-230e-6, 230e-6, 1001)

    x_centres_m = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y_centres_m = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])

    x_grid_m, y_grid_m = np.meshgrid(
        x_centres_m,
        y_centres_m,
        indexing="xy",
    )

    cases = {
        "mild_chamfer": dict(
            rf_start_radius_m=30e-6,
            transition_radius_m=160e-6,
            inner_radius_at_centre_m=20e-6,
            outer_radius_at_centre_m=155e-6,
        ),
        "wide_chamfer": dict(
            rf_start_radius_m=30e-6,
            transition_radius_m=180e-6,
            inner_radius_at_centre_m=16e-6,
            outer_radius_at_centre_m=135e-6,
        ),
        "narrow_chamfer": dict(
            rf_start_radius_m=35e-6,
            transition_radius_m=150e-6,
            inner_radius_at_centre_m=25e-6,
            outer_radius_at_centre_m=115e-6,
        ),
    }

    parameter_sets: list[tuple[str, object]] = []

    print("Step 07b: polygonal/chamfered X-junction smoke test")
    print("=" * 96)

    for label, kwargs in cases.items():
        parameters = make_chamfered_house_x_junction(
            ion_height_m=TARGET_ION_HEIGHT_M,
            arm_length_m=600e-6,
            outer_extent_m=900e-6,
            **kwargs,
        )
        parameters.validate()

        mask = chamfered_x_junction_rf_mask(
            x_grid_m,
            y_grid_m,
            parameters,
        )

        parameter_sets.append((label, parameters))

        inner_m, outer_m = parameters.rail_boundaries_m(
            np.array(
                [
                    parameters.rf_start_radius_m,
                    parameters.transition_radius_m,
                    220e-6,
                ]
            )
        )

        print()
        print(f"=== {label} ===")
        print(
            f"start={parameters.rf_start_radius_m * 1e6:.1f} um, "
            f"transition={parameters.transition_radius_m * 1e6:.1f} um"
        )
        print(
            "inner boundary at start / transition / arm [um]: "
            + ", ".join(f"{value * 1e6:.3f}" for value in inner_m)
        )
        print(
            "outer boundary at start / transition / arm [um]: "
            + ", ".join(f"{value * 1e6:.3f}" for value in outer_m)
        )
        print(
            "RF width at start / transition / arm [um]: "
            + ", ".join(
                f"{value * 1e6:.3f}"
                for value in (outer_m - inner_m)
            )
        )

        plot_mask(
            mask,
            x_edges_m,
            y_edges_m,
            title=f"07b chamfered RF mask: {label}",
            output_path=output_directory / f"mask_{label}.png",
        )

    plot_boundaries(
        parameter_sets,
        output_path=output_directory / "chamfered_boundary_profiles.png",
    )

    print()
    print(f"Saved figures to: {output_directory}")


if __name__ == "__main__":
    main()