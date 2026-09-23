"""Baseline geometric templates for planar four-arm X-junctions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ISLAND_NONE = "none"
ISLAND_SQUARE = "square"
ISLAND_CIRCLE = "circle"

VALID_ISLAND_KINDS = frozenset(
    {
        ISLAND_NONE,
        ISLAND_SQUARE,
        ISLAND_CIRCLE,
    }
)


@dataclass(frozen=True)
class XJunctionParameters:
    """Symmetric four-arm X-junction with parameterized RF boundaries."""

    ground_rail_width_m: float
    rf_rail_width_m: float
    arm_length_m: float
    outer_extent_m: float

    central_island_radius_m: float = 0.0
    central_island_kind: str = ISLAND_NONE

    taper_length_m: float = 0.0
    inner_edge_shift_at_centre_m: float = 0.0
    outer_edge_shift_at_centre_m: float = 0.0
    taper_power: float = 2.0

    rf_start_radius_override_m: float | None = None

    outer_bulge_amplitude_m: float = 0.0
    outer_bulge_center_m: float = 70e-6
    outer_bulge_sigma_m: float = 25e-6

    outer_contour_knots_m: tuple[float, ...] = ()
    outer_contour_offsets_m: tuple[float, ...] = ()

    inner_contour_knots_m: tuple[float, ...] = ()
    inner_contour_offsets_m: tuple[float, ...] = ()

    def validate(self) -> None:
        """Validate dimensions, taper, and piecewise-linear profiles."""
        values = (
            self.ground_rail_width_m,
            self.rf_rail_width_m,
            self.arm_length_m,
            self.outer_extent_m,
            self.central_island_radius_m,
            self.taper_length_m,
            self.inner_edge_shift_at_centre_m,
            self.outer_edge_shift_at_centre_m,
            self.taper_power,
            self.outer_bulge_amplitude_m,
            self.outer_bulge_center_m,
            self.outer_bulge_sigma_m,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("All scalar geometry parameters must be finite.")

        if self.ground_rail_width_m <= 0.0:
            raise ValueError("ground_rail_width_m must be positive.")
        if self.rf_rail_width_m <= 0.0:
            raise ValueError("rf_rail_width_m must be positive.")
        if self.arm_length_m <= 0.0:
            raise ValueError("arm_length_m must be positive.")
        if self.outer_extent_m <= self.arm_length_m:
            raise ValueError("outer_extent_m must exceed arm_length_m.")

        if self.central_island_radius_m < 0.0:
            raise ValueError("central_island_radius_m must be non-negative.")
        if self.central_island_kind not in VALID_ISLAND_KINDS:
            raise ValueError(
                f"Unknown island kind: {self.central_island_kind!r}."
            )
        if (
            self.central_island_kind == ISLAND_NONE
            and self.central_island_radius_m != 0.0
        ):
            raise ValueError(
                "central_island_radius_m must be zero when island kind is none."
            )

        if self.taper_length_m < 0.0:
            raise ValueError("taper_length_m must be non-negative.")
        if self.taper_length_m > self.arm_length_m:
            raise ValueError(
                "taper_length_m cannot exceed arm_length_m."
            )
        if self.taper_power <= 0.0:
            raise ValueError("taper_power must be positive.")

        if self.rf_start_radius_override_m is not None:
            if self.rf_start_radius_override_m < 0.0:
                raise ValueError(
                    "rf_start_radius_override_m must be non-negative."
                )
            if self.rf_start_radius_override_m > self.arm_length_m:
                raise ValueError(
                    "rf_start_radius_override_m cannot exceed arm_length_m."
                )

        if self.outer_bulge_center_m < 0.0:
            raise ValueError(
                "outer_bulge_center_m must be non-negative."
            )
        if self.outer_bulge_center_m > self.arm_length_m:
            raise ValueError(
                "outer_bulge_center_m cannot exceed arm_length_m."
            )
        if self.outer_bulge_sigma_m <= 0.0:
            raise ValueError(
                "outer_bulge_sigma_m must be positive."
            )

        self._validate_contour(
            name="outer",
            knots_m=self.outer_contour_knots_m,
            offsets_m=self.outer_contour_offsets_m,
        )
        self._validate_contour(
            name="inner",
            knots_m=self.inner_contour_knots_m,
            offsets_m=self.inner_contour_offsets_m,
        )

        sample_extent_m = max(
            self.taper_length_m,
            self.outer_bulge_center_m + 4.0 * self.outer_bulge_sigma_m,
            (
                max(self.outer_contour_knots_m)
                if self.outer_contour_knots_m
                else 0.0
            ),
            (
                max(self.inner_contour_knots_m)
                if self.inner_contour_knots_m
                else 0.0
            ),
        )
        sample_extent_m = min(sample_extent_m, self.arm_length_m)

        if sample_extent_m > 0.0:
            samples_m = np.linspace(0.0, sample_extent_m, 1001)
            inner_m, outer_m = self.rail_boundaries_m(samples_m)

            if np.min(inner_m) <= 0.5e-6:
                raise ValueError(
                    "Inner RF boundary reaches or crosses transport axis."
                )

            if np.any(outer_m <= inner_m):
                raise ValueError(
                    "RF inner and outer boundaries cross."
                )

    def _validate_contour(
        self,
        *,
        name: str,
        knots_m: tuple[float, ...],
        offsets_m: tuple[float, ...],
    ) -> None:
        """Validate a local piecewise-linear RF boundary profile."""
        knots = np.asarray(knots_m, dtype=float)
        offsets = np.asarray(offsets_m, dtype=float)

        if knots.size != offsets.size:
            raise ValueError(
                f"{name}_contour_knots_m and "
                f"{name}_contour_offsets_m must have equal length."
            )

        if knots.size == 0:
            return

        if knots.size < 2:
            raise ValueError(
                f"{name} contour requires at least two knots."
            )
        if not np.all(np.isfinite(knots)):
            raise ValueError(f"{name} contour knots must be finite.")
        if not np.all(np.isfinite(offsets)):
            raise ValueError(f"{name} contour offsets must be finite.")
        if np.any(knots < 0.0):
            raise ValueError(
                f"{name} contour knots must be non-negative."
            )
        if np.any(knots > self.arm_length_m):
            raise ValueError(
                f"{name} contour knots cannot exceed arm_length_m."
            )
        if np.any(np.diff(knots) <= 0.0):
            raise ValueError(
                f"{name} contour knots must be strictly increasing."
            )

        if abs(float(offsets[0])) > 1e-15:
            raise ValueError(
                f"First {name} contour offset must be zero."
            )
        if abs(float(offsets[-1])) > 1e-15:
            raise ValueError(
                f"Last {name} contour offset must be zero."
            )

        max_amplitude_m = 30e-6
        if np.max(np.abs(offsets)) > max_amplitude_m:
            raise ValueError(
                f"{name} contour offset exceeds 30 um provisional limit."
            )

        slopes = np.diff(offsets) / np.diff(knots)
        max_slope = 0.80

        if np.max(np.abs(slopes)) > max_slope:
            raise ValueError(
                f"{name} contour slope exceeds provisional 0.80 limit."
            )

    @property
    def half_ground_width_m(self) -> float:
        """Half-width of the straight grounded transport rail."""
        return 0.5 * self.ground_rail_width_m

    @property
    def rf_outer_radius_m(self) -> float:
        """Outer RF edge of the straight arm."""
        return self.half_ground_width_m + self.rf_rail_width_m

    @property
    def maximum_rf_outer_radius_m(self) -> float:
        """Conservative maximum outer RF radius for mesh refinement."""
        outer_offsets = np.asarray(
            self.outer_contour_offsets_m,
            dtype=float,
        )

        max_outer_offset_m = (
            max(0.0, float(np.max(outer_offsets)))
            if outer_offsets.size > 0
            else 0.0
        )

        return (
            self.rf_outer_radius_m
            + max(0.0, self.outer_edge_shift_at_centre_m)
            + max(0.0, self.outer_bulge_amplitude_m)
            + max_outer_offset_m
        )

    @property
    def rf_start_radius_m(self) -> float:
        """Nearest longitudinal coordinate at which an RF rail begins."""
        default_start_m = max(
            self.half_ground_width_m,
            self.central_island_radius_m,
        )

        if self.rf_start_radius_override_m is None:
            return default_start_m

        return float(self.rf_start_radius_override_m)

    def taper_envelope(
        self,
        longitudinal_radius_m: np.ndarray | float,
    ) -> np.ndarray:
        """Return one at centre and zero beyond taper length."""
        s_m = np.abs(
            np.asarray(longitudinal_radius_m, dtype=float)
        )

        if self.taper_length_m <= 0.0:
            return np.zeros_like(s_m, dtype=float)

        normalized = np.clip(
            1.0 - s_m / self.taper_length_m,
            0.0,
            1.0,
        )

        return normalized**self.taper_power

    def outer_bulge_profile_m(
        self,
        longitudinal_radius_m: np.ndarray | float,
    ) -> np.ndarray:
        """Return optional local Gaussian outer-boundary deformation."""
        s_m = np.abs(
            np.asarray(longitudinal_radius_m, dtype=float)
        )

        if self.outer_bulge_amplitude_m == 0.0:
            return np.zeros_like(s_m, dtype=float)

        normalized = (
            (s_m - self.outer_bulge_center_m)
            / self.outer_bulge_sigma_m
        )

        return self.outer_bulge_amplitude_m * np.exp(
            -0.5 * normalized**2
        )

    @staticmethod
    def _piecewise_linear_profile_m(
        longitudinal_radius_m: np.ndarray | float,
        *,
        knots_m: tuple[float, ...],
        offsets_m: tuple[float, ...],
    ) -> np.ndarray:
        """Evaluate a bounded local profile without spline overshoot."""
        s_m = np.abs(
            np.asarray(longitudinal_radius_m, dtype=float)
        )

        knots = np.asarray(knots_m, dtype=float)
        offsets = np.asarray(offsets_m, dtype=float)

        if knots.size == 0:
            return np.zeros_like(s_m, dtype=float)

        return np.interp(
            s_m,
            knots,
            offsets,
            left=0.0,
            right=0.0,
        )

    def outer_contour_profile_m(
        self,
        longitudinal_radius_m: np.ndarray | float,
    ) -> np.ndarray:
        """Return piecewise-linear offset of existing outer RF boundary."""
        return self._piecewise_linear_profile_m(
            longitudinal_radius_m,
            knots_m=self.outer_contour_knots_m,
            offsets_m=self.outer_contour_offsets_m,
        )

    def inner_contour_profile_m(
        self,
        longitudinal_radius_m: np.ndarray | float,
    ) -> np.ndarray:
        """Return piecewise-linear offset of existing inner RF boundary."""
        return self._piecewise_linear_profile_m(
            longitudinal_radius_m,
            knots_m=self.inner_contour_knots_m,
            offsets_m=self.inner_contour_offsets_m,
        )

    def rail_boundaries_m(
        self,
        longitudinal_radius_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return final inner and outer RF boundaries along an arm."""
        envelope = self.taper_envelope(longitudinal_radius_m)

        inner_m = (
            self.half_ground_width_m
            + self.inner_edge_shift_at_centre_m * envelope
            + self.inner_contour_profile_m(longitudinal_radius_m)
        )

        outer_m = (
            self.rf_outer_radius_m
            + self.outer_edge_shift_at_centre_m * envelope
            + self.outer_bulge_profile_m(longitudinal_radius_m)
            + self.outer_contour_profile_m(longitudinal_radius_m)
        )

        return (
            np.asarray(inner_m, dtype=float),
            np.asarray(outer_m, dtype=float),
        )


def _central_island_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: XJunctionParameters,
) -> np.ndarray:
    """Return central grounded-island mask."""
    radius_m = parameters.central_island_radius_m

    if parameters.central_island_kind == ISLAND_NONE or radius_m == 0.0:
        return np.zeros_like(x_m, dtype=bool)

    if parameters.central_island_kind == ISLAND_SQUARE:
        return (np.abs(x_m) <= radius_m) & (np.abs(y_m) <= radius_m)

    if parameters.central_island_kind == ISLAND_CIRCLE:
        return x_m**2 + y_m**2 <= radius_m**2

    raise RuntimeError("Unhandled validated island kind.")


def baseline_x_junction_rf_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    parameters: XJunctionParameters,
) -> np.ndarray:
    """Return RF mask for the validated baseline X-junction family."""
    parameters.validate()

    x_values_m = np.asarray(x_m, dtype=float)
    y_values_m = np.asarray(y_m, dtype=float)

    if x_values_m.shape != y_values_m.shape:
        raise ValueError("x_m and y_m must have identical shapes.")

    abs_x_m = np.abs(x_values_m)
    abs_y_m = np.abs(y_values_m)

    start_m = parameters.rf_start_radius_m
    arm_m = parameters.arm_length_m

    inner_x_m, outer_x_m = parameters.rail_boundaries_m(abs_x_m)
    horizontal_rf = (
        (abs_x_m >= start_m)
        & (abs_x_m <= arm_m)
        & (abs_y_m >= inner_x_m)
        & (abs_y_m <= outer_x_m)
    )

    inner_y_m, outer_y_m = parameters.rail_boundaries_m(abs_y_m)
    vertical_rf = (
        (abs_y_m >= start_m)
        & (abs_y_m <= arm_m)
        & (abs_x_m >= inner_y_m)
        & (abs_x_m <= outer_y_m)
    )

    rf_mask = horizontal_rf | vertical_rf
    island_mask = _central_island_mask(
        x_values_m,
        y_values_m,
        parameters,
    )

    return rf_mask & ~island_mask


def make_house_style_x_junction(
    *,
    ion_height_m: float,
    arm_length_m: float = 600e-6,
    outer_extent_m: float = 900e-6,
    central_island_radius_m: float = 0.0,
    central_island_kind: str = ISLAND_NONE,
    taper_length_m: float = 0.0,
    inner_edge_shift_at_centre_m: float = 0.0,
    outer_edge_shift_at_centre_m: float = 0.0,
    taper_power: float = 2.0,
    rf_start_radius_override_m: float | None = None,
    outer_bulge_amplitude_m: float = 0.0,
    outer_bulge_center_m: float = 70e-6,
    outer_bulge_sigma_m: float = 25e-6,
    outer_contour_knots_m: tuple[float, ...] = (),
    outer_contour_offsets_m: tuple[float, ...] = (),
    inner_contour_knots_m: tuple[float, ...] = (),
    inner_contour_offsets_m: tuple[float, ...] = (),
) -> XJunctionParameters:
    """Create a House-ratio planar X-junction."""
    return XJunctionParameters(
        ground_rail_width_m=0.83 * ion_height_m,
        rf_rail_width_m=1.99 * ion_height_m,
        arm_length_m=arm_length_m,
        outer_extent_m=outer_extent_m,
        central_island_radius_m=central_island_radius_m,
        central_island_kind=central_island_kind,
        taper_length_m=taper_length_m,
        inner_edge_shift_at_centre_m=inner_edge_shift_at_centre_m,
        outer_edge_shift_at_centre_m=outer_edge_shift_at_centre_m,
        taper_power=taper_power,
        rf_start_radius_override_m=rf_start_radius_override_m,
        outer_bulge_amplitude_m=outer_bulge_amplitude_m,
        outer_bulge_center_m=outer_bulge_center_m,
        outer_bulge_sigma_m=outer_bulge_sigma_m,
        outer_contour_knots_m=outer_contour_knots_m,
        outer_contour_offsets_m=outer_contour_offsets_m,
        inner_contour_knots_m=inner_contour_knots_m,
        inner_contour_offsets_m=inner_contour_offsets_m,
    )