"""Tests for the local three-dimensional RF-null solver."""

from __future__ import annotations

import numpy as np

from core.analysis.rf_null import find_rf_null


class LinearFieldModel:
    """Synthetic model with exact RF-null at one chosen point."""

    def __init__(self, null_position_m: np.ndarray) -> None:
        self.null_position_m = np.asarray(null_position_m, dtype=float)

    def electric_field(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
    ) -> tuple[float, float, float]:
        point_m = np.array([x_m, y_m, z_m], dtype=float)
        field = point_m - self.null_position_m
        return tuple(field)


def test_find_rf_null_on_synthetic_linear_field() -> None:
    expected_m = np.array([2e-6, -3e-6, 90e-6])
    model = LinearFieldModel(expected_m)

    result = find_rf_null(
        model,
        initial_guess_m=np.array([8e-6, 4e-6, 100e-6]),
        residual_tolerance_v_m=1e-10,
    )

    assert result.converged
    assert np.allclose(result.position_m, expected_m, atol=1e-12)
    assert result.residual_field_v_m < 1e-10