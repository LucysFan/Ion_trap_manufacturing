from __future__ import annotations

import numpy as np

from core.geometry.rf_shapes import rectangle_mask, spline_values


def test_spline_values_preserve_endpoints() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    s = np.array([0.0, 1.0])
    result = spline_values(values, s)
    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], 4.0)


def test_rectangle_mask_contains_center() -> None:
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 2.0])
    mask = rectangle_mask(x, y, 0.0, 0.0, 1.0, 1.0, 0.0)
    assert bool(mask[0])
    assert not bool(mask[1])