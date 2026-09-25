from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ga.candidates import Candidate
from core.geometry.rf_mask import build_masks, rf_mask


@dataclass
class DummyConfig:
    arm_length_m: float = 900e-6


def test_rf_mask_returns_float_array() -> None:
    cfg = DummyConfig()
    candidate = Candidate(
        inner_m=np.full(8, 40e-6),
        outer_m=np.full(8, 200e-6),
    )
    x = np.array([0.0, 100e-6, 300e-6])
    y = np.array([50e-6, 0.0, 0.0])
    mask = rf_mask(candidate, x, y, cfg)
    assert mask.dtype == float
    assert mask.shape == x.shape


def test_rf_mask_is_c4v_symmetric_for_baseline_candidate() -> None:
    cfg = DummyConfig()
    candidate = Candidate(
        inner_m=np.full(8, 40e-6),
        outer_m=np.full(8, 200e-6),
    )
    x = np.array([120e-6])
    y = np.array([60e-6])
    a = rf_mask(candidate, x, y, cfg)[0]
    b = rf_mask(candidate, y, x, cfg)[0]
    assert a == b


def test_build_masks_stacks_population() -> None:
    cfg = DummyConfig()
    population = [
        Candidate(inner_m=np.full(8, 40e-6), outer_m=np.full(8, 200e-6)),
        Candidate(inner_m=np.full(8, 50e-6), outer_m=np.full(8, 210e-6)),
    ]
    x = np.array([0.0, 50e-6])
    y = np.array([60e-6, 0.0])
    masks = build_masks(population, x, y, cfg)
    assert masks.shape == (2, 2)