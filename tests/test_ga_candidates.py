from __future__ import annotations

from core.ga.candidates import Candidate, N_CTRL, N_FEATURES


def test_candidate_defaults_have_expected_shapes() -> None:
    candidate = Candidate()
    assert candidate.inner_m.shape == (N_CTRL,)
    assert candidate.outer_m.shape == (N_CTRL,)
    assert candidate.feature_kind.shape == (N_FEATURES,)
    assert candidate.feature_operation.shape == (N_FEATURES,)


def test_candidate_copy_is_deep() -> None:
    candidate = Candidate()
    clone = candidate.copy()
    clone.inner_m[0] = 123.0
    assert candidate.inner_m[0] != clone.inner_m[0]