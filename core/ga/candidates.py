from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOPOLOGIES = (
    "open_cross",
    "central_rf_disk",
    "central_rf_ring",
    "central_rf_cross",
    "grounded_moat",
)
N_CTRL = 8
N_FEATURES = 3


@dataclass
class Candidate:
    inner_m: np.ndarray = field(default_factory=lambda: np.zeros(N_CTRL))
    outer_m: np.ndarray = field(default_factory=lambda: np.zeros(N_CTRL))
    topology: int = 0
    topology_size_m: float = 55e-6
    topology_width_m: float = 18e-6
    feature_kind: np.ndarray = field(
        default_factory=lambda: np.zeros(N_FEATURES, dtype=np.int8)
    )
    feature_operation: np.ndarray = field(
        default_factory=lambda: np.ones(N_FEATURES, dtype=np.int8)
    )
    feature_radius_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 110e-6)
    )
    feature_theta_rad: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, np.pi / 8.0)
    )
    feature_p1_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 20e-6)
    )
    feature_p2_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 20e-6)
    )
    feature_angle_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(N_FEATURES)
    )

    def copy(self) -> "Candidate":
        return Candidate(
            inner_m=self.inner_m.copy(),
            outer_m=self.outer_m.copy(),
            topology=int(self.topology),
            topology_size_m=float(self.topology_size_m),
            topology_width_m=float(self.topology_width_m),
            feature_kind=self.feature_kind.copy(),
            feature_operation=self.feature_operation.copy(),
            feature_radius_m=self.feature_radius_m.copy(),
            feature_theta_rad=self.feature_theta_rad.copy(),
            feature_p1_m=self.feature_p1_m.copy(),
            feature_p2_m=self.feature_p2_m.copy(),
            feature_angle_rad=self.feature_angle_rad.copy(),
        )


@dataclass
class Evaluation:
    objectives: np.ndarray
    metrics: dict[str, object]


__all__ = [
    "Candidate",
    "Evaluation",
    "N_CTRL",
    "N_FEATURES",
    "TOPOLOGIES",
]