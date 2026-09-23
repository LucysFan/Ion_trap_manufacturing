from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


TOPOLOGY_OPEN_CROSS = 0
TOPOLOGY_CENTRAL_RF_DISK = 1
TOPOLOGY_CENTRAL_RF_RING = 2
TOPOLOGY_CENTRAL_RF_CROSS = 3
TOPOLOGY_GROUNDED_MOAT = 4

TOPOLOGY_NAMES: tuple[str, ...] = (
    "open_cross",
    "central_rf_disk",
    "central_rf_ring",
    "central_rf_cross",
    "grounded_moat",
)

FEATURE_NONE = 0
FEATURE_CIRCLE = 1
FEATURE_RECTANGLE = 2

FEATURE_KIND_NAMES: tuple[str, ...] = (
    "none",
    "circle",
    "rectangle",
)

FEATURE_OP_ADD_RF = 1
FEATURE_OP_REMOVE_RF = -1

N_CTRL: int = 8
N_FEATURES: int = 3

DEFAULT_BASE_INNER_M: float = 37.35e-6
DEFAULT_BASE_OUTER_M: float = 216.45e-6


@dataclass
class Genome:
    inner_control_m: np.ndarray = field(
        default_factory=lambda: np.full(N_CTRL, DEFAULT_BASE_INNER_M, dtype=np.float64)
    )
    outer_control_m: np.ndarray = field(
        default_factory=lambda: np.full(N_CTRL, DEFAULT_BASE_OUTER_M, dtype=np.float64)
    )

    topology: int = TOPOLOGY_OPEN_CROSS
    topology_size_m: float = 55e-6
    topology_width_m: float = 18e-6

    feature_kind: np.ndarray = field(
        default_factory=lambda: np.zeros(N_FEATURES, dtype=np.int8)
    )
    feature_operation: np.ndarray = field(
        default_factory=lambda: np.ones(N_FEATURES, dtype=np.int8)
    )
    feature_radius_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 110e-6, dtype=np.float64)
    )
    feature_theta_rad: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, np.pi / 8.0, dtype=np.float64)
    )
    feature_p1_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 20e-6, dtype=np.float64)
    )
    feature_p2_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 20e-6, dtype=np.float64)
    )
    feature_angle_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(N_FEATURES, dtype=np.float64)
    )

    INNER_MIN_M: ClassVar[float] = 5e-6
    INNER_MAX_M: ClassVar[float] = 180e-6
    OUTER_MIN_M: ClassVar[float] = 30e-6
    OUTER_MAX_M: ClassVar[float] = 360e-6
    TOPOLOGY_SIZE_MIN_M: ClassVar[float] = 12e-6
    TOPOLOGY_SIZE_MAX_M: ClassVar[float] = 170e-6
    TOPOLOGY_WIDTH_MIN_M: ClassVar[float] = 6e-6
    TOPOLOGY_WIDTH_MAX_M: ClassVar[float] = 70e-6
    FEATURE_RADIUS_MIN_M: ClassVar[float] = 20e-6
    FEATURE_RADIUS_MAX_M: ClassVar[float] = 320e-6
    FEATURE_P_MIN_M: ClassVar[float] = 6e-6
    FEATURE_P_MAX_M: ClassVar[float] = 100e-6

    def __post_init__(self) -> None:
        self.inner_control_m = np.asarray(self.inner_control_m, dtype=np.float64).copy()
        self.outer_control_m = np.asarray(self.outer_control_m, dtype=np.float64).copy()

        self.feature_kind = np.asarray(self.feature_kind, dtype=np.int8).copy()
        self.feature_operation = np.asarray(self.feature_operation, dtype=np.int8).copy()
        self.feature_radius_m = np.asarray(self.feature_radius_m, dtype=np.float64).copy()
        self.feature_theta_rad = np.asarray(self.feature_theta_rad, dtype=np.float64).copy()
        self.feature_p1_m = np.asarray(self.feature_p1_m, dtype=np.float64).copy()
        self.feature_p2_m = np.asarray(self.feature_p2_m, dtype=np.float64).copy()
        self.feature_angle_rad = np.asarray(self.feature_angle_rad, dtype=np.float64).copy()

        self._check_shapes()

    def _check_shapes(self) -> None:
        if self.inner_control_m.shape != (N_CTRL,):
            raise ValueError(f"inner_control_m must have shape {(N_CTRL,)}")
        if self.outer_control_m.shape != (N_CTRL,):
            raise ValueError(f"outer_control_m must have shape {(N_CTRL,)}")

        if self.feature_kind.shape != (N_FEATURES,):
            raise ValueError(f"feature_kind must have shape {(N_FEATURES,)}")
        if self.feature_operation.shape != (N_FEATURES,):
            raise ValueError(f"feature_operation must have shape {(N_FEATURES,)}")
        if self.feature_radius_m.shape != (N_FEATURES,):
            raise ValueError(f"feature_radius_m must have shape {(N_FEATURES,)}")
        if self.feature_theta_rad.shape != (N_FEATURES,):
            raise ValueError(f"feature_theta_rad must have shape {(N_FEATURES,)}")
        if self.feature_p1_m.shape != (N_FEATURES,):
            raise ValueError(f"feature_p1_m must have shape {(N_FEATURES,)}")
        if self.feature_p2_m.shape != (N_FEATURES,):
            raise ValueError(f"feature_p2_m must have shape {(N_FEATURES,)}")
        if self.feature_angle_rad.shape != (N_FEATURES,):
            raise ValueError(f"feature_angle_rad must have shape {(N_FEATURES,)}")

    def copy(self) -> "Genome":
        return Genome(
            inner_control_m=self.inner_control_m.copy(),
            outer_control_m=self.outer_control_m.copy(),
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

    @classmethod
    def vector_length(cls) -> int:
        return (
            N_CTRL
            + N_CTRL
            + 1
            + 1
            + 1
            + N_FEATURES
            + N_FEATURES
            + N_FEATURES
            + N_FEATURES
            + N_FEATURES
            + N_FEATURES
            + N_FEATURES
        )

    def to_vector(self) -> np.ndarray:
        parts = [
            self.inner_control_m.astype(np.float64, copy=False),
            self.outer_control_m.astype(np.float64, copy=False),
            np.asarray([float(self.topology)], dtype=np.float64),
            np.asarray([self.topology_size_m], dtype=np.float64),
            np.asarray([self.topology_width_m], dtype=np.float64),
            self.feature_kind.astype(np.float64, copy=False),
            self.feature_operation.astype(np.float64, copy=False),
            self.feature_radius_m.astype(np.float64, copy=False),
            self.feature_theta_rad.astype(np.float64, copy=False),
            self.feature_p1_m.astype(np.float64, copy=False),
            self.feature_p2_m.astype(np.float64, copy=False),
            self.feature_angle_rad.astype(np.float64, copy=False),
        ]
        vector = np.concatenate(parts)
        if vector.shape != (self.vector_length(),):
            raise RuntimeError("Genome vector has unexpected length")
        return vector

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "Genome":
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        expected = cls.vector_length()
        if values.size != expected:
            raise ValueError(f"Expected vector of length {expected}, got {values.size}")

        offset = 0

        inner = values[offset : offset + N_CTRL]
        offset += N_CTRL

        outer = values[offset : offset + N_CTRL]
        offset += N_CTRL

        topology = int(round(values[offset]))
        offset += 1

        topology_size_m = float(values[offset])
        offset += 1

        topology_width_m = float(values[offset])
        offset += 1

        feature_kind = np.rint(values[offset : offset + N_FEATURES]).astype(np.int8)
        offset += N_FEATURES

        feature_operation = np.rint(values[offset : offset + N_FEATURES]).astype(np.int8)
        offset += N_FEATURES

        feature_radius_m = values[offset : offset + N_FEATURES]
        offset += N_FEATURES

        feature_theta_rad = values[offset : offset + N_FEATURES]
        offset += N_FEATURES

        feature_p1_m = values[offset : offset + N_FEATURES]
        offset += N_FEATURES

        feature_p2_m = values[offset : offset + N_FEATURES]
        offset += N_FEATURES

        feature_angle_rad = values[offset : offset + N_FEATURES]
        offset += N_FEATURES

        if offset != expected:
            raise RuntimeError("Genome vector parsing ended at unexpected offset")

        return cls(
            inner_control_m=inner,
            outer_control_m=outer,
            topology=topology,
            topology_size_m=topology_size_m,
            topology_width_m=topology_width_m,
            feature_kind=feature_kind,
            feature_operation=feature_operation,
            feature_radius_m=feature_radius_m,
            feature_theta_rad=feature_theta_rad,
            feature_p1_m=feature_p1_m,
            feature_p2_m=feature_p2_m,
            feature_angle_rad=feature_angle_rad,
        )

    def clip(self) -> "Genome":
        clipped = self.copy()

        clipped.inner_control_m = np.clip(
            clipped.inner_control_m,
            self.INNER_MIN_M,
            self.INNER_MAX_M,
        )
        clipped.outer_control_m = np.clip(
            clipped.outer_control_m,
            self.OUTER_MIN_M,
            self.OUTER_MAX_M,
        )

        clipped.topology = int(np.clip(clipped.topology, 0, len(TOPOLOGY_NAMES) - 1))
        clipped.topology_size_m = float(
            np.clip(
                clipped.topology_size_m,
                self.TOPOLOGY_SIZE_MIN_M,
                self.TOPOLOGY_SIZE_MAX_M,
            )
        )
        clipped.topology_width_m = float(
            np.clip(
                clipped.topology_width_m,
                self.TOPOLOGY_WIDTH_MIN_M,
                self.TOPOLOGY_WIDTH_MAX_M,
            )
        )

        clipped.feature_kind = np.clip(
            clipped.feature_kind,
            FEATURE_NONE,
            FEATURE_RECTANGLE,
        ).astype(np.int8)

        clipped.feature_operation = np.where(
            clipped.feature_operation >= 0,
            FEATURE_OP_ADD_RF,
            FEATURE_OP_REMOVE_RF,
        ).astype(np.int8)

        clipped.feature_radius_m = np.clip(
            clipped.feature_radius_m,
            self.FEATURE_RADIUS_MIN_M,
            self.FEATURE_RADIUS_MAX_M,
        )
        clipped.feature_theta_rad = np.mod(clipped.feature_theta_rad, np.pi / 4.0)
        clipped.feature_p1_m = np.clip(
            clipped.feature_p1_m,
            self.FEATURE_P_MIN_M,
            self.FEATURE_P_MAX_M,
        )
        clipped.feature_p2_m = np.clip(
            clipped.feature_p2_m,
            self.FEATURE_P_MIN_M,
            self.FEATURE_P_MAX_M,
        )
        clipped.feature_angle_rad = (clipped.feature_angle_rad + np.pi) % (2.0 * np.pi) - np.pi

        return clipped

    def is_finite(self) -> bool:
        arrays = (
            self.inner_control_m,
            self.outer_control_m,
            self.feature_radius_m,
            self.feature_theta_rad,
            self.feature_p1_m,
            self.feature_p2_m,
            self.feature_angle_rad,
        )
        return all(np.all(np.isfinite(array)) for array in arrays) and np.isfinite(
            [self.topology_size_m, self.topology_width_m]
        ).all()

    def is_physically_valid(self, *, min_rf_width_m: float = 18e-6) -> bool:
        if not self.is_finite():
            return False

        if np.any(self.outer_control_m - self.inner_control_m < min_rf_width_m):
            return False

        if self.topology < 0 or self.topology >= len(TOPOLOGY_NAMES):
            return False

        if np.any((self.feature_kind < FEATURE_NONE) | (self.feature_kind > FEATURE_RECTANGLE)):
            return False

        if np.any((self.feature_operation != FEATURE_OP_ADD_RF) & (self.feature_operation != FEATURE_OP_REMOVE_RF)):
            return False

        return True

    def summary(self) -> dict[str, object]:
        return {
            "topology": TOPOLOGY_NAMES[int(np.clip(self.topology, 0, len(TOPOLOGY_NAMES) - 1))],
            "topology_size_m": float(self.topology_size_m),
            "topology_width_m": float(self.topology_width_m),
            "inner_min_m": float(np.min(self.inner_control_m)),
            "inner_max_m": float(np.max(self.inner_control_m)),
            "outer_min_m": float(np.min(self.outer_control_m)),
            "outer_max_m": float(np.max(self.outer_control_m)),
            "active_features": int(np.sum(self.feature_kind != FEATURE_NONE)),
        }


def random_genome(
    rng: np.random.Generator,
    *,
    base_inner_m: float = DEFAULT_BASE_INNER_M,
    base_outer_m: float = DEFAULT_BASE_OUTER_M,
    broad: bool = False,
) -> Genome:
    if broad:
        genome = Genome(
            inner_control_m=rng.uniform(12e-6, 115e-6, size=N_CTRL),
            outer_control_m=rng.uniform(120e-6, 310e-6, size=N_CTRL),
            topology=int(rng.integers(0, len(TOPOLOGY_NAMES))),
            topology_size_m=float(rng.uniform(18e-6, 150e-6)),
            topology_width_m=float(rng.uniform(7e-6, 60e-6)),
            feature_kind=rng.integers(0, 3, size=N_FEATURES, dtype=np.int8),
            feature_operation=rng.choice(
                np.asarray([FEATURE_OP_REMOVE_RF, FEATURE_OP_ADD_RF], dtype=np.int8),
                size=N_FEATURES,
            ),
            feature_radius_m=rng.uniform(30e-6, 310e-6, size=N_FEATURES),
            feature_theta_rad=rng.uniform(0.0, np.pi / 4.0, size=N_FEATURES),
            feature_p1_m=rng.uniform(7e-6, 85e-6, size=N_FEATURES),
            feature_p2_m=rng.uniform(7e-6, 85e-6, size=N_FEATURES),
            feature_angle_rad=rng.uniform(-np.pi, np.pi, size=N_FEATURES),
        )
    else:
        envelope = np.linspace(1.0, 0.0, N_CTRL)
        genome = Genome(
            inner_control_m=base_inner_m + rng.normal(0.0, 18e-6, size=N_CTRL) * envelope,
            outer_control_m=base_outer_m + rng.normal(0.0, 24e-6, size=N_CTRL) * envelope,
            topology=int(rng.integers(0, len(TOPOLOGY_NAMES))),
            topology_size_m=float(rng.uniform(25e-6, 100e-6)),
            topology_width_m=float(rng.uniform(8e-6, 35e-6)),
            feature_kind=np.zeros(N_FEATURES, dtype=np.int8),
            feature_operation=np.ones(N_FEATURES, dtype=np.int8),
            feature_radius_m=np.full(N_FEATURES, 110e-6, dtype=np.float64),
            feature_theta_rad=np.full(N_FEATURES, np.pi / 8.0, dtype=np.float64),
            feature_p1_m=np.full(N_FEATURES, 20e-6, dtype=np.float64),
            feature_p2_m=np.full(N_FEATURES, 20e-6, dtype=np.float64),
            feature_angle_rad=np.zeros(N_FEATURES, dtype=np.float64),
        )

        for index in range(N_FEATURES):
            if rng.random() < 0.55:
                genome.feature_kind[index] = int(rng.integers(1, 3))
                genome.feature_operation[index] = int(
                    rng.choice(np.asarray([FEATURE_OP_REMOVE_RF, FEATURE_OP_ADD_RF], dtype=np.int8))
                )
                genome.feature_radius_m[index] = float(rng.uniform(45e-6, 260e-6))
                genome.feature_theta_rad[index] = float(rng.uniform(0.0, np.pi / 4.0))
                genome.feature_p1_m[index] = float(rng.uniform(8e-6, 50e-6))
                genome.feature_p2_m[index] = float(rng.uniform(8e-6, 50e-6))
                genome.feature_angle_rad[index] = float(rng.uniform(-np.pi, np.pi))

    return genome.clip()