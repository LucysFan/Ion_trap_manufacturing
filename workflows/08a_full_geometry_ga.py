"""Single-population NSGA-II search of a static planar X-junction RF geometry.

Uses the project's real modules:
    core.electrostatics.bem.BEM2D
    core.electrostatics.bem_mesh.geometry_aware_quadtree_rectangular_panels

No core.maskbem import. No island model. CPU requires only NumPy/SciPy.
CUDA is optional and imported only when explicitly selected or available in auto mode.

Run from the project root:
    python workflows/08a_full_geometry_ga.py --backend cpu --quick
    python workflows/08a_full_geometry_ga.py --backend auto --quick
    python workflows/08a_full_geometry_ga.py --backend cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.electrostatics.bem import BEM2D, COULOMB_CONSTANT
from core.electrostatics.bem_mesh import (
    geometry_aware_quadtree_rectangular_panels,
)

try:
    from config.physical_constants import (
        CHARGE_CA40_C,
        MASS_CA40_KG,
        RF_FREQ_NOMINAL_HZ,
        TARGET_HEIGHT_M,
    )
except ImportError:
    # Constants are explicitly defined here rather than silently changing
    # the BEM implementation. Check names in your config if they differ.
    CHARGE_CA40_C = 1.602176634e-19
    MASS_CA40_KG = 39.96259098 * 1.66053906660e-27
    RF_FREQ_NOMINAL_HZ = 20e6
    TARGET_HEIGHT_M = 90e-6


N_CTRL = 8
N_FEATURES = 3

TOPOLOGIES = (
    "open_cross",
    "central_rf_disk",
    "central_rf_ring",
    "central_rf_cross",
    "grounded_moat",
)

OBJECTIVE_NAMES = (
    "height_error",
    "lateral_error",
    "rf_barrier",
    "frequency_error",
    "anisotropy",
    "geometry_penalty",
)


@dataclass
class Config:
    seed: int = 20260923
    population: int = 32
    generations: int = 20
    offspring: int = 32

    path_points: int = 25
    route_extent_m: float = 350e-6

    arm_length_m: float = 900e-6
    outer_extent_m: float = 1150e-6

    baseline_inner_m: float = 37.35e-6
    baseline_outer_m: float = 216.45e-6
    min_rf_width_m: float = 18e-6

    rf_peak_voltage_v: float = 100.0
    target_frequency_hz: float = 1.5e6

    trace_iterations: int = 10
    derivative_step_m: float = 0.35e-6
    maximum_newton_step_m: float = 6e-6

    minimum_z_m: float = 25e-6
    maximum_z_m: float = 220e-6
    maximum_abs_y_m: float = 60e-6

    mutation_probability: float = 0.22
    topology_mutation_probability: float = 0.06

    # Keep the quick mesh modest: dense BEM memory grows as n_panels**2.
    central_half_extent_m: float = 180e-6
    central_max_cell_m: float = 80e-6
    boundary_max_cell_m: float = 40e-6
    outer_max_cell_m: float = 250e-6
    min_cell_m: float = 10e-6

    # Refuse accidental multi-gigabyte CPU allocations.
    max_panels: int = 1600
    checkpoint_every: int = 5


@dataclass
class Candidate:
    inner_m: np.ndarray = field(
        default_factory=lambda: np.full(N_CTRL, 37.35e-6)
    )
    outer_m: np.ndarray = field(
        default_factory=lambda: np.full(N_CTRL, 216.45e-6)
    )

    topology: int = 0
    topology_size_m: float = 55e-6
    topology_width_m: float = 18e-6

    # Feature kind: 0 absent, 1 circle, 2 rectangle.
    feature_kind: np.ndarray = field(
        default_factory=lambda: np.zeros(N_FEATURES, dtype=np.int8)
    )
    # +1 add RF, -1 remove RF.
    feature_operation: np.ndarray = field(
        default_factory=lambda: np.ones(N_FEATURES, dtype=np.int8)
    )
    feature_radius_m: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, 110e-6)
    )
    feature_theta_rad: np.ndarray = field(
        default_factory=lambda: np.full(N_FEATURES, np.pi / 8)
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
    metrics: dict[str, float | bool | str]


def repair(candidate: Candidate, cfg: Config) -> Candidate:
    result = candidate.copy()

    result.inner_m = np.clip(result.inner_m, 5e-6, 180e-6)
    result.outer_m = np.clip(result.outer_m, 30e-6, 360e-6)
    result.outer_m = np.maximum(
        result.outer_m,
        result.inner_m + cfg.min_rf_width_m,
    )

    # The remote arm must join the reference linear-trap section.
    result.inner_m[-1] = cfg.baseline_inner_m
    result.outer_m[-1] = cfg.baseline_outer_m

    result.topology = int(
        np.clip(result.topology, 0, len(TOPOLOGIES) - 1)
    )
    result.topology_size_m = float(
        np.clip(result.topology_size_m, 12e-6, 170e-6)
    )
    result.topology_width_m = float(
        np.clip(result.topology_width_m, 6e-6, 70e-6)
    )

    result.feature_kind = np.clip(
        result.feature_kind, 0, 2
    ).astype(np.int8)
    result.feature_operation = np.where(
        result.feature_operation >= 0, 1, -1
    ).astype(np.int8)

    result.feature_radius_m = np.clip(
        result.feature_radius_m, 20e-6, 320e-6
    )
    result.feature_theta_rad = np.mod(
        result.feature_theta_rad, np.pi / 4
    )
    result.feature_p1_m = np.clip(
        result.feature_p1_m, 6e-6, 100e-6
    )
    result.feature_p2_m = np.clip(
        result.feature_p2_m, 6e-6, 100e-6
    )
    result.feature_angle_rad = (
        result.feature_angle_rad + np.pi
    ) % (2 * np.pi) - np.pi

    return result


def baseline(cfg: Config) -> Candidate:
    return repair(
        Candidate(
            inner_m=np.full(N_CTRL, cfg.baseline_inner_m),
            outer_m=np.full(N_CTRL, cfg.baseline_outer_m),
        ),
        cfg,
    )


def spline_values(control: np.ndarray, normalized_s: np.ndarray) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, len(control))
    spline = CubicSpline(knots, control, bc_type="natural")
    return spline(np.clip(normalized_s, 0.0, 1.0))


def rectangle_mask(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    dx = x - cx
    dy = y - cy
    xr = cosine * dx + sine * dy
    yr = -sine * dx + cosine * dy
    return (np.abs(xr) <= width / 2) & (np.abs(yr) <= height / 2)


def rf_mask(
    candidate: Candidate,
    x_m: np.ndarray,
    y_m: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)

    ax = np.abs(x)
    ay = np.abs(y)

    along_x = ax >= ay
    longitudinal = np.where(along_x, ax, ay)
    transverse = np.where(along_x, ay, ax)
    normalized_s = longitudinal / cfg.arm_length_m

    inner = spline_values(candidate.inner_m, normalized_s)
    outer = spline_values(candidate.outer_m, normalized_s)

    mask = (
        (longitudinal <= cfg.arm_length_m)
        & (transverse >= inner)
        & (transverse <= outer)
    )

    radius = np.hypot(x, y)
    size = candidate.topology_size_m
    width = candidate.topology_width_m

    if candidate.topology == 1:
        mask |= radius <= size
    elif candidate.topology == 2:
        mask |= np.abs(radius - size) <= width / 2
    elif candidate.topology == 3:
        mask |= (
            (radius <= size)
            & ((ax <= width / 2) | (ay <= width / 2))
        )
    elif candidate.topology == 4:
        mask &= np.abs(radius - size) > width / 2

    for feature_index in range(N_FEATURES):
        kind = int(candidate.feature_kind[feature_index])
        if kind == 0:
            continue

        feature_mask = np.zeros_like(mask, dtype=bool)
        radial_position = candidate.feature_radius_m[feature_index]
        theta0 = candidate.feature_theta_rad[feature_index]
        p1 = candidate.feature_p1_m[feature_index]
        p2 = candidate.feature_p2_m[feature_index]
        angle0 = candidate.feature_angle_rad[feature_index]

        for reflection in (-1.0, 1.0):
            for quarter_turn in range(4):
                theta = reflection * theta0 + quarter_turn * np.pi / 2
                cx = radial_position * np.cos(theta)
                cy = radial_position * np.sin(theta)

                if kind == 1:
                    primitive = (x - cx) ** 2 + (y - cy) ** 2 <= p1**2
                else:
                    primitive = rectangle_mask(
                        x,
                        y,
                        cx,
                        cy,
                        p1,
                        p2,
                        reflection * angle0 + quarter_turn * np.pi / 2,
                    )

                feature_mask |= primitive

        if candidate.feature_operation[feature_index] > 0:
            mask |= feature_mask
        else:
            mask &= ~feature_mask

    return mask.astype(np.float64)


def seeded_candidate(
    cfg: Config,
    rng: np.random.Generator,
    *,
    broad: bool,
) -> Candidate:
    candidate = baseline(cfg)

    if broad:
        candidate.inner_m[:-1] = rng.uniform(
            12e-6, 115e-6, N_CTRL - 1
        )
        candidate.outer_m[:-1] = rng.uniform(
            120e-6, 310e-6, N_CTRL - 1
        )
    else:
        envelope = np.linspace(1.0, 0.0, N_CTRL)
        candidate.inner_m += rng.normal(
            0.0, 18e-6, N_CTRL
        ) * envelope
        candidate.outer_m += rng.normal(
            0.0, 24e-6, N_CTRL
        ) * envelope

    candidate.topology = int(rng.integers(len(TOPOLOGIES)))
    candidate.topology_size_m = float(rng.uniform(25e-6, 100e-6))
    candidate.topology_width_m = float(rng.uniform(8e-6, 35e-6))

    for i in range(N_FEATURES):
        if rng.random() < (0.7 if broad else 0.35):
            candidate.feature_kind[i] = rng.integers(1, 3)
            candidate.feature_operation[i] = rng.choice((-1, 1))
            candidate.feature_radius_m[i] = rng.uniform(45e-6, 260e-6)
            candidate.feature_theta_rad[i] = rng.uniform(0, np.pi / 4)
            candidate.feature_p1_m[i] = rng.uniform(8e-6, 50e-6)
            candidate.feature_p2_m[i] = rng.uniform(8e-6, 50e-6)
            candidate.feature_angle_rad[i] = rng.uniform(-np.pi, np.pi)

    return repair(candidate, cfg)


def initial_population(
    cfg: Config,
    rng: np.random.Generator,
) -> list[Candidate]:
    population = [baseline(cfg)]

    while len(population) < cfg.population:
        population.append(
            seeded_candidate(
                cfg,
                rng,
                broad=len(population) >= int(0.65 * cfg.population),
            )
        )

    return population


def crossover(
    first: Candidate,
    second: Candidate,
    cfg: Config,
    rng: np.random.Generator,
) -> Candidate:
    child = first.copy()

    for name in ("inner_m", "outer_m"):
        first_array = getattr(first, name)
        second_array = getattr(second, name)
        alpha = rng.uniform(-0.2, 1.2, N_CTRL)
        setattr(
            child,
            name,
            alpha * first_array + (1.0 - alpha) * second_array,
        )

    if rng.random() < 0.5:
        child.topology = second.topology

    for name in ("topology_size_m", "topology_width_m"):
        alpha = rng.uniform(-0.2, 1.2)
        setattr(
            child,
            name,
            alpha * getattr(first, name)
            + (1.0 - alpha) * getattr(second, name),
        )

    for i in range(N_FEATURES):
        if rng.random() < 0.5:
            child.feature_kind[i] = second.feature_kind[i]
            child.feature_operation[i] = second.feature_operation[i]

        for name in (
            "feature_radius_m",
            "feature_theta_rad",
            "feature_p1_m",
            "feature_p2_m",
            "feature_angle_rad",
        ):
            alpha = rng.uniform(-0.2, 1.2)
            getattr(child, name)[i] = (
                alpha * getattr(first, name)[i]
                + (1.0 - alpha) * getattr(second, name)[i]
            )

    return repair(child, cfg)


def mutate(
    candidate: Candidate,
    cfg: Config,
    rng: np.random.Generator,
    progress: float,
) -> Candidate:
    result = candidate.copy()
    scale = 1.0 - 0.75 * np.clip(progress, 0, 1)

    for array, sigma in (
        (result.inner_m, 18e-6),
        (result.outer_m, 24e-6),
    ):
        selected = rng.random(N_CTRL) < cfg.mutation_probability
        array[selected] += rng.normal(
            0.0, sigma * scale, int(np.sum(selected))
        )

    if rng.random() < cfg.topology_mutation_probability:
        result.topology = int(rng.integers(len(TOPOLOGIES)))

    for name, sigma in (
        ("topology_size_m", 18e-6),
        ("topology_width_m", 10e-6),
    ):
        if rng.random() < cfg.mutation_probability:
            setattr(
                result,
                name,
                getattr(result, name)
                + float(rng.normal(0.0, sigma * scale)),
            )

    for i in range(N_FEATURES):
        if rng.random() < cfg.topology_mutation_probability:
            result.feature_kind[i] = rng.integers(0, 3)

        if rng.random() < cfg.topology_mutation_probability:
            result.feature_operation[i] *= -1

        for name, sigma in (
            ("feature_radius_m", 25e-6),
            ("feature_theta_rad", 0.12),
            ("feature_p1_m", 12e-6),
            ("feature_p2_m", 12e-6),
            ("feature_angle_rad", 0.25),
        ):
            if rng.random() < cfg.mutation_probability:
                getattr(result, name)[i] += rng.normal(
                    0.0, sigma * scale
                )

    return repair(result, cfg)


def dominates(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first <= second) and np.any(first < second))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    count = len(objectives)
    dominated: list[list[int]] = [[] for _ in range(count)]
    domination_count = np.zeros(count, dtype=int)
    first_front: list[int] = []

    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            first_front.append(p)

    fronts = [first_front]

    while fronts[-1]:
        next_front: list[int] = []
        for p in fronts[-1]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(
    front: list[int],
    objectives: np.ndarray,
) -> dict[int, float]:
    distances = {i: 0.0 for i in front}

    if len(front) <= 2:
        return {i: math.inf for i in front}

    for objective_index in range(objectives.shape[1]):
        ordered = sorted(
            front,
            key=lambda i: objectives[i, objective_index],
        )
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf

        low = objectives[ordered[0], objective_index]
        high = objectives[ordered[-1], objective_index]
        if high <= low:
            continue

        for position in range(1, len(ordered) - 1):
            if np.isinf(distances[ordered[position]]):
                continue
            distances[ordered[position]] += (
                objectives[ordered[position + 1], objective_index]
                - objectives[ordered[position - 1], objective_index]
            ) / (high - low)

    return distances


def ranks_and_crowding(
    objectives: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives)
    ranks = np.empty(len(objectives), dtype=int)
    crowding = np.zeros(len(objectives))

    for rank, front in enumerate(fronts):
        distances = crowding_distance(front, objectives)
        for index in front:
            ranks[index] = rank
            crowding[index] = distances[index]

    return fronts, ranks, crowding


def tournament(
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> int:
    first, second = rng.integers(0, len(ranks), size=2)

    if ranks[first] < ranks[second]:
        return int(first)
    if ranks[second] < ranks[first]:
        return int(second)

    if crowding[first] > crowding[second]:
        return int(first)
    if crowding[second] > crowding[first]:
        return int(second)

    return int(rng.choice((first, second)))


def select_survivors(
    population: list[Candidate],
    evaluations: list[Evaluation],
    count: int,
) -> tuple[list[Candidate], list[Evaluation]]:
    objectives = np.stack([e.objectives for e in evaluations])
    fronts = non_dominated_sort(objectives)
    selected: list[int] = []

    for front in fronts:
        remaining = count - len(selected)
        if remaining <= 0:
            break

        if len(front) <= remaining:
            selected.extend(front)
        else:
            distances = crowding_distance(front, objectives)
            selected.extend(
                sorted(
                    front,
                    key=lambda i: distances[i],
                    reverse=True,
                )[:remaining]
            )
            break

    return (
        [population[i] for i in selected],
        [evaluations[i] for i in selected],
    )


def baseline_boundary_intersects(
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    cfg: Config,
) -> bool:
    """Conservative baseline edge detection for the initial fixed mesh.

    This is NOT exact intersection against every future candidate.
    A final candidate must be checked on a newly generated fine mesh.
    """
    x_samples = np.array(
        [x_left, (x_left + x_right) / 2, x_right]
    )
    y_samples = np.array(
        [y_bottom, (y_bottom + y_top) / 2, y_top]
    )
    xx, yy = np.meshgrid(x_samples, y_samples)

    states = rf_mask(
        baseline(cfg),
        xx.ravel(),
        yy.ravel(),
        cfg,
    )

    return bool(np.any(states != states[0]))


class FixedMeshSolver:
    """Fixed-panel BEM with CPU LU or optional existing CUDA backend."""

    def __init__(
        self,
        panels_m: np.ndarray,
        *,
        backend: str,
    ) -> None:
        self.panels_m = np.asarray(panels_m, dtype=float)
        self.centres_m = np.column_stack(
            (
                (self.panels_m[:, 0] + self.panels_m[:, 1]) / 2,
                (self.panels_m[:, 2] + self.panels_m[:, 3]) / 2,
            )
        )
        self.backend = backend
        self.cpu_bem: BEM2D | None = None
        self.lu = None
        self.cuda_bem = None

        if backend == "cuda":
            # Import only on the CUDA path.
            from core.ga.fixed_bem import FixedMeshBEM

            self.cuda_bem = FixedMeshBEM(
                self.panels_m,
                backend="cuda",
            )
            self.cuda_bem.assemble()
            self.cuda_bem.factorize()
            self.cuda_bem.release_matrix()
        elif backend == "cpu":
            self.cpu_bem = BEM2D(
                self.panels_m,
                np.zeros(len(self.panels_m), dtype=float),
            )
            matrix = self.cpu_bem.assemble(show_progress=True)
            self.lu = lu_factor(matrix, check_finite=False)
            self.cpu_bem.influence_matrix = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def solve_masks(self, masks: np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=float)
        if values.ndim == 1:
            values = values[None, :]

        if values.shape[1] != len(self.panels_m):
            raise ValueError("Mask width does not equal panel count.")

        if self.backend == "cuda":
            result = self.cuda_bem.solve_masks(values)
            return np.asarray(self.cuda_bem.asnumpy(result))

        return lu_solve(
            self.lu,
            values.T,
            check_finite=False,
        ).T

    def field(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if z_m <= 0:
            raise ValueError("Field evaluation requires z_m > 0.")

        if self.backend == "cuda":
            result = self.cuda_bem.field_batch(
                np.array([[x_m]]),
                np.array([[y_m]]),
                np.array([[z_m]]),
                np.asarray(sigma)[None, :],
                candidate_chunk=1,
            )
            return np.asarray(
                self.cuda_bem.asnumpy(result)
            )[0, 0]

        # Reuse the project's analytic field formulas, without re-solving.
        self.cpu_bem.surface_charge_density_c_m2 = np.asarray(
            sigma, dtype=float
        )
        return np.asarray(
            self.cpu_bem.electric_field(x_m, y_m, z_m),
            dtype=float,
        )


def choose_backend(requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "forced CPU; CUDA was not imported"

    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no visible CUDA device")

        from core.ga.fixed_bem import FixedMeshBEM

        if not all(
            hasattr(FixedMeshBEM, name)
            for name in (
                "assemble",
                "factorize",
                "solve_masks",
                "field_batch",
                "asnumpy",
                "release_matrix",
            )
        ):
            raise RuntimeError(
                "core.ga.fixed_bem.FixedMeshBEM has an incompatible API"
            )

        return "cuda", "CuPy and a compatible CUDA BEM are available"
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(
                f"CUDA requested but unavailable: {exc}"
            ) from exc
        return "cpu", f"automatic CPU fallback: {exc}"


def trace_rf_null(
    solver: FixedMeshSolver,
    sigma: np.ndarray,
    x_m: float,
    cfg: Config,
) -> tuple[float, float, float]:
    y = 0.0
    z = TARGET_HEIGHT_M
    epsilon = cfg.derivative_step_m

    for _ in range(cfg.trace_iterations):
        field = solver.field(x_m, y, z, sigma)
        field_y_plus = solver.field(x_m, y + epsilon, z, sigma)
        field_y_minus = solver.field(x_m, y - epsilon, z, sigma)
        field_z_plus = solver.field(x_m, y, z + epsilon, sigma)
        field_z_minus = solver.field(x_m, y, z - epsilon, sigma)

        d_ey_dy = (
            field_y_plus[1] - field_y_minus[1]
        ) / (2 * epsilon)
        d_ez_dy = (
            field_y_plus[2] - field_y_minus[2]
        ) / (2 * epsilon)
        d_ey_dz = (
            field_z_plus[1] - field_z_minus[1]
        ) / (2 * epsilon)
        d_ez_dz = (
            field_z_plus[2] - field_z_minus[2]
        ) / (2 * epsilon)

        jacobian = np.array(
            [[d_ey_dy, d_ey_dz], [d_ez_dy, d_ez_dz]]
        )

        if not np.all(np.isfinite(jacobian)):
            break

        try:
            step = np.linalg.solve(
                jacobian,
                -field[1:3],
            )
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(step)):
            break

        step = np.clip(
            step,
            -cfg.maximum_newton_step_m,
            cfg.maximum_newton_step_m,
        )

        y = float(
            np.clip(
                y + step[0],
                -cfg.maximum_abs_y_m,
                cfg.maximum_abs_y_m,
            )
        )
        z = float(
            np.clip(
                z + step[1],
                cfg.minimum_z_m,
                cfg.maximum_z_m,
            )
        )

    final_field = solver.field(x_m, y, z, sigma)
    residual = float(np.linalg.norm(final_field[1:3]))
    return y, z, residual


def transverse_frequencies(
    solver: FixedMeshSolver,
    sigma: np.ndarray,
    x_m: float,
    y_m: float,
    z_m: float,
    cfg: Config,
) -> np.ndarray:
    epsilon = cfg.derivative_step_m
    jacobian = np.empty((2, 2))

    for column, (dy, dz) in enumerate(
        ((epsilon, 0.0), (0.0, epsilon))
    ):
        plus = solver.field(
            x_m, y_m + dy, z_m + dz, sigma
        )[1:3]
        minus = solver.field(
            x_m, y_m - dy, z_m - dz, sigma
        )[1:3]
        jacobian[:, column] = (plus - minus) / (2 * epsilon)

    # At an RF null, Hessian(|E|^2) in the transverse plane is 2 J^T J.
    hessian = 2.0 * jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(hessian)

    omega_rf = 2 * np.pi * RF_FREQ_NOMINAL_HZ
    prefactor = (
        CHARGE_CA40_C
        * cfg.rf_peak_voltage_v
        / (2 * MASS_CA40_KG * omega_rf)
    )

    return (
        prefactor
        * np.sqrt(np.maximum(eigenvalues, 0.0))
        / (2 * np.pi)
    )


def geometry_penalty(candidate: Candidate, cfg: Config) -> float:
    normalized_inner_curvature = (
        np.diff(candidate.inner_m, n=2) / 20e-6
    )
    normalized_outer_curvature = (
        np.diff(candidate.outer_m, n=2) / 25e-6
    )

    return float(
        0.08
        * np.sqrt(
            np.mean(normalized_inner_curvature**2)
            + np.mean(normalized_outer_curvature**2)
        )
    )


def evaluate_one(
    candidate: Candidate,
    sigma: np.ndarray,
    solver: FixedMeshSolver,
    cfg: Config,
) -> Evaluation:
    invalid_objectives = np.full(
        len(OBJECTIVE_NAMES),
        1e6,
        dtype=float,
    )

    try:
        x_positions = np.linspace(
            0.0,
            cfg.route_extent_m,
            cfg.path_points,
        )

        y_positions = []
        z_positions = []
        residuals = []
        barriers_ev = []
        frequencies = []

        omega_rf = 2 * np.pi * RF_FREQ_NOMINAL_HZ

        for x in x_positions:
            y, z, residual = trace_rf_null(
                solver, sigma, float(x), cfg
            )
            y_positions.append(y)
            z_positions.append(z)
            residuals.append(residual)

            field_at_target = solver.field(
                float(x),
                0.0,
                TARGET_HEIGHT_M,
                sigma,
            )
            pseudo_j = (
                CHARGE_CA40_C**2
                * cfg.rf_peak_voltage_v**2
                * float(np.dot(field_at_target, field_at_target))
                / (4 * MASS_CA40_KG * omega_rf**2)
            )
            barriers_ev.append(pseudo_j / CHARGE_CA40_C)

            frequencies.append(
                transverse_frequencies(
                    solver, sigma, float(x), y, z, cfg
                )
            )

        y_positions = np.asarray(y_positions)
        z_positions = np.asarray(z_positions)
        residuals = np.asarray(residuals)
        barriers_ev = np.asarray(barriers_ev)
        frequencies = np.asarray(frequencies)

        frequency_mean = np.mean(frequencies, axis=1)
        frequency_min = float(np.min(frequencies))
        frequency_max = float(np.max(frequencies))

        height_peak = float(
            np.max(np.abs(z_positions - TARGET_HEIGHT_M))
        )
        lateral_peak = float(np.max(np.abs(y_positions)))
        barrier = float(
            np.max(barriers_ev)
            - np.min(barriers_ev[-min(5, len(barriers_ev)):])
        )
        frequency_error = float(
            np.max(
                np.abs(
                    frequency_mean - cfg.target_frequency_hz
                )
            )
        )
        anisotropy = float(
            np.max(
                frequencies[:, 1]
                / np.maximum(frequencies[:, 0], 1.0)
            )
        )
        max_residual = float(np.max(residuals))
        penalty = geometry_penalty(candidate, cfg)

        values = np.array(
            (
                height_peak,
                lateral_peak,
                barrier,
                frequency_error,
                anisotropy,
                max_residual,
                frequency_min,
                frequency_max,
            )
        )

        valid = bool(
            np.all(np.isfinite(values))
            and max_residual < 5e3
            and frequency_min > 0.15e6
            and np.all(z_positions > cfg.minimum_z_m + 1e-9)
            and np.all(z_positions < cfg.maximum_z_m - 1e-9)
        )

        objectives = np.array(
            (
                height_peak / 3e-6,
                lateral_peak / 3e-6,
                max(barrier, 0.0) / 0.100,
                frequency_error / 0.5e6,
                0.15 * max(anisotropy - 3.0, 0.0),
                penalty,
            ),
            dtype=float,
        )

        if not valid:
            objectives += 1000.0

        return Evaluation(
            objectives=objectives,
            metrics={
                "valid": valid,
                "height_peak_m": height_peak,
                "lateral_peak_m": lateral_peak,
                "barrier_ev": barrier,
                "frequency_error_hz": frequency_error,
                "anisotropy": anisotropy,
                "field_residual_v_m": max_residual,
                "frequency_min_hz": frequency_min,
                "frequency_max_hz": frequency_max,
                "geometry_penalty": penalty,
            },
        )
    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return Evaluation(
            objectives=invalid_objectives,
            metrics={
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def evaluate_population(
    population: list[Candidate],
    solver: FixedMeshSolver,
    cfg: Config,
) -> list[Evaluation]:
    x_centres = solver.centres_m[:, 0]
    y_centres = solver.centres_m[:, 1]

    masks = np.stack(
        [
            rf_mask(candidate, x_centres, y_centres, cfg)
            for candidate in population
        ]
    )

    charges = solver.solve_masks(masks)

    return [
        evaluate_one(candidate, charges[i], solver, cfg)
        for i, candidate in enumerate(population)
    ]


def candidate_record(
    candidate: Candidate,
    evaluation: Evaluation,
    rank: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "rank": rank,
        "topology": TOPOLOGIES[candidate.topology],
        "topology_size_m": candidate.topology_size_m,
        "topology_width_m": candidate.topology_width_m,
    }

    for i, value in enumerate(candidate.inner_m):
        record[f"inner_{i}_m"] = float(value)

    for i, value in enumerate(candidate.outer_m):
        record[f"outer_{i}_m"] = float(value)

    for i in range(N_FEATURES):
        record[f"feature_{i}_kind"] = int(
            candidate.feature_kind[i]
        )
        record[f"feature_{i}_operation"] = int(
            candidate.feature_operation[i]
        )
        record[f"feature_{i}_radius_m"] = float(
            candidate.feature_radius_m[i]
        )
        record[f"feature_{i}_theta_rad"] = float(
            candidate.feature_theta_rad[i]
        )
        record[f"feature_{i}_p1_m"] = float(
            candidate.feature_p1_m[i]
        )
        record[f"feature_{i}_p2_m"] = float(
            candidate.feature_p2_m[i]
        )
        record[f"feature_{i}_angle_rad"] = float(
            candidate.feature_angle_rad[i]
        )

    for name, value in zip(
        OBJECTIVE_NAMES, evaluation.objectives
    ):
        record[f"objective_{name}"] = float(value)

    record.update(evaluation.metrics)
    return record


def write_results(
    output_dir: Path,
    population: list[Candidate],
    evaluations: list[Evaluation],
    generation: int,
) -> None:
    objectives = np.stack(
        [evaluation.objectives for evaluation in evaluations]
    )
    fronts, ranks, _ = ranks_and_crowding(objectives)

    records = [
        candidate_record(candidate, evaluation, int(ranks[i]))
        for i, (candidate, evaluation) in enumerate(
            zip(population, evaluations)
        )
    ]

    csv_path = output_dir / "final_population.csv"
    fieldnames = list(dict.fromkeys(
        key for record in records for key in record
    ))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    pareto_records = [
        records[index] for index in fronts[0]
    ]

    with (output_dir / "pareto_front.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(pareto_records)

    payload = {
        "generation": generation,
        "population": records,
        "pareto_indices": fronts[0],
    }

    (output_dir / "checkpoint.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def build_fixed_mesh(cfg: Config) -> np.ndarray:
    reference = baseline(cfg)

    def classify_point(x: float, y: float) -> float:
        return float(
            rf_mask(
                reference,
                np.asarray([x]),
                np.asarray([y]),
                cfg,
            )[0]
        )

    def intersects_boundary(
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
    ) -> bool:
        return baseline_boundary_intersects(
            x_left,
            x_right,
            y_bottom,
            y_top,
            cfg,
        )

    panels, _ = geometry_aware_quadtree_rectangular_panels(
        outer_extent_m=cfg.outer_extent_m,
        classify_point=classify_point,
        intersects_boundary=intersects_boundary,
        central_half_extent_m=cfg.central_half_extent_m,
        central_max_cell_m=cfg.central_max_cell_m,
        boundary_max_cell_m=cfg.boundary_max_cell_m,
        outer_max_cell_m=cfg.outer_max_cell_m,
        min_cell_m=cfg.min_cell_m,
    )

    if len(panels) > cfg.max_panels:
        raise RuntimeError(
            f"Mesh has {len(panels)} panels, "
            f"exceeding --max-panels={cfg.max_panels}. "
            "Increase mesh cell sizes or raise the panel limit "
            "only if you have enough RAM."
        )

    return panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static RF-geometry NSGA-II without islands"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-panels", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.population < 2:
        raise ValueError("--population must be at least 2")
    if args.generations < 1:
        raise ValueError("--generations must be at least 1")

    cfg = Config(
        seed=args.seed,
        population=args.population,
        offspring=args.population,
        generations=args.generations,
        max_panels=args.max_panels,
    )

    if args.quick:
        cfg.population = 6
        cfg.offspring = 6
        cfg.generations = 2
        cfg.path_points = 7
        cfg.trace_iterations = 5

    backend, reason = choose_backend(args.backend)

    output_dir = (
        ROOT / "reports" / "figures" / "08a_full_geometry_ga"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )
    (output_dir / "backend.json").write_text(
        json.dumps(
            {"requested": args.backend, "active": backend, "reason": reason},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Backend: {backend}; {reason}", flush=True)
    print("Building fixed mesh...", flush=True)

    started = time.perf_counter()
    panels = build_fixed_mesh(cfg)

    print(
        f"Panels: {len(panels)}; "
        f"dense matrix: {8 * len(panels) ** 2 / 2**20:.1f} MiB",
        flush=True,
    )

    solver = FixedMeshSolver(panels, backend=backend)

    print(
        f"BEM prepared in {time.perf_counter() - started:.1f} s",
        flush=True,
    )

    rng = np.random.default_rng(cfg.seed)
    population = initial_population(cfg, rng)
    evaluations = evaluate_population(population, solver, cfg)

    for generation in range(cfg.generations):
        objective_matrix = np.stack(
            [evaluation.objectives for evaluation in evaluations]
        )
        _, ranks, crowding = ranks_and_crowding(
            objective_matrix
        )

        offspring: list[Candidate] = []

        while len(offspring) < cfg.offspring:
            first = population[
                tournament(ranks, crowding, rng)
            ]
            second = population[
                tournament(ranks, crowding, rng)
            ]

            child = crossover(first, second, cfg, rng)
            child = mutate(
                child,
                cfg,
                rng,
                progress=generation / max(cfg.generations - 1, 1),
            )
            offspring.append(child)

        offspring_evaluations = evaluate_population(
            offspring,
            solver,
            cfg,
        )

        population, evaluations = select_survivors(
            population + offspring,
            evaluations + offspring_evaluations,
            cfg.population,
        )

        objective_matrix = np.stack(
            [evaluation.objectives for evaluation in evaluations]
        )
        fronts = non_dominated_sort(objective_matrix)
        valid_count = sum(
            bool(evaluation.metrics.get("valid", False))
            for evaluation in evaluations
        )

        print(
            f"Generation {generation + 1}/{cfg.generations}: "
            f"Pareto={len(fronts[0])}, "
            f"valid={valid_count}/{len(population)}, "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

        if (
            (generation + 1) % cfg.checkpoint_every == 0
            or generation == cfg.generations - 1
        ):
            write_results(
                output_dir,
                population,
                evaluations,
                generation + 1,
            )

    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()