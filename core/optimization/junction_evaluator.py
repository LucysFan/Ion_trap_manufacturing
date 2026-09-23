from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from core.optimization.genome import (
    FEATURE_CIRCLE,
    FEATURE_NONE,
    FEATURE_RECTANGLE,
    Genome,
    TOPOLOGY_CENTRAL_RF_CROSS,
    TOPOLOGY_CENTRAL_RF_DISK,
    TOPOLOGY_CENTRAL_RF_RING,
    TOPOLOGY_GROUNDED_MOAT,
)

try:
    from config.targets import RF_ANGULAR_FREQUENCY_RAD_S as OMEGA_RF
    from config.targets import TARGET_ION_HEIGHT_M as TARGET_HEIGHT_M
except ImportError:
    from config.physical_constants import RF_FREQ_NOMINAL_HZ, TARGET_HEIGHT_M
    OMEGA_RF = 2.0 * np.pi * RF_FREQ_NOMINAL_HZ

try:
    from config.physical_constants import MASS_CA40_KG, CHARGE_CA40_C
except ImportError:
    MASS_CA40_KG = 39.96259098 * 1.66053906660e-27
    CHARGE_CA40_C = 1.602176634e-19


@dataclass(frozen=True)
class EvaluatorConfig:
    route_extent_m: float = 350e-6
    path_points: int = 41
    target_frequency_hz: float = 1.5e6
    v_rf_peak_v: float = 100.0

    trace_iterations: int = 12
    trace_step_m: float = 0.35e-6
    max_newton_step_m: float = 6e-6

    z_min_m: float = 25e-6
    z_max_m: float = 220e-6
    y_limit_m: float = 60e-6

    min_rf_width_m: float = 18e-6
    min_feature_m: float = 6e-6
    gpu_candidate_chunk: int = 12


@dataclass
class JunctionEvaluation:
    genome: Genome
    objectives: np.ndarray
    metrics: dict[str, float]
    valid: bool
    failure_reason: str | None = None


@dataclass
class JunctionMetrics:
    valid: bool
    failure_reason: str | None

    height_peak_m: float
    height_rms_m: float
    lateral_peak_m: float
    lateral_rms_m: float

    barrier_ev: float
    pseudopotential_min_ev: float

    secular_freq_min_hz: float
    secular_freq_max_hz: float
    secular_freq_nominal_hz: float
    secular_freq_spread_hz: float

    curvature_anisotropy_max: float
    field_residual_max_v_m: float

    manufacturability_penalty: float
    geometry_penalty: float

def _spline_boundary(values_m: np.ndarray, s: np.ndarray) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, len(values_m))
    spline = CubicSpline(knots, values_m, bc_type="natural")
    return spline(np.clip(s, 0.0, 1.0))


def _rectangle_mask(
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
    return (np.abs(xr) <= width / 2.0) & (np.abs(yr) <= height / 2.0)


def _apply_c4v_feature(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    genome: Genome,
    feature_index: int,
) -> None:
    kind = int(genome.feature_kind[feature_index])
    if kind == FEATURE_NONE:
        return

    operation = int(genome.feature_operation[feature_index])
    radius = float(genome.feature_radius_m[feature_index])
    theta0 = float(genome.feature_theta_rad[feature_index])
    p1 = float(genome.feature_p1_m[feature_index])
    p2 = float(genome.feature_p2_m[feature_index])
    angle0 = float(genome.feature_angle_rad[feature_index])

    feature_mask = np.zeros_like(mask, dtype=bool)

    for reflection in (-1.0, 1.0):
        for quarter_turn in range(4):
            theta = reflection * theta0 + quarter_turn * np.pi / 2.0
            cx = radius * np.cos(theta)
            cy = radius * np.sin(theta)

            if kind == FEATURE_CIRCLE:
                primitive = (x - cx) ** 2 + (y - cy) ** 2 <= p1 ** 2
            elif kind == FEATURE_RECTANGLE:
                primitive = _rectangle_mask(
                    x,
                    y,
                    cx,
                    cy,
                    p1,
                    p2,
                    reflection * angle0 + quarter_turn * np.pi / 2.0,
                )
            else:
                continue

            feature_mask |= primitive

    if operation > 0:
        mask[feature_mask] = True
    else:
        mask[feature_mask] = False


def build_mask(
    genome: Genome,
    x: np.ndarray,
    y: np.ndarray,
    *,
    l_arm_m: float,
) -> np.ndarray:
    ax = np.abs(x)
    ay = np.abs(y)

    along_x = ax >= ay
    longitudinal = np.where(along_x, ax, ay)
    transverse = np.where(along_x, ay, ax)
    normalized_s = longitudinal / l_arm_m

    inner = _spline_boundary(genome.inner_control_m, normalized_s)
    outer = _spline_boundary(genome.outer_control_m, normalized_s)

    mask = (longitudinal <= l_arm_m) & (transverse >= inner) & (transverse <= outer)

    radial = np.hypot(x, y)
    size = float(genome.topology_size_m)
    width = float(genome.topology_width_m)
    topology = int(genome.topology)

    if topology == TOPOLOGY_CENTRAL_RF_DISK:
        mask[radial <= size] = True
    elif topology == TOPOLOGY_CENTRAL_RF_RING:
        ring = np.abs(radial - size) <= width / 2.0
        mask[ring] = True
    elif topology == TOPOLOGY_CENTRAL_RF_CROSS:
        cross = (radial <= size) & ((np.abs(x) <= width / 2.0) | (np.abs(y) <= width / 2.0))
        mask[cross] = True
    elif topology == TOPOLOGY_GROUNDED_MOAT:
        moat = np.abs(radial - size) <= width / 2.0
        mask[moat] = False

    for feature_index in range(genome.feature_kind.shape[0]):
        _apply_c4v_feature(mask, x, y, genome, feature_index)

    return mask.astype(np.float64, copy=False)


def build_masks(
    genomes: list[Genome],
    x: np.ndarray,
    y: np.ndarray,
    *,
    l_arm_m: float,
) -> np.ndarray:
    return np.stack([build_mask(genome, x, y, l_arm_m=l_arm_m) for genome in genomes], axis=0)


def _trace_rf_null(
    bem: Any,
    sigma: Any,
    cfg: EvaluatorConfig,
) -> tuple[Any, Any, Any, Any, Any]:
    cp = bem.cp
    batch = int(sigma.shape[0])

    x_line = cp.linspace(0.0, cfg.route_extent_m, cfg.path_points)
    x = cp.broadcast_to(x_line[None, :], (batch, cfg.path_points)).copy()
    y = cp.zeros_like(x)
    z = cp.full_like(x, TARGET_HEIGHT_M)

    eps = cfg.trace_step_m

    for _ in range(cfg.trace_iterations):
        field = bem.field_batch(x, y, z, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        field_yp = bem.field_batch(x, y + eps, z, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        field_ym = bem.field_batch(x, y - eps, z, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        field_zp = bem.field_batch(x, y, z + eps, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        field_zm = bem.field_batch(x, y, z - eps, sigma, candidate_chunk=cfg.gpu_candidate_chunk)

        dEy_dy = (field_yp[:, :, 1] - field_ym[:, :, 1]) / (2.0 * eps)
        dEz_dy = (field_yp[:, :, 2] - field_ym[:, :, 2]) / (2.0 * eps)
        dEy_dz = (field_zp[:, :, 1] - field_zm[:, :, 1]) / (2.0 * eps)
        dEz_dz = (field_zp[:, :, 2] - field_zm[:, :, 2]) / (2.0 * eps)

        determinant = dEy_dy * dEz_dz - dEy_dz * dEz_dy
        safe = cp.abs(determinant) > 1e-24

        ey = field[:, :, 1]
        ez = field[:, :, 2]

        dy = cp.where(safe, (-ey * dEz_dz + ez * dEy_dz) / determinant, 0.0)
        dz = cp.where(safe, (-ez * dEy_dy + ey * dEz_dy) / determinant, 0.0)

        dy = cp.clip(dy, -cfg.max_newton_step_m, cfg.max_newton_step_m)
        dz = cp.clip(dz, -cfg.max_newton_step_m, cfg.max_newton_step_m)

        y = cp.clip(y + dy, -cfg.y_limit_m, cfg.y_limit_m)
        z = cp.clip(z + dz, cfg.z_min_m, cfg.z_max_m)

    field = bem.field_batch(x, y, z, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
    residual = cp.sqrt(field[:, :, 1] ** 2 + field[:, :, 2] ** 2)
    return x, y, z, field, residual


def _curvature_frequencies(
    bem: Any,
    sigma: Any,
    x: Any,
    y: Any,
    z: Any,
    cfg: EvaluatorConfig,
) -> Any:
    cp = bem.cp
    eps = cfg.trace_step_m

    derivatives = []
    for dx, dy, dz in ((eps, 0.0, 0.0), (0.0, eps, 0.0), (0.0, 0.0, eps)):
        plus = bem.field_batch(x + dx, y + dy, z + dz, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        minus = bem.field_batch(x - dx, y - dy, z - dz, sigma, candidate_chunk=cfg.gpu_candidate_chunk)
        derivatives.append((plus - minus) / (2.0 * eps))

    jacobian = cp.stack(derivatives, axis=3)
    hessian_e2 = 2.0 * cp.matmul(cp.swapaxes(jacobian, 2, 3), jacobian)
    eigenvalues = cp.linalg.eigvalsh(hessian_e2)

    prefactor = CHARGE_CA40_C * cfg.v_rf_peak_v ** 2 / (4.0 * MASS_CA40_KG * OMEGA_RF ** 2)
    return prefactor * cp.sqrt(cp.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)


def geometry_penalty(
    genome: Genome,
    *,
    min_rf_width_m: float,
) -> float:
    rf_width = genome.outer_control_m - genome.inner_control_m
    width_violation = np.maximum(min_rf_width_m - rf_width, 0.0)

    inner_curvature = np.diff(genome.inner_control_m, n=2)
    outer_curvature = np.diff(genome.outer_control_m, n=2)

    curvature_penalty = np.sqrt(
        np.mean((inner_curvature / 20e-6) ** 2) +
        np.mean((outer_curvature / 25e-6) ** 2)
    )

    return float(np.sum((width_violation / min_rf_width_m) ** 2) + 0.08 * curvature_penalty)


class JunctionEvaluator:
    def __init__(
        self,
        *,
        bem: Any,
        x_centers_m: np.ndarray,
        y_centers_m: np.ndarray,
        l_arm_m: float,
        config: EvaluatorConfig | None = None,
    ) -> None:
        self.bem = bem
        self.x_centers_m = np.asarray(x_centers_m, dtype=float)
        self.y_centers_m = np.asarray(y_centers_m, dtype=float)
        self.l_arm_m = float(l_arm_m)
        self.config = config or EvaluatorConfig()

    def evaluate_population(self, genomes: list[Genome]) -> list[JunctionEvaluation]:
        cfg = self.config
        repaired = [genome.clip() for genome in genomes]

        masks = build_masks(
            repaired,
            self.x_centers_m,
            self.y_centers_m,
            l_arm_m=self.l_arm_m,
        )

        sigma = self.bem.solve_masks(masks)
        x, y, z, _, residual = _trace_rf_null(self.bem, sigma, cfg)

        cp = self.bem.cp
        target_y = cp.zeros_like(x)
        target_z = cp.full_like(x, TARGET_HEIGHT_M)

        target_field = self.bem.field_batch(
            x,
            target_y,
            target_z,
            sigma,
            candidate_chunk=cfg.gpu_candidate_chunk,
        )
        target_e2 = cp.sum(target_field * target_field, axis=2)
        pseudo_ev = CHARGE_CA40_C * cfg.v_rf_peak_v ** 2 * target_e2 / (4.0 * MASS_CA40_KG * OMEGA_RF ** 2)

        barrier_ev = cp.max(pseudo_ev, axis=1) - cp.min(pseudo_ev[:, -5:], axis=1)
        frequencies_hz = _curvature_frequencies(self.bem, sigma, x, y, z, cfg)

        transverse = frequencies_hz[:, :, :2]
        transverse_mean = cp.mean(transverse, axis=2)
        frequency_error = cp.max(cp.abs(transverse_mean - cfg.target_frequency_hz), axis=1)
        anisotropy = cp.max(transverse[:, :, 1] / cp.maximum(transverse[:, :, 0], 1.0), axis=1)

        height_peak = cp.max(cp.abs(z - TARGET_HEIGHT_M), axis=1)
        height_rms = cp.sqrt(cp.mean((z - TARGET_HEIGHT_M) ** 2, axis=1))
        lateral_peak = cp.max(cp.abs(y), axis=1)
        max_residual = cp.max(residual, axis=1)

        arrays = self.bem.asnumpy(
            cp.stack(
                [
                    height_peak,
                    height_rms,
                    lateral_peak,
                    barrier_ev,
                    frequency_error,
                    anisotropy,
                    max_residual,
                    cp.min(transverse[:, :, 0], axis=1),
                    cp.max(transverse[:, :, 1], axis=1),
                ],
                axis=1,
            )
        )

        evaluations: list[JunctionEvaluation] = []
        for genome, values in zip(repaired, arrays):
            peak, rms, lateral, barrier, freq_error, anisotropy_value, residual_value, f_min, f_max = values

            penalty = geometry_penalty(genome, min_rf_width_m=cfg.min_rf_width_m)

            invalid = (
                (not genome.is_physically_valid(min_rf_width_m=cfg.min_rf_width_m))
                or (not np.all(np.isfinite(values)))
                or (peak >= cfg.z_max_m - TARGET_HEIGHT_M - 1e-9)
                or (residual_value > 5e3)
                or (f_min < 0.15e6)
            )
            invalid_penalty = 1000.0 if invalid else 0.0

            objectives = np.asarray(
                [
                    peak / 3e-6 + invalid_penalty,
                    lateral / 3e-6 + invalid_penalty,
                    max(barrier, 0.0) / 0.100 + invalid_penalty,
                    freq_error / 0.5e6,
                    max(anisotropy_value - 3.0, 0.0),
                    penalty + invalid_penalty,
                ],
                dtype=np.float64,
            )

            metrics = {
                "height_peak_m": float(peak),
                "height_rms_m": float(rms),
                "lateral_peak_m": float(lateral),
                "barrier_ev": float(barrier),
                "frequency_error_hz": float(freq_error),
                "anisotropy_max": float(anisotropy_value),
                "field_residual_max_v_m": float(residual_value),
                "frequency_min_hz": float(f_min),
                "frequency_max_hz": float(f_max),
                "geometry_penalty": float(penalty),
                "invalid": float(bool(invalid)),
            }

            evaluations.append(
                JunctionEvaluation(
                    genome=genome,
                    objectives=objectives,
                    metrics=metrics,
                    valid=not invalid,
                    failure_reason=None if not invalid else "invalid_geometry_or_trace",
                )
            )

        return evaluations

    def evaluate(self, genome: Genome) -> JunctionEvaluation:
        return self.evaluate_population([genome])[0]