from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.optimization.constraints import validate_nominal_genome
from core.optimization.genome import JunctionGenome


@dataclass(frozen=True)
class NominalObjectives:
    barrier_mev: float
    height_variation_um: float
    lateral_variation_um: float
    residual_field_v_m: float
    curvature_variation: float
    invalid_fraction: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.barrier_mev,
                self.height_variation_um,
                self.curvature_variation,
                self.residual_field_v_m / 1e3,
                self.invalid_fraction * 100.0,
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "barrier_mev": self.barrier_mev,
            "height_variation_um": self.height_variation_um,
            "lateral_variation_um": self.lateral_variation_um,
            "residual_field_v_m": self.residual_field_v_m,
            "curvature_variation": self.curvature_variation,
            "invalid_fraction": self.invalid_fraction,
        }


@dataclass(frozen=True)
class EvaluationResult:
    genome: JunctionGenome
    objectives: NominalObjectives
    feasible: bool
    penalty: float
    message: str


def invalid_objectives(
    penalty: float,
) -> NominalObjectives:
    scale = 1.0 + penalty
    return NominalObjectives(
        barrier_mev=1e5 * scale,
        height_variation_um=1e4 * scale,
        lateral_variation_um=1e4 * scale,
        residual_field_v_m=1e8 * scale,
        curvature_variation=1e4 * scale,
        invalid_fraction=1.0,
    )


class NominalJunctionEvaluator:
    def __init__(
        self,
        *,
        evaluate_geometry: Callable[[JunctionGenome], dict[str, float]],
    ) -> None:
        self._evaluate_geometry = evaluate_geometry

    def __call__(self, genome: JunctionGenome) -> EvaluationResult:
        constraints = validate_nominal_genome(genome)

        if not constraints.feasible:
            return EvaluationResult(
                genome=genome,
                objectives=invalid_objectives(constraints.penalty),
                feasible=False,
                penalty=constraints.penalty,
                message="; ".join(constraints.messages),
            )

        try:
            metrics = self._evaluate_geometry(genome)
        except Exception as error:
            return EvaluationResult(
                genome=genome,
                objectives=invalid_objectives(10.0),
                feasible=False,
                penalty=10.0,
                message=f"BEM evaluation failed: {error}",
            )

        objectives = NominalObjectives(
            barrier_mev=float(metrics["barrier_mev"]),
            height_variation_um=float(metrics["height_variation_um"]),
            lateral_variation_um=float(metrics["lateral_variation_um"]),
            residual_field_v_m=float(metrics["residual_field_v_m"]),
            curvature_variation=float(metrics["curvature_variation"]),
            invalid_fraction=float(metrics["invalid_fraction"]),
        )

        physical_failure = (
            objectives.invalid_fraction > 0.0
            or not np.all(np.isfinite(objectives.as_array()))
        )

        return EvaluationResult(
            genome=genome,
            objectives=(
                invalid_objectives(5.0)
                if physical_failure
                else objectives
            ),
            feasible=not physical_failure,
            penalty=5.0 if physical_failure else 0.0,
            message=(
                "RF-null trace contains invalid points."
                if physical_failure
                else "ok"
            ),
        )