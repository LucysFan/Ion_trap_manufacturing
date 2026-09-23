from __future__ import annotations

import numpy as np

from core.optimization.constraints import repair_genome, validate_genome
from core.optimization.genome import Genome


def test_default_genome_is_valid() -> None:
    genome = Genome()
    report = validate_genome(genome)

    assert report.valid
    assert len(report.violations) == 0


def test_repair_enforces_minimum_rf_width() -> None:
    genome = Genome()
    genome.outer_control_m[:] = genome.inner_control_m[:] - 5e-6

    repaired = repair_genome(genome)
    report = validate_genome(repaired)

    assert report.valid
    assert np.all(repaired.outer_control_m > repaired.inner_control_m)


def test_repair_clips_topology_parameters() -> None:
    genome = Genome()
    genome.topology = 999
    genome.topology_size_m = -10e-6
    genome.topology_width_m = -5e-6

    repaired = repair_genome(genome)

    assert repaired.topology >= 0
    assert repaired.topology_size_m >= 0.0
    assert repaired.topology_width_m >= 0.0