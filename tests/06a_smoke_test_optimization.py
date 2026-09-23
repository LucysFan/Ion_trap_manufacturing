from core.optimization.constraints import validate_nominal_genome
from core.optimization.genome import GenomeBounds, JunctionGenome
from core.optimization.pareto import crowding_distance, nondominated_indices

import numpy as np


def main() -> None:
    bounds = GenomeBounds.default()
    genome = JunctionGenome.from_array(
        np.array([30.0, -10.0, 15.0, 120.0, 1.5])
    )

    print(genome.as_dict())
    print(validate_nominal_genome(genome))

    objectives = np.array(
        [
            [100.0, 20.0, 1.0],
            [80.0, 30.0, 1.1],
            [120.0, 18.0, 0.9],
            [140.0, 35.0, 1.4],
        ]
    )

    indices = nondominated_indices(objectives)
    print("Pareto indices:", indices.tolist())
    print("Crowding:", crowding_distance(objectives[indices]).tolist())
    print("Bounds span:", bounds.span.tolist())


if __name__ == "__main__":
    main()