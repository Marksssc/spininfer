from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from stats.thermodynamics import Thermodynamics


@dataclass
class ThermodynamicsIntegrationEstimator:
    n_samples: int
    iterations: int = 1000
    seed: int = 0
    n_points: int = 16

    def estimate(self, model, h, J) -> Thermodynamics:
        nodes, raw_weights = np.polynomial.legendre.leggauss(self.n_points)
        lambdas = (0.5 * (nodes + 1.0)).tolist()
        weights = (0.5 * raw_weights).tolist()

        integral = 0.0
        for lam, w in zip(lambdas, weights):
            samples = model.simulate(h, lam * J, self.n_samples, self.iterations, seed=self.seed)
            moments = model.compute_moments(samples)
            integral += w * model.interaction_energy(J, moments.mean_ss)

        free_energy = model.reference_free_energy(h) - integral

        samples_1 = model.simulate(h, J, self.n_samples, self.iterations, seed=self.seed)
        energies = model.compute_energy(h, J, samples_1)
        energy = float(energies.mean())
        heat_capacity = float(energies.var())
        entropy = energy - free_energy

        return Thermodynamics(entropy=entropy, energy=energy, free_energy=free_energy, heat_capacity=heat_capacity)
