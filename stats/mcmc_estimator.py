from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from stats.moments import Moments
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class McmcEstimator:
    """Estimates moments by drawing MCMC samples from `model` and averaging over them."""

    n_samples: int
    iterations: int = 1000
    seed: int = 0

    def estimate(self, model: Model, h: Array, J: Array) -> Moments:
        """Draw `n_samples` MCMC samples at (h, J) and return their empirical moments."""
        samples = model.simulate(h, J, self.n_samples, self.iterations, seed=self.seed)
        return model.compute_moments(samples)
