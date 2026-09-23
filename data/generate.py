from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class SyntheticDataset:
    """MCMC-generated samples from a model, with the ground-truth (h, J) kept alongside."""

    h: Array
    J: Array
    samples: Array
    seed: int

def generate_data(model: Model, n_samples: int, iterations: int = 1000, seed: int = 0,
                   h: Array | None = None, J: Array | None = None,
                   loc_h: float = 0.0, scale_h: float = 0.2, loc_J: float = 0.0, scale_J: float = 0.3) -> SyntheticDataset:
    """Simulate `n_samples` from `model`, using `h`/`J` if given or drawing random ones otherwise, and return both alongside the samples."""
    if h is None or J is None:
        h, J = model.random_params(loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J, seed=seed)
    samples = model.simulate(h, J, n_samples, iterations, seed=seed)
    return SyntheticDataset(h=h, J=J, samples=samples, seed=seed)
