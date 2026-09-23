from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from stats.moments import Moments
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class Dataset:
    """Sample data for a model: validates site count, moves samples onto the model's array backend,
    and eagerly computes their empirical moments."""

    samples: Array
    model: Model
    moments: Moments = field(init=False)

    def __post_init__(self) -> None:
        """Validate `samples` against `model.n_sites`, cast onto the model's backend, and compute moments."""
        n_sites = self.samples.shape[1]
        if n_sites != self.model.n_sites:
            raise ValueError(f"data has {n_sites} sites, model expects {self.model.n_sites}")

        xp = self.model.array_backend.xp
        self.samples = xp.asarray(self.samples)

        self.moments = self.model.compute_moments(self.samples)
