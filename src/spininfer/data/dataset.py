from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from spininfer.stats.moments import Moments
from spininfer.models import Model
from spininfer.models.ising import IsingModel

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class Dataset:
    """Data for a model, given as either samples or moments."""

    model: Model
    samples: Array | None = None
    moments: Moments | None = None
    n_samples: int | None = None  

    def __post_init__(self) -> None:
        """Validate the given samples or moments against `model.n_sites` and cast onto the model's backend."""
        if (self.samples is None) == (self.moments is None):
            raise ValueError("give exactly one of `samples` or `moments`")

        xp = self.model.array_backend.xp
        if self.samples is not None:
            n_sites = self.samples.shape[1]
            if n_sites != self.model.n_sites:
                raise ValueError(f"data has {n_sites} sites, model expects {self.model.n_sites}")
            dtype = float if isinstance(self.model, IsingModel) else int
            self.samples = xp.asarray(self.samples, dtype=dtype)
            self.moments = self.model.compute_moments(self.samples)
            if self.n_samples is None:
                self.n_samples = self.samples.shape[0]
        else:
            n_sites = self.moments.mean_s.shape[0]
            if n_sites != self.model.n_sites:
                raise ValueError(f"moments have {n_sites} sites, model expects {self.model.n_sites}")
            self.moments = Moments(mean_s=xp.asarray(self.moments.mean_s),
                                   mean_ss=xp.asarray(self.moments.mean_ss))

