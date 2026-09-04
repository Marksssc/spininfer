from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from stats.mcmc import ising_numba, ising_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict

_BACKENDS = build_backend_dict(
    required={
        "numba": ising_numba.simulate,
        "numpy": ising_numpy.simulate,
    },
    optional={"cupy": ("stats.mcmc.ising_cupy", "simulate"),
              "jax": ("stats.mcmc.ising_jax", "simulate")},
)

@dataclass
class IsingModel:
    n_sites: int
    backend: str = "numba"
    gauge: bool=True

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Available: {list(_BACKENDS)}"
            )

    def _validate_params(self, h, J) -> None:
        if h.shape != (self.n_sites,):
            raise ValueError(f"h has shape {h.shape}, expected ({self.n_sites},)")
        if J.shape != (self.n_sites, self.n_sites):
            raise ValueError(f"J has shape {J.shape}, expected ({self.n_sites}, {self.n_sites})")

    def simulate(self, h, J, samples, iterations=1000, seed=0) -> np.ndarray:
        self._validate_params(h, J)
        kernel = _BACKENDS[self.backend]
        if self.backend == "jax":
            import jax
            return kernel(h, J, samples, iterations, key=jax.random.PRNGKey(seed))
        elif self.backend == "cupy":
            import cupy as cp
            return kernel(cp.asarray(h), cp.asarray(J), samples, iterations, seed=seed)
        else:
            return kernel(h, J, samples, iterations, seed=seed)

    def random_params(self, loc_h=0.0, scale_h =0.3, loc_J=0.0, scale_J=0.3, seed=0)-> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        J = rng.normal(loc=loc_J, scale=scale_J, size=(self.n_sites, self.n_sites))
        h = rng.normal(loc=loc_h ,scale=scale_h, size=self.n_sites)
        if self.gauge:
            h, J = self.apply_gauge(h, J)
        return h, J

    def compute_moments(self, samples: np.ndarray) -> Moments:
        mean_s = samples.mean(axis=0)                     
        mean_ss = (samples.T @ samples) / samples.shape[0] 
        return Moments(mean_s=mean_s, mean_ss=mean_ss)

    def apply_gauge(self, h, J):
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0.0)
        return h, J