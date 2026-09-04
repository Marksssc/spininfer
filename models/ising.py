from __future__ import annotations
from typing import Any

from dataclasses import dataclass
from stats.mcmc import ising_numba as ising_mcmc_numba, ising_numpy as ising_mcmc_numpy
from stats.exact import ising_numba as ising_exact_numba, ising_numpy as ising_exact_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict
from backend.array_backend import get_array_backend

_BACKENDS = build_backend_dict(
    required={
        "numba": ising_mcmc_numba.simulate,
        "numpy": ising_mcmc_numpy.simulate,
    },
    optional={"cupy": ("stats.mcmc.ising_cupy", "simulate"),
              "jax": ("stats.mcmc.ising_jax", "simulate")},
)

_EXACT_BACKENDS = build_backend_dict(
    required={
        "numba": ising_exact_numba.get_exact_statistics,
        "numpy": ising_exact_numpy.get_exact_statistics,
    },
    optional={"cupy": ("stats.exact.ising_cupy", "get_exact_statistics"),
              "jax": ("stats.exact.ising_jax", "get_exact_statistics")},
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
        self.array_backend = get_array_backend(self.backend)

    def _validate_params(self, h, J) -> None:
        if h.shape != (self.n_sites,):
            raise ValueError(f"h has shape {h.shape}, expected ({self.n_sites},)")
        if J.shape != (self.n_sites, self.n_sites):
            raise ValueError(f"J has shape {J.shape}, expected ({self.n_sites}, {self.n_sites})")
        if not self.array_backend.xp.allclose(J, J.T):
            raise ValueError(f"The coupling matrix is not symmetric")

    def simulate(self, h, J, samples, iterations=1000, seed=0) -> Any:
        self._validate_params(h, J)
        
        h_array = self.array_backend.xp.asarray(h)
        J_array = self.array_backend.xp.asarray(J)
        
        kwargs = self.array_backend.get_kernel_kwargs(seed)
        
        kernel = _BACKENDS[self.backend]
        return kernel(h_array, J_array, samples, iterations, **kwargs)

    def random_params(self, loc_h=0.0, scale_h =0.3, loc_J=0.0, scale_J=0.3, seed=0)-> tuple[Any, Any]:
        J = self.array_backend.random_normal((self.n_sites, self.n_sites), loc_J, scale_J, seed)
        h = self.array_backend.random_normal((self.n_sites,), loc_h, scale_h, seed + 1)
        
        if self.gauge:
            h, J = self.apply_gauge(h, J)
        return h, J

    def compute_moments(self, samples: Any) -> Moments:
        mean_s = samples.mean(axis=0)                     
        mean_ss = (samples.T @ samples) / samples.shape[0] 
        return Moments(mean_s=mean_s, mean_ss=mean_ss)

    def apply_gauge(self, h, J):
        J = (J + J.T) / 2
        J = self.array_backend.zero_diagonal(J)
        return h, J

    def exact_statistics(self, h, J):
        self._validate_params(h, J)
        return _EXACT_BACKENDS[self.backend](h, J)