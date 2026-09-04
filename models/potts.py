from __future__ import annotations
from typing import Any

import numpy as np
from dataclasses import dataclass
from stats.mcmc import potts_numba, potts_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict
from backend.array_backend import get_array_backend

_BACKENDS = build_backend_dict(
    required={
        "numba": potts_numba.simulate,
        "numpy": potts_numpy.simulate,
    },
    optional={"cupy": ("stats.mcmc.potts_cupy", "simulate"),
              "jax": ("stats.mcmc.potts_jax", "simulate")},
)

@dataclass
class PottsModel:
    n_sites: int
    n_states: int
    backend: str="numba"
    gauge: bool=True

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Available: {list(_BACKENDS)}"
            )
        self.array_backend = get_array_backend(self.backend)

    def _validate_params(self, h, J) -> None:
        if h.shape != (self.n_sites, self.n_states):
            raise ValueError(f"h has shape {h.shape}, expected ({self.n_sites}, {self.n_states})")
        if J.shape != (self.n_sites, self.n_sites, self.n_states, self.n_states):
            raise ValueError(
                f"J has shape {J.shape}, expected "
                f"{(self.n_sites, self.n_sites, self.n_states, self.n_states)}"
            )
        if not self.array_backend.xp.allclose(J, J.transpose(1, 0, 3, 2)):
            raise ValueError(f"The coupling matrix is not symmetric")

    def simulate(self, h, J, samples, iterations=1000, seed=0) -> Any:
        self._validate_params(h, J)

        h_array = self.array_backend.xp.asarray(h)
        J_array = self.array_backend.xp.asarray(J)
        
        kwargs = self.array_backend.get_kernel_kwargs(seed)
        
        kernel = _BACKENDS[self.backend]
        return kernel(h_array, J_array, samples, iterations, **kwargs)
        

    def random_params(self, loc_h=0.0, scale_h =0.3, loc_J=0.0, scale_J=0.3, seed=0) -> tuple[Any, Any]:
        J = self.array_backend.random_normal((self.n_sites, self.n_sites, self.n_states, self.n_states), loc_J, scale_J, seed)
        h = self.array_backend.random_normal((self.n_sites, self.n_states), loc_h, scale_h, seed + 1)

        if self.gauge:
            h, J = self.apply_gauge(h, J)
        else:
            J = (J + J.transpose(1, 0, 3, 2)) / 2 
            mask = 1.0 - self.array_backend.xp.eye(self.n_sites)[:, :, None, None]
            J = J * mask

        return h, J

    def compute_moments(self, samples: Any) -> Moments:
        one_hot = self.array_backend.xp.eye(self.n_states)[samples]
        mean_s = one_hot.mean(axis=0)             
        mean_ss = self.array_backend.xp.einsum('nia,njb->ijab', one_hot, one_hot) / samples.shape[0]
        return Moments(mean_s=mean_s, mean_ss=mean_ss)

    def apply_gauge(self, h, J) -> tuple[Any, Any]:
        h_fixed = h - self.array_backend.xp.mean(h, axis=1, keepdims=True)
        
        row_mean = self.array_backend.xp.mean(J, axis=3, keepdims=True)
        col_mean = self.array_backend.xp.mean(J, axis=2, keepdims=True)
        tot_mean = self.array_backend.xp.mean(J, axis=(2, 3), keepdims=True)
        J_fixed = J - row_mean - col_mean + tot_mean

        J_fixed = (J_fixed + J_fixed.transpose(1, 0, 3, 2)) / 2.0

        mask = 1.0 - self.array_backend.xp.eye(self.n_sites)[:, :, None, None]
        J_fixed = J_fixed * mask

        return h_fixed, J_fixed

    