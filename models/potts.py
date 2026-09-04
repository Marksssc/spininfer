from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from stats.mcmc import potts_numba, potts_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict

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

    def _validate_params(self, h, J) -> None:
        if h.shape != (self.n_sites, self.n_states):
            raise ValueError(f"h has shape {h.shape}, expected ({self.n_sites}, {self.n_states})")
        if J.shape != (self.n_sites, self.n_sites, self.n_states, self.n_states):
            raise ValueError(
                f"J has shape {J.shape}, expected "
                f"{(self.n_sites, self.n_sites, self.n_states, self.n_states)}"
            )

    def simulate(self, h, J, samples, iterations=1000, seed=0) -> np.ndarray:
        self._validate_params(h, J)
        kernel = _BACKENDS[self.backend]
        return kernel(h, J, samples, iterations, seed)

    def random_params(self, loc_h=0.0, scale_h =0.3, loc_J=0.0, scale_J=0.3, seed=0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        h = rng.normal(loc= loc_h, scale=scale_h, size=(self.n_sites, self.n_states))

        J = rng.normal(loc=loc_J, scale=scale_J, size=(self.n_sites, self.n_sites, self.n_states, self.n_states))

        if self.gauge:
            h, J = self.apply_gauge(h, J)
        else:
            J = (J + J.transpose(1, 0, 3, 2)) / 2 
            idx = np.arange(self.n_sites)
            J[idx, idx, :, :] = 0.0

        return h, J

    def compute_moments(self, samples: np.ndarray) -> Moments:
        one_hot = np.eye(self.n_states)[samples]
        mean_s = one_hot.mean(axis=0)             
        mean_ss = np.einsum('nia,njb->ijab', one_hot, one_hot) / samples.shape[0]
        return Moments(mean_s=mean_s, mean_ss=mean_ss)

    def apply_gauge(self, h, J) -> tuple[np.ndarray, np.ndarray]:
        h_fixed = h - np.mean(h, axis=1, keepdims=True)
        
        row_mean = np.mean(J, axis=3, keepdims=True)
        col_mean = np.mean(J, axis=2, keepdims=True)
        tot_mean = np.mean(J, axis=(2, 3), keepdims=True)
        J_fixed = J - row_mean - col_mean + tot_mean

        J_fixed = (J_fixed + J_fixed.transpose(1, 0, 3, 2)) / 2.0

        idx = np.arange(self.n_sites)
        J_fixed[idx, idx, :, :] = 0.0

        return h_fixed, J_fixed

    