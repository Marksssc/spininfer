from __future__ import annotations
from typing import Any

from dataclasses import dataclass
from stats.mcmc import potts_numba as potts_mcmc_numba, potts_numpy as potts_mcmc_numpy
from stats.exact import potts_numba as potts_exact_numba, potts_numpy as potts_exact_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict
from backend.array_backend import get_array_backend

_BACKENDS = build_backend_dict(
    required={
        "numba": potts_mcmc_numba.simulate,
        "numpy": potts_mcmc_numpy.simulate,
    },
    optional={"cupy": ("stats.mcmc.potts_cupy", "simulate"),
              "jax": ("stats.mcmc.potts_jax", "simulate")},
)

_EXACT_BACKENDS = build_backend_dict(
    required={
        "numba": potts_exact_numba.get_exact_statistics,
        "numpy": potts_exact_numpy.get_exact_statistics,
    },
    optional={"cupy": ("stats.exact.potts_cupy", "get_exact_statistics"),
              "jax": ("stats.exact.potts_jax", "get_exact_statistics")},
)

_EXACT_THERMO_BACKENDS = build_backend_dict(
    required={
        "numba": potts_exact_numba.get_exact_thermodynamics,
        "numpy": potts_exact_numpy.get_exact_thermodynamics,
    },
    optional={"cupy": ("stats.exact.potts_cupy", "get_exact_thermodynamics"),
              "jax": ("stats.exact.potts_jax", "get_exact_thermodynamics")},
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

    def compute_energy(self, h, J, samples) -> Any:
        xp = self.array_backend.xp
        one_hot = xp.eye(self.n_states)[samples]
        e_h = xp.einsum('nia,ia->n', one_hot, h)
        e_J = 0.5 * xp.einsum('nia,njb,ijab->n', one_hot, one_hot, J)
        return -(e_h + e_J)

    def interaction_energy(self, J, mean_ss) -> Any:
        return 0.5 * self.array_backend.xp.einsum('ijab,ijab->', J, mean_ss)

    def reference_free_energy(self, h) -> Any:
        xp = self.array_backend.xp
        m = xp.max(h, axis=1, keepdims=True)
        return -xp.sum(xp.log(xp.sum(xp.exp(h - m), axis=1)) + m.squeeze(axis=1))

    def apply_gauge(self, h, J) -> tuple[Any, Any]:  
        '''
        Potts model is overparametrized, apply gauge fix to be in the "Ising gauge",
        following the method outlined by ekeberg et al.
        '''      
        xp = self.array_backend.xp
        mask = 1.0 - xp.eye(self.n_sites)[:, :, None, None]

        J_sym = (J + J.transpose(1, 0, 3, 2)) / 2.0

        row_mean = xp.mean(J_sym, axis=3, keepdims=True)
        col_mean = xp.mean(J_sym, axis=2, keepdims=True)
        tot_mean = xp.mean(J_sym, axis=(2, 3), keepdims=True)

        J_fixed = (J_sym - row_mean - col_mean + tot_mean) * mask

        idx = xp.arange(self.n_sites)
        diag_self = xp.diagonal(J_sym[idx, idx], axis1=1, axis2=2)   # J_ii(a,a) per site
        compensation = xp.sum((row_mean - tot_mean) * mask, axis=1).squeeze(-1)
        h_fixed = h + compensation + 0.5 * diag_self
        h_fixed = h_fixed - xp.mean(h_fixed, axis=1, keepdims=True)

        return h_fixed, J_fixed

    def exact_statistics(self, h, J):
        self._validate_params(h, J)
        return _EXACT_BACKENDS[self.backend](h, J)

    def exact_thermodynamics(self, h, J):
        self._validate_params(h, J)
        return _EXACT_THERMO_BACKENDS[self.backend](h, J)
