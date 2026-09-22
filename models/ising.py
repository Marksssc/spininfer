from __future__ import annotations
from typing import Any

from dataclasses import dataclass
from stats.mcmc import ising_numba as ising_mcmc_numba, ising_numpy as ising_mcmc_numpy
from stats.exact import ising_numba as ising_exact_numba, ising_numpy as ising_exact_numpy
from stats.moments import Moments
from backend.registry import build_backend_dict
from backend.array_backend import get_array_backend

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

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

_EXACT_THERMO_BACKENDS = build_backend_dict(
    required={"numba": ising_exact_numba.get_exact_thermodynamics,
              "numpy": ising_exact_numpy.get_exact_thermodynamics},
    optional={"cupy": ("stats.exact.ising_cupy", "get_exact_thermodynamics"),
              "jax": ("stats.exact.ising_jax", "get_exact_thermodynamics")},
)

@dataclass
class IsingModel:
    """Pairwise Ising model over `n_sites` binary spins, with pluggable array backends."""

    n_sites: int
    backend: str = "numba"
    gauge: bool=True

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Available: {list(_BACKENDS)}"
            )
        self.array_backend = get_array_backend(self.backend)

    def _validate_params(self, h: Array, J: Array) -> None:
        if h.shape != (self.n_sites,):
            raise ValueError(f"h has shape {h.shape}, expected ({self.n_sites},)")
        if J.shape != (self.n_sites, self.n_sites):
            raise ValueError(f"J has shape {J.shape}, expected ({self.n_sites}, {self.n_sites})")
        if not self.array_backend.xp.allclose(J, J.T):
            raise ValueError(f"The coupling matrix is not symmetric")

    def simulate(self, h: Array, J: Array, samples: int, iterations=1000, seed=0) -> Array:
        """
        Draw 'sample' spin configurations via MCMC with the parameters (h, J).

        Returns an array shaped(samples, n_sites)
        """
        self._validate_params(h, J)
        
        h_array = self.array_backend.xp.asarray(h)
        J_array = self.array_backend.xp.asarray(J)
        
        kwargs = self.array_backend.get_kernel_kwargs(seed)
        
        kernel = _BACKENDS[self.backend]
        return kernel(h_array, J_array, samples, iterations, **kwargs)

    def random_params(self, loc_h=0.0, scale_h =0.3, loc_J=0.0, scale_J=0.3, seed=0)-> tuple[Array, Array]:
        """Sample random (h, J) from independent Gaussians, optionally gauge-fixed if self.gauge is True."""
        J = self.array_backend.random_normal((self.n_sites, self.n_sites), loc_J, scale_J, seed)
        h = self.array_backend.random_normal((self.n_sites,), loc_h, scale_h, seed + 1)
        
        if self.gauge:
            h, J = self.apply_gauge(h, J)
        return h, J

    def compute_moments(self, samples: Array) -> Moments:
        """Compute empirical first and second moments from a sample array of shape (n_samples, n_sites)."""
        mean_s = samples.mean(axis=0)                     
        mean_ss = (samples.T @ samples) / samples.shape[0] 
        return Moments(mean_s=mean_s, mean_ss=mean_ss)

    def compute_energy(self, h: Array, J: Array, samples: Array) -> Array:
        """Compute the Ising energy for each sample in `samples`, shape (n_samples,)."""
        xp = self.array_backend.xp
        e_h = samples @ h
        e_J = 0.5 * xp.einsum('ni,ij,nj->n', samples, J, samples)
        return -(e_h + e_J)

    def interaction_energy(self, J: Array, mean_ss: Array) -> float:
        """Compute the average interaction energy contributed by the coupling term."""
        return 0.5 * self.array_backend.xp.einsum('ij,ij->', J, mean_ss)

    def reference_free_energy(self, h: Array) -> float:
        """Compute the free energy of the non-interacting (J=0) reference system with fields h."""
        xp = self.array_backend.xp
        return -xp.sum(xp.logaddexp(h, -h))

    def project_to_gauge(self, h: Array, J: Array) -> tuple[Array, Array]:
        """Symmetrize J and zero its diagonal; h is passed through unchanged."""
        J = (J + J.T) / 2
        J = self.array_backend.zero_diagonal(J)
        return h, J

    def apply_gauge(self, h: Array, J: Array) -> tuple[Array, Array]:
        """Apply the model's gauge-fixing convention to (h, J)."""
        return self.project_to_gauge(h, J)

    def exact_statistics(self, h: Array, J: Array) -> tuple[Array, Array]:
        """Compute exact (mean_s, mean_ss) via full enumeration over 2**n_sites states."""
        self._validate_params(h, J)
        return _EXACT_BACKENDS[self.backend](h, J)

    def exact_thermodynamics(self, h: Array, J: Array) -> tuple[float, float, float]:
        """Compute exact (entropy, enthalpy, heat_capacity) via full enumeration over 2**n_sites states."""
        self._validate_params(h, J)
        return _EXACT_THERMO_BACKENDS[self.backend](h, J)