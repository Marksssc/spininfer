from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

class ArrayBackend(Protocol):
    """Interface implemented by each array backend (numpy/numba, cupy, jax)."""

    xp: object
    def zero_diagonal(self, J: Array) -> Array:
        """Zero the diagonal of `J` and return it."""
        ...
    def random_normal(self, shape: tuple[int, ...], loc: float, scale: float, seed: int) -> Array:
        """Draw a Gaussian-distributed array of the given `shape` from a seeded RNG."""
        ...
    def get_kernel_kwargs(self, seed: int) -> dict:
        """Build the extra kwargs this backend's simulation kernels need to be seeded."""
        ...
    def is_traced(self, x: Array) -> bool:
        """True if `x` is a placeholder inside jax tracing (jit/vmap), whose values are not known yet."""
        ...

@dataclass
class NumpyBackend:
    """Array backend using numpy (also used for the numba backend, which operates on numpy arrays)."""

    xp = np

    def zero_diagonal(self, J: Array) -> Array:
        """Zero the diagonal of `J` in place and return it."""
        np.fill_diagonal(J, 0.0)
        return J

    def random_normal(self, shape: tuple[int, ...], loc: float, scale: float, seed: int) -> Array:
        """Draw a Gaussian-distributed array of the given `shape` from a seeded RNG."""
        return np.random.default_rng(seed).normal(loc, scale, size=shape)

    def get_kernel_kwargs(self, seed: int) -> dict:
        """Return {"seed": seed}, as expected by the numpy/numba simulation kernels."""
        return {"seed": seed}

    def is_traced(self, x: Array) -> bool:
        """Always False: numpy arrays always hold values."""
        return False

@dataclass
class CupyBackend:
    """Array backend using cupy for GPU-resident arrays."""

    def __post_init__(self) -> None:
        import cupy as cp
        self.xp = cp

    def zero_diagonal(self, J: Array) -> Array:
        """Zero the diagonal of `J` in place and return it."""
        self.xp.fill_diagonal(J, 0.0)
        return J

    def random_normal(self, shape: tuple[int, ...], loc: float, scale: float, seed: int) -> Array:
        """Draw a Gaussian-distributed array on the GPU, seeded via numpy for consistency with other backends."""
        return self.xp.asarray(np.random.default_rng(seed).normal(loc, scale, size=shape))

    def get_kernel_kwargs(self, seed: int) -> dict:
        """Return {"seed": seed}, as expected by the cupy simulation kernels."""
        return {"seed": seed}

    def is_traced(self, x: Array) -> bool:
        """Always False: cupy arrays always hold values."""
        return False

@dataclass
class JaxBackend:
    """Array backend using jax."""

    def __post_init__(self) -> None:
        import jax.numpy as jnp
        self.xp = jnp

    def zero_diagonal(self, J: Array) -> Array:
        """Return a copy of `J` with its diagonal set to zero."""
        n = J.shape[0]
        return J.at[self.xp.diag_indices(n)].set(0.0)

    def random_normal(self, shape: tuple[int, ...], loc: float, scale: float, seed: int) -> Array:
        """Draw a Gaussian-distributed array on-device, seeded via numpy for consistency with other backends."""
        return self.xp.asarray(np.random.default_rng(seed).normal(loc, scale, size=shape))

    def get_kernel_kwargs(self, seed: int) -> dict:
        """Return {"key": <jax PRNGKey>}, as expected by the jax simulation kernels."""
        import jax
        return {"key": jax.random.PRNGKey(seed)}

    def is_traced(self, x: Array) -> bool:
        """True inside jit/vmap tracing, where value-dependent checks cannot run."""
        import jax
        return isinstance(x, jax.core.Tracer)

_BACKENDS = {
    "numpy": NumpyBackend,
    "numba": NumpyBackend,
    "cupy": CupyBackend,
    "jax": JaxBackend,
}

def get_array_backend(name: str) -> ArrayBackend:
    """Instantiate the array backend registered under `name`.

    Raises ValueError if `name` is not one of "numpy", "numba", "cupy", "jax".
    """
    if name not in _BACKENDS:
        raise ValueError(f"Unknown array backend {name!r}. Available: {list(_BACKENDS)}")
    return _BACKENDS[name]()
