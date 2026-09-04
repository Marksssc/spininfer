from dataclasses import dataclass
from typing import Protocol
import numpy as np

class ArrayBackend(Protocol):
    xp: object  
    def zero_diagonal(self, J): ...
    def random_normal(self, shape, loc, scale, seed): ...
    def get_kernel_kwargs(self, seed: int) -> dict: ...

@dataclass
class NumpyBackend:
    xp = np
    def zero_diagonal(self, J):
        np.fill_diagonal(J, 0.0)
        return J
    def random_normal(self, shape, loc, scale, seed):
        return np.random.default_rng(seed).normal(loc, scale, size=shape)
    def get_kernel_kwargs(self, seed: int) -> dict: # ADDED
        return {"seed": seed}

@dataclass
class CupyBackend:
    def __post_init__(self):
        import cupy as cp
        self.xp = cp
    def zero_diagonal(self, J):
        self.xp.fill_diagonal(J, 0.0)
        return J
    def random_normal(self, shape, loc, scale, seed):
        return self.xp.random.default_rng(seed).normal(loc, scale, size=shape)
    def get_kernel_kwargs(self, seed: int) -> dict: # ADDED
        return {"seed": seed}

@dataclass
class JaxBackend:
    def __post_init__(self):
        import jax.numpy as jnp
        self.xp = jnp
    def zero_diagonal(self, J):
        n = J.shape[0]
        return J.at[self.xp.diag_indices(n)].set(0.0)
    def random_normal(self, shape, loc, scale, seed):
        import jax
        key = jax.random.PRNGKey(seed)
        return loc + scale * jax.random.normal(key, shape)
    def get_kernel_kwargs(self, seed: int) -> dict: # ADDED
        import jax
        return {"key": jax.random.PRNGKey(seed)}

_BACKENDS = {
    "numpy": NumpyBackend,
    "numba": NumpyBackend,
    "cupy": CupyBackend,
    "jax": JaxBackend,
}

def get_array_backend(name: str) -> ArrayBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown array backend {name!r}. Available: {list(_BACKENDS)}")
    return _BACKENDS[name]()