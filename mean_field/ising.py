from __future__ import annotations
from typing import Any

from models.ising import IsingModel

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

_M_CLIP = 1e-9


def naive_mean_field(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Naive mean-field (nMF) inversion of a pairwise Ising model from its first/second moments.

    The coupling is J = -C^-1 (off-diagonal only), and h is from m_i = tanh(h_i + sum_j J_ij m_j).
    Exact in the weak-coupling limit, accuracy degrades as couplings grow.
    """
    xp = model.array_backend.xp
    m = mean_s
    C = mean_ss - xp.outer(m, m)
    J = model.array_backend.zero_diagonal(-xp.linalg.inv(C))
    m_clipped = xp.clip(m, -1 + _M_CLIP, 1 - _M_CLIP)
    h = xp.arctanh(m_clipped) - J @ m
    return h, J

