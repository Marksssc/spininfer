from __future__ import annotations
from typing import Any

from models.ising import IsingModel

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

_M_CLIP = 1e-9


def naive_mean_field(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Naive mean-field (nMF) inversion of the Ising model from its first/second moments.

    The coupling is J = -C^-1 (off-diagonal only), and h is from m_i = tanh(h_i + sum_j J_ij m_j).
    Exact in the weak-coupling limit, accuracy degrades as couplings grow.
    """
    xp = model.array_backend.xp
    m = mean_s
    C = mean_ss - xp.outer(m, m)
    C_inv = xp.linalg.solve(C, xp.eye(model.n_sites))
    J = model.array_backend.zero_diagonal(-C_inv)
    m_clipped = xp.clip(m, -1 + _M_CLIP, 1 - _M_CLIP)
    h = xp.arctanh(m_clipped) - J @ m

    h, J = model.apply_gauge(h, J)
    return h, J

def TAP_mean_field(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Thouless-Anderson-Palmer (TAP) approximation for the Ising model with the Onsager term.
    
    The coupling is J = (-1 + sqrt(1 - 8 m_i m_j (C^-1)_ij))/(4 m_i m_j), 
    and h is derived from the TAP correction.
    """
    xp = model.array_backend.xp
    m = mean_s

    outer_prod = xp.outer(m, m)
    C = mean_ss - outer_prod
    C_inv = xp.linalg.solve(C, xp.eye(model.n_sites))

    discriminant = xp.maximum(1.0 - 8.0 * outer_prod * C_inv, 0.0)
    small = xp.abs(outer_prod) < 1e-8
    safe = xp.where(small, 1.0, outer_prod)
    J_tap = (-1.0 + xp.sqrt(discriminant)) / (4.0 * safe)
    J = xp.where(small, -C_inv, J_tap)
    J = model.array_backend.zero_diagonal(J)

    m_clipped = xp.clip(m, -1 + _M_CLIP, 1 - _M_CLIP)
    arctan_term = xp.arctanh(m_clipped)
    mean_field = J @ m
    onsager = m_clipped * xp.sum(J**2 * (1.0 - m[None, :]**2), axis=1)
    h = arctan_term - mean_field + onsager

    h, J = model.apply_gauge(h, J)
    return h, J