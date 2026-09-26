from __future__ import annotations
from typing import Any

from models.ising import IsingModel

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

_M_CLIP = 1e-9
_P_EPS = 1e-12

def _get_covariance(model: IsingModel, mean_s: Array, mean_ss: Array) -> Array:
    """Helper function for the empirical covariance matrix"""
    xp = model.array_backend.xp
    m = mean_s
    C = mean_ss - xp.outer(m, m)
    return C


def naive_mean_field(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Naive mean-field (nMF) inversion of the Ising model from its first/second moments.

    The coupling is J = -C^-1 (off-diagonal only), and h is from m_i = tanh(h_i + sum_j J_ij m_j).
    Exact in the weak-coupling limit, accuracy degrades as couplings grow.
    """
    xp = model.array_backend.xp
    m = mean_s
    C = _get_covariance(model, m, mean_ss)
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
    C = _get_covariance(model, m, mean_ss)
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

def independent_pair_approximation(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Independent pair approximation for the Ising model, needs no matrix inversion."""

    xp = model.array_backend.xp
    m = mean_s

    mi, mj = m[:, None], m[None, :]
    c = xp.where(xp.eye(model.n_sites, dtype=bool), mi * mj, mean_ss)

    p_pp = xp.maximum(1.0 + mi + mj + c, _P_EPS)
    p_mm = xp.maximum(1.0 - mi - mj + c, _P_EPS)
    p_pm = xp.maximum(1.0 + mi - mj - c, _P_EPS)
    p_mp = xp.maximum(1.0 - mi + mj - c, _P_EPS)

    J = 0.25 * (xp.log(p_pp) + xp.log(p_mm) - xp.log(p_pm) - xp.log(p_mp))
    h_pair = 0.25 * (xp.log(p_pp) + xp.log(p_pm) - xp.log(p_mp) - xp.log(p_mm))

    atanh_m = xp.arctanh(xp.clip(m, -1 + _M_CLIP, 1 - _M_CLIP))
    h = atanh_m + xp.sum(h_pair - atanh_m[:, None], axis=1)

    h, J = model.apply_gauge(h, J)
    return h, J

def sessak_monasson_approximation(model: IsingModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Sessak-Monasson (SM) small-correlation expansion for the Ising model using TAP expansion for the h."""

    xp = model.array_backend.xp
    m = mean_s
    C = _get_covariance(model, m, mean_ss)
    C_inv = xp.linalg.solve(C, xp.eye(model.n_sites))
    C_diag = xp.diag(C)

    h_IPA, J_IPA = independent_pair_approximation(model, m, mean_ss)
    denom = xp.outer(C_diag, C_diag) - C**2
    denom = xp.where(xp.eye(model.n_sites, dtype=bool), 1.0, denom)
    correction_terms = - C/(xp.maximum(denom, _P_EPS)) - C_inv

    J = J_IPA + correction_terms
    J = model.array_backend.zero_diagonal(J)

    atanh_m = xp.arctanh(xp.clip(m, -1 + _M_CLIP, 1 - _M_CLIP))
    h = atanh_m - J @ m + m * xp.sum(J**2 * (1.0 - m[None, :]**2), axis=1)

    h, J = model.apply_gauge(h, J)
    return h, J





