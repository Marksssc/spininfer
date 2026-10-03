from __future__ import annotations
from typing import Any

from models.potts import PottsModel

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

_M_CLIP = 1e-9
_P_EPS = 1e-12

def _get_covariance(model: PottsModel, mean_s: Array, mean_ss: Array) -> Array:
    """Helper function for the empirical covariance matrix"""
    xp = model.array_backend.xp
    m = mean_s
    C = mean_ss - m[:, None, :, None] * m[None, :, None, :]
    return C

def naive_mean_field(model: PottsModel, mean_s: Array, mean_ss: Array) -> tuple[Array, Array]:
    """Naive mean-field (nMF) inversion of the Potts model from its one-hot first/second moments.

    The last state of each site is the reference and h is determined from the self-consistent equations.
    """
    xp = model.array_backend.xp
    n, q = model.n_sites, model.n_states

    C = _get_covariance(model, mean_s, mean_ss)[:, :, :-1, :-1]
    C_flat = C.transpose(0, 2, 1, 3).reshape(n * (q - 1), n * (q - 1))
    C_inv = xp.linalg.solve(C_flat, xp.eye(n * (q - 1)))
    J_reduced = -C_inv.reshape(n, q - 1, n, q - 1).transpose(0, 2, 1, 3)

    mask = 1.0 - xp.eye(n)[:, :, None, None]
    J_zeroed = J_reduced * mask
    J = xp.pad(J_zeroed, ((0, 0), (0, 0), (0, 1), (0, 1)))

    log_p = xp.log(xp.clip(mean_s, _P_EPS, None))
    h = log_p - log_p[:, -1:] - xp.einsum('ijab,jb->ia', J, mean_s)

    return model.apply_gauge(h, J)