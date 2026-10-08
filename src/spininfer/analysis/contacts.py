from __future__ import annotations
from typing import Any

from spininfer.models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

def frobenius_norm(model: Model, J: Array, excluded_states: tuple[int, ...] = (0,)) -> Array:
    """Return the (n_sites, n_sites) Frobenius norms of the Potts coupling blocks J_ij, leaving out `excluded_states` (default: the gap)."""
    xp = model.array_backend.xp
    keep = xp.asarray([a for a in range(model.n_states) if a not in excluded_states])
    J_kept = J[:, :, keep][:, :, :, keep]
    F = xp.sqrt((J_kept ** 2).sum(axis=(2, 3)))
    return F * (1.0 - xp.eye(model.n_sites))

def apc(model: Model, F: Array) -> Array:
    """Return F with the average product correction applied (Dunn et al. 2008, Ekeberg et al. 2013), with a zero diagonal."""
    xp = model.array_backend.xp
    n = model.n_sites
    row_mean = F.sum(axis=1) / (n - 1)
    total_mean = F.sum() / (n * (n - 1))
    F_apc = F - xp.outer(row_mean, row_mean) / total_mean
    return F_apc * (1.0 - xp.eye(n))

def contact_scores(model: Model, J: Array, excluded_states: tuple[int, ...] = (0,)) -> Array:
    """Return the standard DCA contact score per pair: the Frobenius norm of J_ij followed by the APC."""
    return apc(model, frobenius_norm(model, J, excluded_states=excluded_states))
