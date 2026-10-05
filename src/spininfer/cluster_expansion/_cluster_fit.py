from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Sequence

from spininfer.data.dataset import Dataset
from spininfer.stats.moments import Moments
from spininfer.stats.exact_estimator import ExactEstimator
from spininfer.objectives.moment_matching import MomentMatchingObjective
from spininfer.objectives.regularized import RegularizedObjective
from spininfer.fitters.lbfgs_fitter import LbfgsFitter
from spininfer.models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Regularizer = Any  # e.g. L2Regularizer | CompositeRegularizer

@dataclass
class ClusterFit:
    """Exact maximum-likelihood fit restricted to one cluster of sites."""

    sites: tuple[int, ...]
    h: Array
    J: Array
    entropy: float 
    converged: bool

def cluster_dataset(model: Model, dataset: Dataset, sites: Sequence[int]) -> Dataset:
    """Return a moments-only Dataset (and its model) restricted to `sites`; works for Ising and Potts."""
    xp = model.array_backend.xp
    idx = xp.asarray(sites)
    moments = Moments(mean_s=dataset.moments.mean_s[idx],
                      mean_ss=dataset.moments.mean_ss[idx[:, None], idx])
    cluster_model = replace(model, n_sites=len(sites))
    return Dataset(model=cluster_model, moments=moments, n_samples=dataset.n_samples)

def fit_cluster(model: Model, dataset: Dataset, sites: Sequence[int], regularizer: Regularizer = None,
                tol: float = 1e-8, maxiter: int = 1000) -> ClusterFit:
    """Fit (h, J) on the moments of `sites` by exact likelihood and return them with the cluster entropy."""
    sub = cluster_dataset(model, dataset, sites)
    objective = MomentMatchingObjective(ExactEstimator())
    if regularizer is not None:
        objective = RegularizedObjective(objective, regularizer)

    xp = model.array_backend.xp
    h_init = xp.zeros_like(sub.moments.mean_s)
    J_init = xp.zeros_like(sub.moments.mean_ss)
    result = LbfgsFitter(model=sub.model, objective=objective, dataset=sub, tol=tol, maxiter=maxiter).fit(h_init, J_init)

    entropy = -float(objective.compute_value(sub.model, result.h, result.J, sub))
    return ClusterFit(sites=tuple(sites), h=result.h, J=result.J, entropy=entropy, converged=result.converged)