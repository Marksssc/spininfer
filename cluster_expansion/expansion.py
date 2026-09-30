from __future__ import annotations
from typing import Any, Iterable
from models import Model
from cluster_expansion._cluster_fit import fit_cluster, ClusterFit
from itertools import combinations
from collections import defaultdict

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset
Convergence = Any  # e.g. GradientNormConvergence | MomentMatchConvergence | None
Regularizer = Any  # e.g. L2Regularizer | CompositeRegularizer

def next_level(kept: set[tuple[int, ...]], n_sites: int) -> list[tuple[int, ...]]:
    """Get all the k+1 cluster candidates from the current clusters."""
    candidates = []
    for cluster in sorted(kept):
        for site in range(cluster[-1] + 1, n_sites):
            candidate = cluster + (site,)
            if all(candidate[:i] + candidate[i + 1:] in kept for i in range(len(cluster))):
                candidates.append(candidate)
    return candidates


def fit_level(model: Model, dataset: Dataset, clusters: list[tuple[int, ...]], regularizer: Regularizer = None,
              tol: float = 1e-8, maxiter: int = 1000) -> dict[tuple[int, ...], ClusterFit]:
    """Fit and get the entropy of all of the clusters for a certain size."""
    return {cluster: fit_cluster(model, dataset, cluster, regularizer=regularizer, tol=tol, maxiter=maxiter)
            for cluster in clusters}


def compute_delta_S(cluster: tuple[int, ...], entropy: float, delta_S: dict[tuple[int, ...], float]) -> float:
    """Compute the entropy change of a given cluster."""
    result = entropy
    for size in range(1, len(cluster)):
        for sub in combinations(cluster, size):
            result -= delta_S[sub]
    return result


def assemble(model: Model, kept: Iterable[tuple[int, ...]],
             fits: dict[tuple[int, ...], ClusterFit]) -> tuple[Array, Array]:
    """Assemble the h and J parameters through the generated and accepteb clusters using Moebius inversion."""
    coefficients = defaultdict(int)
    for cluster in kept:
        for size in range(1, len(cluster) + 1):
            for sub in combinations(cluster, size):
                coefficients[sub] += (-1) ** (len(cluster) - size)

    xp = model.array_backend.xp
    n = model.n_sites
    reference = fits[(0,)]
    h = xp.zeros((n,) + reference.h.shape[1:], dtype=reference.h.dtype)
    J = xp.zeros((n, n) + reference.J.shape[2:], dtype=reference.J.dtype)
    for sub, c in coefficients.items():
        if c == 0:
            continue
        idx = xp.asarray(sub)
        fit = fits[sub]
        if model.backend == "jax":
            h = h.at[idx].add(c * fit.h)
            J = J.at[idx[:, None], idx].add(c * fit.J)
        else:
            h[idx] += c * fit.h
            J[idx[:, None], idx] += c * fit.J
    return h, J




