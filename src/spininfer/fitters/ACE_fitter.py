from dataclasses import dataclass, field
from typing import Any
from itertools import chain
import warnings

from spininfer.cluster_expansion._cluster_fit import ClusterFit
from spininfer.cluster_expansion.expansion import fit_level, next_level, assemble, compute_delta_S
from spininfer.objectives.regularization import L2Regularizer
from spininfer.convergence.criteria import FitResult
from spininfer.models import Model

Dataset = Any  # data.dataset.Dataset
Regularizer = Any  # e.g. L2Regularizer | CompositeRegularizer

@dataclass
class ClusterExpansionFitResult(FitResult):
    """FitResult plus the expansion's entropy estimate, kept clusters per level and delta S of every fitted cluster."""

    entropy: float
    kept: dict[int, set[tuple[int, ...]]]
    delta_S: dict[tuple[int, ...], float]


@dataclass
class ACEfitter:
    model: Model
    dataset: Dataset
    threshold: float
    regularizer: Regularizer = field(default_factory=L2Regularizer)
    max_size: int=15
    tol: float = 1e-8
    maxiter: int=1000

    def fit(self) -> ClusterExpansionFitResult:
        """Grow clusters level by level, keep those with sufficient entropy difference, and assemble (h, J) from them."""
        self._fits: dict[tuple[int, ...], ClusterFit] = {}
        self._delta_S: dict[tuple[int, ...], float] = {}
        self._kept: dict[int, set[tuple[int, ...]]] = {}

        singletons = [(i,) for i in range(self.model.n_sites)]
        self._fit_and_score(singletons)
        self._kept[1] = set(singletons)

        for k in range(1, self.max_size):
            candidates = next_level(self._kept[k], self.model.n_sites)
            if not candidates:
                break
            self._fit_and_score(candidates)
            self._kept[k + 1] = {c for c in candidates if abs(self._delta_S[c]) > self.threshold}

        converged = all(fit.converged for fit in self._fits.values())
        if not converged:
            warnings.warn("some cluster fits did not converge!")

        kept = list(chain.from_iterable(self._kept.values()))
        h, J = assemble(self.model, kept, self._fits)
        entropy = sum(self._delta_S[c] for c in kept)
        return ClusterExpansionFitResult(h=h, J=J, converged=converged, n_steps=len(self._kept),
                                         final_grad_norm=float("nan"), entropy=entropy,
                                         kept=self._kept, delta_S=self._delta_S)

    def _fit_and_score(self, clusters: list[tuple[int, ...]]) -> None:
        """Fit clusters and store their fits and delta S."""
        self._fits.update(fit_level(self.model, self.dataset, clusters, regularizer=self.regularizer,
                                    tol=self.tol, maxiter=self.maxiter))
        for cluster in clusters:
            self._delta_S[cluster] = compute_delta_S(cluster, self._fits[cluster].entropy, self._delta_S)
