from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from objectives.gradient import Gradient
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset
Estimator = Any  # e.g. ExactEstimator | McmcEstimator

@dataclass
class MomentMatchingObjective:
    """Average log-likelihood of the data; its gradient pushes model moments toward the empirical ones.

    The gradient only needs the estimator's moments (exact or MCMC). The value also needs
    log Z, so it is only available when the estimator provides `log_partition` (e.g. ExactEstimator).
    """

    estimator: Estimator

    def compute_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> Gradient:
        """Return (empirical - model) moments."""
        empirical = dataset.moments
        model_moments = self.estimator.estimate(model, h, J)
        return Gradient(
            grad_h=empirical.mean_s - model_moments.mean_s,
            grad_J=0.5 * (empirical.mean_ss - model_moments.mean_ss),
        )

    def compute_value(self, model: Model, h: Array, J: Array, dataset: Dataset) -> float:
        """Return h.<s> + 1/2 sum J<ss> - log Z at the empirical moments."""
        if not hasattr(self.estimator, "log_partition"):
            raise NotImplementedError(
                f"{type(self.estimator).__name__} cannot compute log Z, so the likelihood value is unavailable")
        xp = model.array_backend.xp
        empirical = dataset.moments
        log_Z = self.estimator.log_partition(model, h, J)
        return xp.sum(h * empirical.mean_s) + model.interaction_energy(J, empirical.mean_ss) - log_Z

    def compute_value_and_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> tuple[float, Gradient]:
        """Return (log-likelihood value, gradient) at (h, J)."""
        return self.compute_value(model, h, J, dataset), self.compute_gradient(model, h, J, dataset)
