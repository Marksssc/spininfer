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
    """Gradient of the moment-matching objective: pushes model moments toward the empirical ones."""

    estimator: Estimator

    def compute_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> Gradient:
        """Return (empirical - model) moments as the ascent gradient of (h, J)."""
        empirical = dataset.moments
        model_moments = self.estimator.estimate(model, h, J)
        return Gradient(
            grad_h=empirical.mean_s - model_moments.mean_s,
            grad_J=empirical.mean_ss - model_moments.mean_ss,
        )
