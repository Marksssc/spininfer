from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from stats.moments import Moments
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class ExactEstimator:
    """Computes moments via full enumeration (`model.exact_statistics`), no sampling noise."""

    def estimate(self, model: Model, h: Array, J: Array) -> Moments:
        """Return the exact (mean_s, mean_ss) moments of `model` at parameters (h, J)."""
        mean_s, mean_ss = model.exact_statistics(h, J)
        return Moments(mean_s=mean_s, mean_ss=mean_ss)
