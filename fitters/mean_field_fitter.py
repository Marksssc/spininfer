from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from convergence.criteria import FitResult
from models.ising import IsingModel
from mean_field.ising import naive_mean_field

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset


@dataclass
class NaiveMeanFieldFitter:
    """Fits (h, J) by naive mean-field inversion of the dataset's empirical moments, closed form."""

    model: IsingModel
    dataset: Dataset

    def fit(self) -> FitResult:
        """Invert `dataset.moments` via naive mean-field and return the resulting (h, J)."""
        moments = self.dataset.moments
        h, J = naive_mean_field(self.model, moments.mean_s, moments.mean_ss)
        return FitResult(h=h, J=J, converged=True, n_steps=1, final_grad_norm=float("nan"))

