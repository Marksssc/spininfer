from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from convergence.criteria import FitResult
from models.ising import IsingModel
from mean_field.ising import naive_mean_field, TAP_mean_field, independent_pair_approximation, sessak_monasson_approximation

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

@dataclass
class TAPMeanFieldFitter:
    """Fits (h, J) by the TAP method, closed form."""

    model: IsingModel
    dataset: Dataset

    def fit(self) -> FitResult:
        """Get the parameters with the TAP method."""
        moments = self.dataset.moments
        h, J = TAP_mean_field(model=self.model, mean_s=moments.mean_s, mean_ss=moments.mean_ss)
        return FitResult(h=h, J=J, converged=True, n_steps=1, final_grad_norm=float("nan"))

@dataclass
class IndependentPairFitter:
    """Fits (h, J) by the independent pair approximation and returns the resulting (h, J)."""

    model: IsingModel
    dataset: Dataset

    def fit(self) -> FitResult:
        """Get the parameteres with the independent pair approximation."""
        moments = self.dataset.moments
        h, J = independent_pair_approximation(model=self.model, mean_s=moments.mean_s, mean_ss=moments.mean_ss)
        return FitResult(h=h, J=J, converged=True, n_steps=1, final_grad_norm=float("nan"))

@dataclass
class SessakMonassonFitter:
    """Fits (h, J) by the Sessak-Monasson approximation and returns the resulting (h, J)"""

    model: IsingModel
    dataset: Dataset

    def fit(self) -> FitResult:
        """Get the parameteres with the Sessak-Monasson approximation."""
        moments = self.dataset.moments
        h, J = sessak_monasson_approximation(model=self.model, mean_s=moments.mean_s, mean_ss=moments.mean_ss)
        return FitResult(h=h, J=J, converged=True, n_steps=1, final_grad_norm=float("nan"))