from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from objectives.gradient import Gradient
from objectives import PLE_potts_numpy, PLE_potts_numba
from backend.registry import build_backend_dict
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset

_BACKENDS = build_backend_dict(
    required={
        "numba": PLE_potts_numba.value_and_gradient,
        "numpy": PLE_potts_numpy.value_and_gradient,
    },
    optional={"cupy": ("objectives.PLE_potts_cupy", "value_and_gradient"),
              "jax": ("objectives.PLE_potts_jax", "value_and_gradient")},
)

@dataclass
class PlePottsObjective:
    """Pseudolikelihood objective for the Potts model, dispatched to the model's array backend."""

    def compute_value_and_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> tuple[float, Gradient]:
        """Return (pseudolikelihood value, gradient) at (h, J) on `dataset.samples`."""
        moments = dataset.moments
        data = dataset.samples
        value, grad_h, grad_J = _BACKENDS[model.backend](h, J, data, moments.mean_s, moments.mean_ss)
        return value, Gradient(grad_h=grad_h, grad_J=grad_J)

    def compute_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> Gradient:
        """Return just the gradient half of compute_value_and_gradient."""
        _, grad = self.compute_value_and_gradient(model, h, J, dataset)
        return grad
