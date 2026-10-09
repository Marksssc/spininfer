from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from spininfer.objectives.gradient import Gradient
from spininfer.objectives import ple_potts_numba, ple_potts_numpy
from spininfer.backend.registry import build_backend_dict
from spininfer.models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset

_BACKENDS = build_backend_dict(
    required={
        "numba": ple_potts_numba.value_and_gradient,
        "numpy": ple_potts_numpy.value_and_gradient,
    },
    optional={"cupy": ("spininfer.objectives.ple_potts_cupy", "value_and_gradient"),
              "jax": ("spininfer.objectives.ple_potts_jax", "value_and_gradient")},
)

@dataclass
class PlePottsObjective:
    """Pseudolikelihood objective for the Potts model, dispatched to the model's array backend."""

    def compute_value_and_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> tuple[float, Gradient]:
        """Return (pseudolikelihood value, gradient) at (h, J) on `dataset.samples`."""
        if dataset.samples is None:
            raise ValueError("PLE needs samples, but this dataset only has moments")
        moments = dataset.moments
        data = dataset.samples
        value, grad_h, grad_J = _BACKENDS[model.backend](h, J, data, moments.mean_s, moments.mean_ss)
        return value, Gradient(grad_h=grad_h, grad_J=grad_J)

    def compute_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> Gradient:
        """Return just the gradient half of compute_value_and_gradient."""
        _, grad = self.compute_value_and_gradient(model, h, J, dataset)
        return grad
