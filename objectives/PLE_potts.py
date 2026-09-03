import numpy as np

from dataclasses import dataclass
from objectives.gradient import Gradient
from objectives import PLE_potts_numpy, PLE_potts_numba, PLE_potts_cupy, PLE_potts_jax
from backend.registry import build_backend_dict

_BACKENDS = build_backend_dict(
    required={
        "numba": PLE_potts_numba.compute_value_and_gradient,
        "numpy": PLE_potts_numpy.compute_value_and_gradient,
    },
    optional={"cupy": PLE_potts_cupy.compute_value_and_gradient,
              "jax": PLE_potts_jax.compute_value_and_gradient},
)

@dataclass
class PlePottsObjective:
    def compute_value_and_gradient(self, model, h, J, dataset):
        moments = dataset.moments
        data = dataset.samples
        value, grad_h, grad_J = _BACKENDS[model.backend](h, J, data, moments.mean_s, moments.mean_ss)
        return value, Gradient(grad_h=grad_h, grad_J=grad_J)

    def compute_gradient(self, model, h, J, dataset) -> Gradient:
        _, grad = self.compute_value_and_gradient(model, h, J, dataset)
        return grad
