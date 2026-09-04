# objectives/PLE_ising.py
from dataclasses import dataclass
from objectives.gradient import Gradient
from objectives import PLE_ising_numpy, PLE_ising_numba
from backend.registry import build_backend_dict

_BACKENDS = build_backend_dict(
    required={
        "numba": PLE_ising_numba.value_and_gradient,
        "numpy": PLE_ising_numpy.value_and_gradient,
    },
    optional={"cupy": ("objectives.PLE_ising_cupy", "value_and_gradient"),
              "jax": ("objectives.PLE_ising_jax", "value_and_gradient")},
)

@dataclass
class PleIsingObjective:
    def compute_value_and_gradient(self, model, h, J, dataset):
        moments = dataset.moments
        value, grad_h, grad_J = _BACKENDS[model.backend](h, J, dataset.samples, moments.mean_s, moments.mean_ss)
        return value, Gradient(grad_h=grad_h, grad_J=grad_J)

    def compute_gradient(self, model, h, J, dataset) -> Gradient:
        _, grad = self.compute_value_and_gradient(model, h, J, dataset)
        return grad