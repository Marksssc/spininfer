from dataclasses import dataclass

from stats.moments import Moments
from stats.exact import ising_numba, ising_numpy, ising_cupy, ising_jax, potts_numba, potts_numpy, potts_cupy, potts_jax
from models.ising import IsingModel
from models.potts import PottsModel
from backend.registry import build_backend_dict

_ISING_BACKENDS = build_backend_dict(
    required={
        "numba": ising_numba.get_exact_statistics,
        "numpy": ising_numpy.get_exact_statistics,
    },
    optional={"cupy": ising_cupy.get_exact_statistics,
              "jax": ising_jax.get_exact_statistics},
)
_POTTS_BACKENDS = build_backend_dict(
    required={
        "numba": potts_numba.get_exact_statistics,
        "numpy": potts_numpy.get_exact_statistics,
    },
    optional={"cupy": potts_cupy.get_exact_statistics,
              "jax": potts_jax.get_exact_statistics},
)

@dataclass
class ExactEstimator:
    def estimate(self, model, h, J) -> Moments:
        model._validate_params(h, J)
        if isinstance(model, IsingModel):
            mean_s, mean_ss = _ISING_BACKENDS[model.backend](h, J)
        elif isinstance(model, PottsModel):
            mean_s, mean_ss = _POTTS_BACKENDS[model.backend](h, J)
        else:
            raise TypeError(f"Unknown model type {type(model)}")
        return Moments(mean_s=mean_s, mean_ss=mean_ss)