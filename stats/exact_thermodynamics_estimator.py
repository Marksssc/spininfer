from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from stats.thermodynamics import Thermodynamics
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class ExactThermodynamicsEstimator:
    """Computes thermodynamic quantities via full enumeration (`model.exact_thermodynamics`)."""

    def estimate(self, model: Model, h: Array, J: Array) -> Thermodynamics:
        """Return the exact entropy/energy/free energy/heat capacity of `model` at (h, J)."""
        entropy, enthalpy, heat_capacity = model.exact_thermodynamics(h, J)
        free_energy = enthalpy - entropy
        return Thermodynamics(entropy=entropy, energy=enthalpy, free_energy=free_energy, heat_capacity=heat_capacity)
