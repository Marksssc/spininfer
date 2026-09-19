from dataclasses import dataclass
from stats.thermodynamics import Thermodynamics

@dataclass
class ExactThermodynamicsEstimator:
    def estimate(self, model, h, J) -> Thermodynamics:
        entropy, enthalpy, heat_capacity = model.exact_thermodynamics(h, J)
        free_energy = enthalpy - entropy
        return Thermodynamics(entropy=entropy, energy=enthalpy, free_energy=free_energy, heat_capacity=heat_capacity)