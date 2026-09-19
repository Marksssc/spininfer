from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class Thermodynamics:
    entropy: float
    energy: float
    free_energy: float
    heat_capacity: float