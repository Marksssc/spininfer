from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class Thermodynamics:
    entropy: Any
    energy: Any
    free_energy: Any