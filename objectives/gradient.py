from __future__ import annotations
from dataclasses import dataclass
from typing import Any

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class Gradient:
    """Gradient of an objective w.r.t. (h, J)."""
    grad_h: Array
    grad_J: Array
