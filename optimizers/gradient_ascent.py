from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

@dataclass
class GradientAscent:
    """Gradient ascent: h, J += lr * grad, then re-applies the model's gauge."""

    lr: float = 0.01

    def step(self, h: Array, J: Array, grad_h: Array, grad_J: Array, model: Model | None) -> tuple[Array, Array]:
        """Take one ascent step on (h, J) and gauge-project the result if `model` is given."""
        h_new = h + self.lr * grad_h
        J_new = J + self.lr * grad_J
        if model is not None:
            h_new, J_new = model.project_to_gauge(h_new, J_new)
        return h_new, J_new
