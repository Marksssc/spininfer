from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array

def default_regularization_strength(n_samples: int) -> float:
    """From the thesis: appropriate lambda_h / lambda_J both lie around 1/N."""
    return 1.0 / n_samples

@dataclass
class L2Regularizer:
    """L2 (ridge) penalty: strength/2 * ||h||^2 and/or strength/2 * ||J||^2."""

    strength: float = 0.01
    apply_to_h: bool = False
    apply_to_J: bool = True

    def penalty(self, h: Array, J: Array, model: Model) -> float:
        """Return the scalar L2 penalty value for the selected parameters."""
        xp = model.array_backend.xp
        p = 0.0
        if self.apply_to_h:
            p += 0.5 * self.strength * xp.sum(h**2)
        if self.apply_to_J:
            p += 0.5 * self.strength * xp.sum(J**2)
        return p

    def gradient(self, h: Array, J: Array, model: Model) -> tuple[Array, Array]:
        """Return (grad_h, grad_J) of the L2 penalty (zero where not applied)."""
        xp = model.array_backend.xp
        grad_h = self.strength * h if self.apply_to_h else xp.zeros_like(h)
        grad_J = self.strength * J if self.apply_to_J else xp.zeros_like(J)
        return grad_h, grad_J

@dataclass
class L1Regularizer:
    """L1 (lasso) penalty: strength * |h| and/or strength * |J|, encouraging sparsity."""

    strength: float = 0.01
    apply_to_h: bool = False
    apply_to_J: bool = True

    def penalty(self, h: Array, J: Array, model: Model) -> float:
        """Return the scalar L1 penalty value for the selected parameters."""
        xp = model.array_backend.xp
        p = 0.0
        if self.apply_to_h:
            p += self.strength * xp.sum(xp.abs(h))
        if self.apply_to_J:
            p += self.strength * xp.sum(xp.abs(J))
        return p

    def gradient(self, h: Array, J: Array, model: Model) -> tuple[Array, Array]:
        """Return (grad_h, grad_J) of the L1 penalty (subgradient sign(.), zero where not applied)."""
        xp = model.array_backend.xp
        grad_h = self.strength * xp.sign(h) if self.apply_to_h else xp.zeros_like(h)
        grad_J = self.strength * xp.sign(J) if self.apply_to_J else xp.zeros_like(J)
        return grad_h, grad_J

@dataclass
class CompositeRegularizer:
    """Sums the penalty/gradient of several regularizers."""

    regularizers: list = field(default_factory=list)

    def penalty(self, h: Array, J: Array, model: Model) -> float:
        """Return the sum of penalties over all regularizers."""
        return sum(r.penalty(h, J, model) for r in self.regularizers)

    def gradient(self, h: Array, J: Array, model: Model) -> tuple[Array, Array]:
        """Return the elementwise sum of (grad_h, grad_J) over all regularizers."""
        grad_hs, grad_Js = zip(*(r.gradient(h, J, model) for r in self.regularizers))
        return sum(grad_hs), sum(grad_Js)
