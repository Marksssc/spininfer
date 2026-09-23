from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from objectives.gradient import Gradient
from models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset
Objective = Any  # e.g. MomentMatchingObjective | PleIsingObjective | PlePottsObjective
Regularizer = Any  # e.g. L2Regularizer | L1Regularizer | CompositeRegularizer

@dataclass
class RegularizedObjective:
    """Wraps an objective, subtracting a regularizer's penalty/gradient from its value/gradient."""

    objective: Objective
    regularizer: Regularizer

    def compute_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> Gradient:
        """Return the wrapped objective's gradient minus the regularizer's gradient."""
        grad = self.objective.compute_gradient(model, h, J, dataset)
        reg_grad_h, reg_grad_J = self.regularizer.gradient(h, J, model)
        return Gradient(grad_h=grad.grad_h - reg_grad_h, grad_J=grad.grad_J - reg_grad_J)

    def compute_value_and_gradient(self, model: Model, h: Array, J: Array, dataset: Dataset) -> tuple[float, Gradient]:
        """Return (value, gradient), both with the regularizer's penalty/gradient subtracted."""
        value, grad = self.objective.compute_value_and_gradient(model, h, J, dataset)
        reg_grad_h, reg_grad_J = self.regularizer.gradient(h, J, model)
        regularized_grad = Gradient(grad_h=grad.grad_h - reg_grad_h, grad_J=grad.grad_J - reg_grad_J)
        regularized_value = value - self.regularizer.penalty(h, J, model)
        return regularized_value, regularized_grad
