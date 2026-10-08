from dataclasses import dataclass
from typing import Any

from spininfer.convergence.criteria import FitResult
from spininfer.models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Objective = Any  # e.g. MomentMatchingObjective | PleIsingObjective | PlePottsObjective | RegularizedObjective
Dataset = Any  # data.dataset.Dataset
Optimizer = Any  # e.g. GradientAscent | Adam
Convergence = Any  # e.g. GradientNormConvergence | MomentMatchConvergence | None

@dataclass
class InverseFitter:
    """Fits (h, J) by running a fixed number of custom gradient-ascent steps."""

    model: Model
    objective: Objective
    dataset: Dataset
    optimizer: Optimizer
    convergence: Convergence = None
    n_steps: int = 300
    verbose: bool = False

    def __post_init__(self) -> None:
        """Check whether the user wants to take steps."""
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

    def _grad_norm(self, grad) -> Array:
        """Helper function, returning the gradient norm."""
        xp = self.model.array_backend.xp
        return xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)

    def fit(self, h_init: Array | None = None, J_init: Array | None = None) -> FitResult:
        """Run gradient ascent for n_steps (or until convergence), returning the final FitResult."""
        moments = self.dataset.moments
        xp = self.model.array_backend.xp
        if h_init is None:
            h_init = xp.zeros_like(moments.mean_s)
        if J_init is None:
            J_init = xp.zeros_like(moments.mean_ss)

        h, J = h_init, J_init

        self.optimizer.reset()
        if self.convergence is not None:
            self.convergence.reset()

        for step in range(self.n_steps):
            grad = self.objective.compute_gradient(self.model, h, J, self.dataset)
            h, J = self.optimizer.step(h, J, grad.grad_h, grad.grad_J, self.model)

            if self.verbose and step % 50 == 0:
                grad_norm = self._grad_norm(grad=grad)
                print(f"step {step:4d}  |grad| = {grad_norm:.4f}")

            if self.convergence is not None and self.convergence.check(grad, h, J):
                grad_norm = self._grad_norm(grad=grad)
                return FitResult(h=h, J=J, converged=True, n_steps=step+1, final_grad_norm=float(grad_norm))

        grad_norm = self._grad_norm(grad=grad)
        return FitResult(h=h, J=J, converged=False, n_steps=self.n_steps, final_grad_norm=float(grad_norm))