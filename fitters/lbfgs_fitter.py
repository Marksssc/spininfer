from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from objectives.gradient import Gradient
from convergence.criteria import FitResult


class _ConvergedEarly(Exception):
    pass


@dataclass
class LbfgsFitter:
    model: object
    objective: object
    dataset: object
    convergence: object = None
    tol: float = 1e-6
    maxiter: int = 500
    scipy_kwargs: dict = field(default_factory=dict)

    @staticmethod
    def flatten(h, J):
        return np.concatenate([h.ravel(), J.ravel()])

    @staticmethod
    def unflatten(x, h_shape, J_shape):
        n_h = int(np.prod(h_shape))
        return x[:n_h].reshape(h_shape), x[n_h:].reshape(J_shape)

    def fit(self, h_init, J_init):
        h0, J0 = self.model.apply_gauge(h_init, J_init)
        h_shape, J_shape = h0.shape, J0.shape
        latest = {"grad": None, "x": None, "iteration_count": 0}

        def cost_and_grad(x):
            h, J = self.unflatten(x, h_shape, J_shape)
            value, grad = self.objective.compute_value_and_gradient(self.model, h, J, self.dataset)
            grad_h, grad_J = self.model.apply_gauge(grad.grad_h, grad.grad_J)
            flat_grad = -self.flatten(grad_h, grad_J)
            latest["grad"] = Gradient(grad_h=grad_h, grad_J=grad_J)
            return -value, flat_grad

        def callback(xk):
            latest["iteration_count"] += 1
            latest["x"] = xk
            if self.convergence is None:
                return
            h, J = self.unflatten(xk, h_shape, J_shape)
            if self.convergence.check(latest["grad"], h, J):
                raise _ConvergedEarly()

        x0 = self.flatten(h0, J0)

        try:
            result = minimize(cost_and_grad, x0, jac=True, method="L-BFGS-B", callback=callback,
                               tol=self.tol, options={"maxiter": self.maxiter}, **self.scipy_kwargs)
            h, J = self.unflatten(result.x, h_shape, J_shape)
            grad_norm = np.linalg.norm(result.jac)
            return FitResult(h=h, J=J, converged=result.success, n_steps=result.nit, final_grad_norm=grad_norm)

        except _ConvergedEarly:
            h, J = self.unflatten(latest["x"], h_shape, J_shape)
            grad = latest["grad"]
            grad_norm = np.linalg.norm(grad.grad_h) + np.linalg.norm(grad.grad_J)
            return FitResult(h=h, J=J, converged=True, n_steps=latest["iteration_count"], final_grad_norm=grad_norm)