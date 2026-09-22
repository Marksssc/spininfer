import math
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
    def flatten(xp, h, J):
        return xp.concatenate([h.ravel(), J.ravel()])

    @staticmethod
    def unflatten(xp, x, h_shape, J_shape):
        n_h = int(math.prod(h_shape))
        return x[:n_h].reshape(h_shape), x[n_h:].reshape(J_shape)

    def _to_host(self, array):
        if self.model.backend == "cupy":
            return array.get()
        return np.asarray(array)

    def fit(self, h_init, J_init):
        xp = self.model.array_backend.xp

        h0, J0 = self.model.apply_gauge(h_init, J_init)
        h_shape, J_shape = h0.shape, J0.shape
        n_h = math.prod(h_shape)
        latest = {"grad": None, "x": None, "iteration_count": 0}

        def to_arrays(x):
            return self.unflatten(xp, xp.asarray(x), h_shape, J_shape)

        def cost_and_grad(x):
            h, J = to_arrays(x)
            value, grad = self.objective.compute_value_and_gradient(self.model, h, J, self.dataset)
            grad_h, grad_J = self.model.project_to_gauge(grad.grad_h, grad.grad_J)

            latest["grad"] = Gradient(grad_h=grad_h, grad_J=grad_J)
            flat_grad = -self.flatten(xp, grad_h, grad_J)
            return float(-value), self._to_host(flat_grad).astype(np.float64, copy=False)

        def callback(xk):
            latest["iteration_count"] += 1
            latest["x"] = xk
            if self.convergence is None:
                return
            h, J = to_arrays(xk)
            if self.convergence.check(latest["grad"], h, J):
                raise _ConvergedEarly()

        x0 = self._to_host(self.flatten(xp, h0, J0)).astype(np.float64, copy=False)

        min_kwargs = {
            "jac": True,
            "method": "L-BFGS-B",
            "tol": self.tol,
            "callback": callback,
            "options": {"maxiter": self.maxiter},
            **self.scipy_kwargs,
        }
        try:
            result = minimize(cost_and_grad, x0, **min_kwargs)
        except _ConvergedEarly:
            h, J = to_arrays(latest["x"])
            grad = latest["grad"]
            grad_norm = float(xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J))
            return FitResult(h=h, J=J, converged=True, n_steps=latest["iteration_count"], final_grad_norm=grad_norm)

        h, J = to_arrays(result.x)
        grad_norm = float(np.linalg.norm(result.jac[:n_h]) + np.linalg.norm(result.jac[n_h:]))
        return FitResult(h=h, J=J, converged=result.success, n_steps=result.nit, final_grad_norm=grad_norm)