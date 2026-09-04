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
        n_h = int(xp.prod(h_shape))
        return x[:n_h].reshape(h_shape), x[n_h:].reshape(J_shape)

    def fit(self, h_init, J_init):
        if self.model.backend == "cupy":
            raise NotImplementedError("LbfgsFitter does not support the CuPy backend. Use NumPy or JAX.")
            
        xp = self.model.array_backend.xp

        if self.model.backend == "jax":
            print("Jax does not support custom stopping criteria.")

            from jax.scipy.optimize import minimize
            method = "BFGS" 
            use_callback = False 
        else:
            from scipy.optimize import minimize
            method = "L-BFGS-B"
            use_callback = True

        h0, J0 = self.model.apply_gauge(h_init, J_init)
        h_shape, J_shape = h0.shape, J0.shape
        latest = {"grad": None, "x": None, "iteration_count": 0}

        def cost_and_grad(x):
            h, J = self.unflatten(xp, x, h_shape, J_shape)
            value, grad = self.objective.compute_value_and_gradient(self.model, h, J, self.dataset)
            grad_h, grad_J = self.model.apply_gauge(grad.grad_h, grad.grad_J)

            flat_grad = -self.flatten(xp, grad_h, grad_J)
            latest["grad"] = Gradient(grad_h=grad_h, grad_J=grad_J)
            return -value, flat_grad

        def callback(xk):
            latest["iteration_count"] += 1
            latest["x"] = xk
            if self.convergence is None:
                return
            h, J = self.unflatten(xp, xk, h_shape, J_shape)
            if self.convergence.check(latest["grad"], h, J):
                raise _ConvergedEarly()

        x0 = self.flatten(xp, h0, J0)

        min_kwargs = {
            "jac": True,
            "method": method,
            "tol": self.tol,
            "options": {"maxiter": self.maxiter},
            **self.scipy_kwargs
        }
        if use_callback:
            min_kwargs["callback"] = callback

        try:
            result = minimize(cost_and_grad, x0, **min_kwargs)
            h, J = self.unflatten(xp, result.x, h_shape, J_shape)
            grad_norm = xp.linalg.norm(result.jac)
            
            nit = getattr(result, "nit", self.maxiter)
            success = getattr(result, "success", True)
            
            return FitResult(h=h, J=J, converged=success, n_steps=nit, final_grad_norm=grad_norm)

        except _ConvergedEarly:
            h, J = self.unflatten(xp, latest["x"], h_shape, J_shape)
            grad = latest["grad"]
            grad_norm = xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)
            return FitResult(h=h, J=J, converged=True, n_steps=latest["iteration_count"], final_grad_norm=grad_norm)