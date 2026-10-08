import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from spininfer.objectives.gradient import Gradient
from spininfer.convergence.criteria import FitResult
from spininfer.models import Model

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Objective = Any  # e.g. MomentMatchingObjective | PleIsingObjective | PlePottsObjective | RegularizedObjective
Dataset = Any  # data.dataset.Dataset
Convergence = Any  # e.g. GradientNormConvergence | MomentMatchConvergence | None


class _ConvergedEarly(Exception):
    pass


@dataclass
class LbfgsFitter:
    """Fits (h, J) via scipy's (numpy/numba/cupy) or optax's (jax) L-BFGS optimizer."""

    model: Model
    objective: Objective
    dataset: Dataset
    convergence: Convergence = None
    tol: float = 1e-6
    maxiter: int = 500
    scipy_kwargs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def flatten(xp: Any, h: Array, J: Array) -> Array:
        """Flatten and concatenate h and J into a single 1D array."""
        return xp.concatenate([h.ravel(), J.ravel()])

    @staticmethod
    def unflatten(xp: Any, x: Array, h_shape: tuple[int, ...], J_shape: tuple[int, ...]) -> tuple[Array, Array]:
        """Inverse of flatten: split a 1D array back into (h, J) with the given shapes."""
        n_h = int(math.prod(h_shape))
        return x[:n_h].reshape(h_shape), x[n_h:].reshape(J_shape)

    def _to_host(self, array):
        if self.model.backend == "cupy":
            return array.get()
        return np.asarray(array)

    def _fit_scipy(self, h_init: Array, J_init: Array) -> FitResult:
        """Run L-BFGS to convergence (or maxiter), returning the final FitResult using the SciPy library."""
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
            "options": {"maxiter": self.maxiter, "gtol": self.tol, "ftol": 1e-15},
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

    def _fit_jax(self, h_init: Array, J_init: Array) -> FitResult:
        """Run L-BFGS to convergence (or maxiter) via optax's `lbfgs` optimizer."""
        import jax
        import optax

        xp = self.model.array_backend.xp
        h0, J0 = self.model.apply_gauge(h_init, J_init)

        def value_fn(params):
            h, J = params
            value, _ = self.objective.compute_value_and_gradient(self.model, h, J, self.dataset)
            return -value  

        solver = optax.lbfgs()

        def step(params, opt_state):
            h, J = params
            value, grad = self.objective.compute_value_and_gradient(self.model, h, J, self.dataset)
            grad_h, grad_J = self.model.project_to_gauge(grad.grad_h, grad.grad_J)
            neg_grad = (-grad_h, -grad_J)

            updates, opt_state = solver.update(
                neg_grad, opt_state, params, value=-value, grad=neg_grad, value_fn=value_fn,
            )
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, grad_h, grad_J

        step = jax.jit(step)
        params0 = (h0, J0)
        opt_state0 = solver.init(params0)

        if self.convergence is None:
            import jax.numpy as jnp

            def cond_fn(carry):
                _, _, grad_h, grad_J, n_steps = carry
                grad_norm = jnp.linalg.norm(grad_h) + jnp.linalg.norm(grad_J)
                return (n_steps == 0) | ((grad_norm > self.tol) & (n_steps < self.maxiter))

            def body_fn(carry):
                params, opt_state, _, _, n_steps = carry
                params, opt_state, grad_h, grad_J = step(params, opt_state)
                return params, opt_state, grad_h, grad_J, n_steps + 1

            init_carry = (params0, opt_state0, xp.zeros_like(h0), xp.zeros_like(J0), 0)
            params, opt_state, grad_h, grad_J, n_steps = jax.lax.while_loop(cond_fn, body_fn, init_carry)

            n_steps = int(n_steps)
            grad_norm = float(xp.linalg.norm(grad_h) + xp.linalg.norm(grad_J))
            h, J = params
            converged = grad_norm <= self.tol
            return FitResult(h=h, J=J, converged=converged, n_steps=n_steps, final_grad_norm=grad_norm)

        params, opt_state = params0, opt_state0
        h, J = params
        grad = Gradient(grad_h=xp.zeros_like(h0), grad_J=xp.zeros_like(J0))

        converged = False
        n_steps = 0
        for n_steps in range(1, self.maxiter + 1):
            params, opt_state, grad_h, grad_J = step(params, opt_state)
            h, J = params
            grad = Gradient(grad_h=grad_h, grad_J=grad_J)

            if self.convergence.check(grad, h, J):
                converged = True
                break

        grad_norm = float(xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J))
        return FitResult(h=h, J=J, converged=converged, n_steps=n_steps, final_grad_norm=grad_norm)

    def fit(self, h_init: Array | None = None, J_init: Array | None = None) -> FitResult:
        """Run L-BFGS to convergence (or maxiter), returning the final FitResult."""
        moments = self.dataset.moments
        xp = self.model.array_backend.xp
        if h_init is None:
            h_init = xp.zeros_like(moments.mean_s)
        if J_init is None:
            J_init = xp.zeros_like(moments.mean_ss)
        
        if self.convergence is not None:
            self.convergence.reset()
        
        if self.model.backend == "jax":
            return self._fit_jax(h_init, J_init)
        return self._fit_scipy(h_init, J_init)
    