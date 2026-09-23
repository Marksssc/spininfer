from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from models import Model
from objectives.gradient import Gradient

Array = Any  # backend-dependent: numpy.ndarray | cupy.ndarray | jax.Array
Dataset = Any  # data.dataset.Dataset

@dataclass
class FitResult:
    """Outcome of a fitter run: final (h, J), whether convergence was reached, and diagnostics."""

    h: Array
    J: Array
    converged: bool
    n_steps: int
    final_grad_norm: float


@dataclass
class GradientNormConvergence:
    """Declares convergence once the gradient norm stays below `tol` for `patience` consecutive checks."""

    model: Model
    tol: float = 1e-6
    patience: int = 5
    _below_tol_count: int = field(default=0, init=False, repr=False)

    def check(self, grad: Gradient, h: Array, J: Array) -> bool:
        """Update the below-tolerance streak from `grad`'s norm and return True once it reaches `patience`."""
        xp = self.model.array_backend.xp
        grad_norm = xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)
        if grad_norm < self.tol:
            self._below_tol_count += 1
        else:
            self._below_tol_count = 0
        return self._below_tol_count >= self.patience


@dataclass
class MomentMatchConvergence:
    """Declares convergence once simulated moments track the dataset's moments closely for `patience`
    consecutive checks (checked every `check_every` steps)."""

    model: Model
    dataset: Dataset
    n_samples: int = 5000
    iterations: int = 1000
    check_every: int = 50
    pcc_tol: float = 0.99
    slope_tol: float = 0.01
    intercept_tol: float = 0.05
    patience: int = 2
    _below_tol_count: int = field(default=0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def check(self, grad: Gradient, h: Array, J: Array) -> bool:
        """Every `check_every` steps, simulate at (h, J) and compare moments to the dataset; return True
        once matches have held for `patience` consecutive checks."""
        self._step += 1
        if self._step % self.check_every != 0:
            return False

        sim_samples = self.model.simulate(h, J, self.n_samples, self.iterations, seed=self._step)
        sim_moments = self.model.compute_moments(sim_samples)

        mean_ok = self._matches(self.dataset.moments.mean_s, sim_moments.mean_s)
        corr_ok = self._matches(self.dataset.moments.mean_ss, sim_moments.mean_ss)

        if mean_ok and corr_ok:
            self._below_tol_count += 1
        else:
            self._below_tol_count = 0
        return self._below_tol_count >= self.patience

    def _matches(self, real: Array, sim: Array) -> bool:
        """Check whether `sim` tracks `real`: correlated above `pcc_tol`, slope within `slope_tol` of 1,
        and intercept within `intercept_tol` (scaled by mean |real|) of 0."""
        xp = self.model.array_backend.xp

        real_flat, sim_flat = real.ravel(), sim.ravel()
        if xp.std(real_flat) < 1e-8 or xp.std(sim_flat) < 1e-8:
            return False
        pcc = xp.corrcoef(real_flat, sim_flat)[0, 1]
        slope, intercept = xp.polyfit(real_flat, sim_flat, 1)
        return (
            pcc > self.pcc_tol
            and abs(slope - 1.0) < self.slope_tol
            and abs(intercept) < self.intercept_tol*xp.abs(real_flat).mean()
        )
