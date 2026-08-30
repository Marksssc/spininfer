from dataclasses import dataclass, field
import numpy as np

@dataclass
class FitResult:
    h: np.ndarray
    J: np.ndarray
    converged: bool
    n_steps: int
    final_grad_norm: float


@dataclass
class GradientNormConvergence:
    tol: float = 1e-6
    patience: int = 5
    _below_tol_count: int = field(default=0, init=False, repr=False)

    def check(self, grad, h, J) -> bool:
        grad_norm = np.linalg.norm(grad.grad_h) + np.linalg.norm(grad.grad_J)
        if grad_norm < self.tol:
            self._below_tol_count += 1
        else:
            self._below_tol_count = 0
        return self._below_tol_count >= self.patience


@dataclass
class MomentMatchConvergence:
    model: object
    dataset: object
    n_samples: int = 5000
    iterations: int = 1000
    check_every: int = 50
    pcc_tol: float = 0.99
    slope_tol: float = 0.01
    intercept_tol: float = 0.05
    patience: int = 2
    _below_tol_count: int = field(default=0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def check(self, grad, h, J) -> bool:
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

    def _matches(self, real, sim) -> bool:
        real_flat, sim_flat = real.ravel(), sim.ravel()
        if np.std(real_flat) < 1e-8 or np.std(sim_flat) < 1e-8:
            return False
        pcc = np.corrcoef(real_flat, sim_flat)[0, 1]
        slope, intercept = np.polyfit(real_flat, sim_flat, 1)
        return (
            pcc > self.pcc_tol
            and abs(slope - 1.0) < self.slope_tol
            and abs(intercept) < self.intercept_tol*np.abs(real_flat).mean()
        )