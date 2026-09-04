from dataclasses import dataclass
from convergence.criteria import FitResult

@dataclass
class InverseFitter:
    model: object
    objective: object
    dataset: object
    optimizer: object
    convergence: object = None
    n_steps: int = 300
    verbose: bool = True

    def fit(self, h_init, J_init):
        h, J = h_init, J_init
        xp = self.model.array_backend.xp

        for step in range(self.n_steps):
            grad = self.objective.compute_gradient(self.model, h, J, self.dataset)
            h, J = self.optimizer.step(h, J, grad.grad_h, grad.grad_J, self.model)

            if self.verbose and step % 50 == 0:
                grad_norm = xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)
                print(f"step {step:4d}  |grad| = {grad_norm:.4f}")

            if self.convergence is not None and self.convergence.check(grad, h, J):
                grad_norm = xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)
                return FitResult(h=h, J=J, converged=True, n_steps=step, final_grad_norm=grad_norm)

        grad_norm = xp.linalg.norm(grad.grad_h) + xp.linalg.norm(grad.grad_J)
        return FitResult(h=h, J=J, converged=False, n_steps=self.n_steps, final_grad_norm=grad_norm)