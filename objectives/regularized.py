from dataclasses import dataclass
from objectives.gradient import Gradient

@dataclass
class RegularizedObjective:
    objective: object
    regularizer: object

    def compute_gradient(self, model, h, J, dataset) -> Gradient:
        grad = self.objective.compute_gradient(model, h, J, dataset)
        reg_grad_h, reg_grad_J = self.regularizer.gradient(h, J)
        return Gradient(grad_h=grad.grad_h - reg_grad_h, grad_J=grad.grad_J - reg_grad_J)

    def compute_value_and_gradient(self, model, h, J, dataset):
        value, grad = self.objective.compute_value_and_gradient(model, h, J, dataset)
        reg_grad_h, reg_grad_J = self.regularizer.gradient(h, J)
        regularized_grad = Gradient(grad_h=grad.grad_h - reg_grad_h, grad_J=grad.grad_J - reg_grad_J)
        regularized_value = value - self.regularizer.penalty(h, J)
        return regularized_value, regularized_grad