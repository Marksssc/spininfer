from dataclasses import dataclass
import numpy as np

@dataclass
class GradientAscent:
    lr: float = 0.01

    def step(self, h, J, grad_h, grad_J, model):
        h_new = h + self.lr * grad_h
        J_new = J + self.lr * grad_J
        if model is not None:
            h_new, J_new = model.apply_gauge(h_new, J_new)
        return h_new, J_new