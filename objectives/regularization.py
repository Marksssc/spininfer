from dataclasses import dataclass, field

def default_regularization_strength(n_samples: int) -> float:
    """From the thesis: appropriate lambda_h / lambda_J both lie around 1/N."""
    return 1.0 / n_samples

@dataclass
class L2Regularizer:
    strength: float = 0.01
    apply_to_h: bool = False
    apply_to_J: bool = True

    def penalty(self, h, J, model):
        xp = model.array_backend.xp
        p = 0.0
        if self.apply_to_h:
            p += 0.5 * self.strength * xp.sum(h**2)
        if self.apply_to_J:
            p += 0.5 * self.strength * xp.sum(J**2)
        return p

    def gradient(self, h, J, model):
        xp = model.array_backend.xp
        grad_h = self.strength * h if self.apply_to_h else xp.zeros_like(h)
        grad_J = self.strength * J if self.apply_to_J else xp.zeros_like(J)
        return grad_h, grad_J

@dataclass
class L1Regularizer:
    strength: float = 0.01
    apply_to_h: bool = False
    apply_to_J: bool = True

    def penalty(self, h, J, model):
        xp = model.array_backend.xp
        p = 0.0
        if self.apply_to_h:
            p += self.strength * xp.sum(xp.abs(h))
        if self.apply_to_J:
            p += self.strength * xp.sum(xp.abs(J))
        return p

    def gradient(self, h, J, model):
        xp = model.array_backend.xp
        grad_h = self.strength * xp.sign(h) if self.apply_to_h else xp.zeros_like(h)
        grad_J = self.strength * xp.sign(J) if self.apply_to_J else xp.zeros_like(J)
        return grad_h, grad_J

@dataclass
class CompositeRegularizer:
    regularizers: list = field(default_factory=list)

    def penalty(self, h, J, model):
        return sum(r.penalty(h, J, model) for r in self.regularizers)

    def gradient(self, h, J, model):
        grad_hs, grad_Js = zip(*(r.gradient(h, J, model) for r in self.regularizers))
        return sum(grad_hs), sum(grad_Js)