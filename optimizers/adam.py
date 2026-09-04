from dataclasses import dataclass, field
from typing import Any

@dataclass
class Adam:
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m_h: Any = field(default=None, repr=False)
    _v_h: Any = field(default=None, repr=False)
    _m_J: Any = field(default=None, repr=False)
    _v_J: Any = field(default=None, repr=False)
    _t: int = 0


    def step(self, h, J, grad_h, grad_J, model):
        xp = model.array_backend.xp

        if self._m_h is None:
            self._m_h, self._v_h = xp.zeros_like(h), xp.zeros_like(h)
            self._m_J, self._v_J = xp.zeros_like(J), xp.zeros_like(J)

        self._t += 1
        self._m_h = self.beta1 * self._m_h + (1 - self.beta1) * grad_h
        self._v_h = self.beta2 * self._v_h + (1 - self.beta2) * grad_h**2
        self._m_J = self.beta1 * self._m_J + (1 - self.beta1) * grad_J
        self._v_J = self.beta2 * self._v_J + (1 - self.beta2) * grad_J**2

        m_h_hat = self._m_h / (1 - self.beta1**self._t)
        v_h_hat = self._v_h / (1 - self.beta2**self._t)
        m_J_hat = self._m_J / (1 - self.beta1**self._t)
        v_J_hat = self._v_J / (1 - self.beta2**self._t)

        h_new = h + self.lr * m_h_hat / (xp.sqrt(v_h_hat) + self.eps)
        J_new = J + self.lr * m_J_hat / (xp.sqrt(v_J_hat) + self.eps)

        if model is not None:
            h_new, J_new = model.apply_gauge(h_new, J_new)
        return h_new, J_new