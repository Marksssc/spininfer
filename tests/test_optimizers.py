import numpy as np
import pytest
from optimizers.gradient_ascent import GradientAscent
from optimizers.adam import Adam
from models.ising import IsingModel

@pytest.mark.parametrize("optimizer_cls", [GradientAscent, Adam])
def test_step_moves_toward_gradient(optimizer_cls):
    model = IsingModel(n_sites=5, backend="numpy")
    h, J = model.random_params(seed=0)
    grad_h = np.ones_like(h)
    grad_J = np.zeros_like(J)

    optimizer = optimizer_cls(lr=0.1)
    h_new, J_new = optimizer.step(h, J, grad_h, grad_J, model)

    assert h_new.shape == h.shape
    assert np.all(h_new > h)   

@pytest.mark.parametrize("optimizer_cls", [GradientAscent, Adam])
def test_step_applies_gauge(optimizer_cls):
    model = IsingModel(n_sites=5, backend="numpy")
    h, J = model.random_params(seed=0)
    rng = np.random.default_rng(1)
    grad_h = rng.normal(size=5)
    grad_J = rng.normal(size=(5, 5))

    optimizer = optimizer_cls(lr=0.1)
    h_new, J_new = optimizer.step(h, J, grad_h, grad_J, model)

    assert np.allclose(J_new, J_new.T)
    assert np.allclose(np.diag(J_new), 0.0)

def test_adam_state_persists_across_steps():
    model = IsingModel(n_sites=5, backend="numpy")
    h, J = model.random_params(seed=0)
    grad_h, grad_J = np.ones_like(h), np.zeros_like(J)

    optimizer = Adam(lr=0.1)
    h1, J1 = optimizer.step(h, J, grad_h, grad_J, model)
    h2, J2 = optimizer.step(h1, J1, grad_h, grad_J, model)

    fresh = Adam(lr=0.1)
    h2_fresh, _ = fresh.step(h1, J1, grad_h, grad_J, model)

    assert not np.allclose(h2, h2_fresh)