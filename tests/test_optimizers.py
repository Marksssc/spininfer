import numpy as np
import pytest
from spininfer.optimizers.gradient_ascent import GradientAscent
from spininfer.optimizers.adam import Adam
from spininfer.models.ising import IsingModel

@pytest.mark.parametrize("optimizer_cls", [GradientAscent, Adam])
def test_step_moves_toward_gradient(optimizer_cls, backend, to_numpy):
    model = IsingModel(n_sites=5, backend=backend)
    xp = model.array_backend.xp
    h, J = model.random_params(seed=0)
    grad_h = xp.asarray(np.ones_like(h))
    grad_J = xp.asarray(np.zeros_like(J))

    optimizer = optimizer_cls(lr=0.1)
    h_new, J_new = optimizer.step(h, J, grad_h, grad_J, model)

    assert h_new.shape == h.shape
    assert np.all(to_numpy(h_new) > to_numpy(h))   

@pytest.mark.parametrize("optimizer_cls", [GradientAscent, Adam])
def test_step_applies_gauge(optimizer_cls, backend, to_numpy):
    model = IsingModel(n_sites=5, backend=backend)
    xp = model.array_backend.xp
    h, J = model.random_params(seed=0)
    rng = np.random.default_rng(1)
    grad_h = xp.asarray(rng.normal(size=5))
    grad_J = xp.asarray(rng.normal(size=(5, 5)))

    optimizer = optimizer_cls(lr=0.1)
    h_new, J_new = optimizer.step(h, J, grad_h, grad_J, model)

    assert np.allclose(to_numpy(J_new), to_numpy(J_new.T))
    assert np.allclose(np.diag(to_numpy(J_new)), 0.0)

def test_adam_state_persists_across_steps(backend, to_numpy):
    model = IsingModel(n_sites=5, backend=backend)
    xp = model.array_backend.xp

    h, J = model.random_params(seed=0)
    grad_h_a, grad_J_a = xp.asarray(np.ones_like(h)), xp.asarray(np.zeros_like(J))
    grad_h_b, grad_J_b = -0.5 * xp.asarray(np.ones_like(h)), xp.asarray(np.zeros_like(J))

    optimizer = Adam(lr=0.1)
    h1, J1 = optimizer.step(h, J, grad_h_a, grad_J_a, model)
    h2, J2 = optimizer.step(h1, J1, grad_h_b, grad_J_b, model)

    fresh = Adam(lr=0.1)
    h2_fresh, _ = fresh.step(h1, J1, grad_h_b, grad_J_b, model)

    assert not np.allclose(to_numpy(h2), to_numpy(h2_fresh))