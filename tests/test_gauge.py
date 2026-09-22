import numpy as np
import pytest
import itertools
from models.ising import IsingModel
from models.potts import PottsModel

def _all_states(model):
    if isinstance(model, IsingModel):
        return np.array(list(itertools.product([-1, 1], repeat=model.n_sites)))
    return np.array(list(itertools.product(range(model.n_states), repeat=model.n_sites)))


@pytest.mark.parametrize("model", [
    IsingModel(n_sites=6, backend="numpy"),
    PottsModel(n_sites=6, n_states=3, backend="numpy"),
])

def test_gauge_zeros_diagonal(model):
    h, J = model.random_params(seed=0)
    if isinstance(model, IsingModel):
        assert np.allclose(np.diag(J), 0.0)
    else:
        idx = np.arange(model.n_sites)
        assert np.allclose(J[idx, idx, :, :], 0.0)

@pytest.mark.parametrize("model", [
    IsingModel(n_sites=6, backend="numpy"),
    PottsModel(n_sites=6, n_states=3, backend="numpy"),
])

def test_gauge_symmetrizes_J(model):
    h, J = model.random_params(seed=0)
    if isinstance(model, IsingModel):
        assert np.allclose(J, J.T)
    else:
        assert np.allclose(J, J.transpose(1, 0, 3, 2))

@pytest.mark.parametrize("model", [
    IsingModel(n_sites=6, backend="numpy"),
    PottsModel(n_sites=6, n_states=3, backend="numpy"),
])

def test_gauge_is_idempotent(model):
    rng = np.random.default_rng(1)
    if isinstance(model, IsingModel):
        h = rng.normal(size=model.n_sites)
        J = rng.normal(size=(model.n_sites, model.n_sites))
    else:
        h = rng.normal(size=(model.n_sites, model.n_states))
        J = rng.normal(size=(model.n_sites, model.n_sites, model.n_states, model.n_states))

    h1, J1 = model.apply_gauge(h, J)
    h2, J2 = model.apply_gauge(h1, J1)

    assert np.allclose(h1, h2)
    assert np.allclose(J1, J2)

@pytest.mark.parametrize("model", [
    IsingModel(n_sites=4, backend="numpy"),
    PottsModel(n_sites=4, n_states=3, backend="numpy"),
])
def test_gauge_preserves_distribution(model):
    rng = np.random.default_rng(2)
    if isinstance(model, IsingModel):
        h = rng.normal(size=model.n_sites)
        J = rng.normal(size=(model.n_sites, model.n_sites))
    else:
        h = rng.normal(size=(model.n_sites, model.n_states))
        J = rng.normal(size=(model.n_sites, model.n_sites, model.n_states, model.n_states))

    h_gauge, J_gauge = model.apply_gauge(h, J)

    states = _all_states(model)
    energy = model.compute_energy(h, J, states)
    energy_gauge = model.compute_energy(h_gauge, J_gauge, states)

    shift = energy - energy_gauge
    assert np.allclose(shift, shift[0], atol=1e-8)

def test_project_to_gauge_does_not_corrupt_h_from_gradient_diagonal():
    model = PottsModel(6, 3, backend="numpy")
    rng = np.random.default_rng(0)
    h = rng.normal(size=(6, 3))
    J = rng.normal(size=(6, 6, 3, 3)) * 0.3
    J = (J + J.transpose(1, 0, 3, 2)) / 2
    idx = np.arange(6); J[idx, idx] = 0.0
    J[idx, idx] = np.eye(3) * 0.5   # simulate a raw update whose diagonal mirrors grad_h, like a real optimizer step would produce
    h_proj, _ = model.project_to_gauge(h, J)
    assert np.allclose(h_proj, h - h.mean(axis=1, keepdims=True))  # no extra shift from J's diagonal

