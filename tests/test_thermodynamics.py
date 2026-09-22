import numpy as np
import pytest

from models.ising import IsingModel
from models.potts import PottsModel
from stats.exact_thermodynamics_estimator import ExactThermodynamicsEstimator
from stats.thermodynamics_integration_estimator import ThermodynamicsIntegrationEstimator


def test_ising_zero_field_zero_coupling_matches_closed_form():
    model = IsingModel(n_sites=5, backend="numpy")
    h, J = np.zeros(5), np.zeros((5, 5))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.energy, 0.0, atol=1e-8)
    assert np.isclose(t.heat_capacity, 0.0, atol=1e-8)
    assert np.isclose(t.entropy, 5 * np.log(2), atol=1e-8)
    assert np.isclose(t.free_energy, -5 * np.log(2), atol=1e-8)


def test_potts_zero_field_zero_coupling_matches_closed_form():
    model = PottsModel(n_sites=4, n_states=3, backend="numpy")
    h, J = np.zeros((4, 3)), np.zeros((4, 4, 3, 3))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.energy, 0.0, atol=1e-8)
    assert np.isclose(t.heat_capacity, 0.0, atol=1e-8)
    assert np.isclose(t.entropy, 4 * np.log(3), atol=1e-8)
    assert np.isclose(t.free_energy, -4 * np.log(3), atol=1e-8)


def test_ising_free_energy_matches_reference_when_uncoupled():
    model = IsingModel(n_sites=5, backend="numpy")
    h, _ = model.random_params(scale_h=0.5, seed=0)
    J = np.zeros((5, 5))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.free_energy, model.reference_free_energy(h), atol=1e-8)


def test_potts_free_energy_matches_reference_when_uncoupled():
    model = PottsModel(n_sites=4, n_states=3, backend="numpy")
    h, _ = model.random_params(scale_h=0.5, seed=0)
    J = np.zeros((4, 4, 3, 3))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.free_energy, model.reference_free_energy(h), atol=1e-8)


@pytest.mark.parametrize("model_np, model_nb", [
    (IsingModel(n_sites=6, backend="numpy"), IsingModel(n_sites=6, backend="numba")),
    (PottsModel(n_sites=4, n_states=3, backend="numpy"), PottsModel(n_sites=4, n_states=3, backend="numba")),
])
def test_exact_thermodynamics_numpy_numba_agree(model_np, model_nb):
    h, J = model_nb.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    est = ExactThermodynamicsEstimator()
    t_np = est.estimate(model_np, h, J)
    t_nb = est.estimate(model_nb, h, J)

    assert np.isclose(t_np.entropy, t_nb.entropy, atol=1e-6)
    assert np.isclose(t_np.energy, t_nb.energy, atol=1e-6)
    assert np.isclose(t_np.free_energy, t_nb.free_energy, atol=1e-6)
    assert np.isclose(t_np.heat_capacity, t_nb.heat_capacity, atol=1e-6)


@pytest.mark.parametrize("model", [
    IsingModel(n_sites=5, backend="numpy"),
    PottsModel(n_sites=4, n_states=3, backend="numpy"),
])
def test_exact_thermodynamics_bounds(model):
    h, J = model.random_params(scale_h=0.4, scale_J=0.4, seed=3)

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert t.heat_capacity >= -1e-8   # heat capacity is a variance, must be non-negative
    assert t.entropy >= -1e-8         # Shannon entropy of a discrete distribution is non-negative
    assert t.free_energy <= t.energy + 1e-8   # F = E - S and S >= 0


@pytest.mark.parametrize("model", [
    IsingModel(n_sites=6, backend="numba"),
    PottsModel(n_sites=4, n_states=3, backend="numba"),
])
def test_thermodynamic_integration_matches_exact(model):
    h, J = model.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    exact = ExactThermodynamicsEstimator().estimate(model, h, J)
    ti = ThermodynamicsIntegrationEstimator(
        n_samples=4000, iterations=model.n_sites * 1000, seed=2, n_points=8
    ).estimate(model, h, J)

    assert np.isclose(exact.entropy, ti.entropy, atol=0.15)
    assert np.isclose(exact.energy, ti.energy, atol=0.15)
    assert np.isclose(exact.free_energy, ti.free_energy, atol=0.1)
    assert np.isclose(exact.heat_capacity, ti.heat_capacity, atol=0.2)
