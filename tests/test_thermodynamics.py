import numpy as np
import pytest

from spininfer.models.ising import IsingModel
from spininfer.models.potts import PottsModel
from spininfer.stats.exact_thermodynamics_estimator import ExactThermodynamicsEstimator
from spininfer.stats.thermodynamics_integration_estimator import ThermodynamicsIntegrationEstimator


def test_ising_zero_field_zero_coupling_matches_closed_form(backend):
    model = IsingModel(n_sites=5, backend=backend)
    xp = model.array_backend.xp
    h, J = xp.zeros(5), xp.zeros((5, 5))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.energy, 0.0, atol=1e-8)
    assert np.isclose(t.heat_capacity, 0.0, atol=1e-8)
    assert np.isclose(t.entropy, 5 * xp.log(2), atol=1e-8)
    assert np.isclose(t.free_energy, -5 * xp.log(2), atol=1e-8)


def test_potts_zero_field_zero_coupling_matches_closed_form(backend):
    model = PottsModel(n_sites=4, n_states=3, backend=backend)
    xp = model.array_backend.xp
    h, J = xp.zeros((4, 3)), xp.zeros((4, 4, 3, 3))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.energy, 0.0, atol=1e-8)
    assert np.isclose(t.heat_capacity, 0.0, atol=1e-8)
    assert np.isclose(t.entropy, 4 * xp.log(3), atol=1e-8)
    assert np.isclose(t.free_energy, -4 * xp.log(3), atol=1e-8)


def test_ising_free_energy_matches_reference_when_uncoupled(backend):
    model = IsingModel(n_sites=5, backend=backend)
    xp = model.array_backend.xp
    h, _ = model.random_params(scale_h=0.5, seed=0)
    J = xp.asarray(np.zeros((5, 5)))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.free_energy, model.reference_free_energy(h), atol=1e-8)


def test_potts_free_energy_matches_reference_when_uncoupled(backend):
    model = PottsModel(n_sites=4, n_states=3, backend=backend)
    xp = model.array_backend.xp

    h, _ = model.random_params(scale_h=0.5, seed=0)
    J = xp.zeros((4, 4, 3, 3))

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert np.isclose(t.free_energy, model.reference_free_energy(h), atol=1e-8)

def test_exact_thermodynamics_numpy_backend_agree_ising(backend):
    ising_numpy = IsingModel(n_sites=6, backend="numpy")
    ising_backend = IsingModel(n_sites=6, backend=backend)
    xp = ising_backend.array_backend.xp
    
    h, J = ising_numpy.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    est = ExactThermodynamicsEstimator()
    t_np = est.estimate(ising_numpy, h, J)
    t_nb = est.estimate(ising_backend, xp.asarray(h), xp.asarray(J))

    assert np.isclose(t_np.entropy, t_nb.entropy, atol=1e-6)
    assert np.isclose(t_np.energy, t_nb.energy, atol=1e-6)
    assert np.isclose(t_np.free_energy, t_nb.free_energy, atol=1e-6)
    assert np.isclose(t_np.heat_capacity, t_nb.heat_capacity, atol=1e-6)

def test_exact_thermodynamics_numpy_backend_agree_potts(backend):
    ising_numpy = PottsModel(n_sites=6, n_states=3, backend="numpy")
    ising_backend = PottsModel(n_sites=6, n_states=3, backend=backend)
    xp = ising_backend.array_backend.xp
    
    h, J = ising_numpy.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    est = ExactThermodynamicsEstimator()
    t_np = est.estimate(ising_numpy, h, J)
    t_nb = est.estimate(ising_backend, xp.asarray(h), xp.asarray(J))

    assert np.isclose(t_np.entropy, t_nb.entropy, atol=1e-6)
    assert np.isclose(t_np.energy, t_nb.energy, atol=1e-6)
    assert np.isclose(t_np.free_energy, t_nb.free_energy, atol=1e-6)
    assert np.isclose(t_np.heat_capacity, t_nb.heat_capacity, atol=1e-6)

def test_exact_thermodynamics_bounds_ising(backend):
    model = IsingModel(n_sites=5, backend=backend)
    h, J = model.random_params(scale_h=0.4, scale_J=0.4, seed=3)

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert t.heat_capacity >= -1e-8   # heat capacity is a variance, must be non-negative
    assert t.entropy >= -1e-8         # Shannon entropy of a discrete distribution is non-negative
    assert t.free_energy <= t.energy + 1e-8   # F = E - S and S >= 0

def test_exact_thermodynamics_bounds_potts(backend):
    model = PottsModel(n_sites=4, n_states=3, backend=backend)
    h, J = model.random_params(scale_h=0.4, scale_J=0.4, seed=3)

    t = ExactThermodynamicsEstimator().estimate(model, h, J)

    assert t.heat_capacity >= -1e-8   # heat capacity is a variance, must be non-negative
    assert t.entropy >= -1e-8         # Shannon entropy of a discrete distribution is non-negative
    assert t.free_energy <= t.energy + 1e-8   # F = E - S and S >= 0

def test_thermodynamic_integration_matches_exact_ising(backend):
    model = IsingModel(n_sites=6, backend=backend)
    h, J = model.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    exact = ExactThermodynamicsEstimator().estimate(model, h, J)
    ti = ThermodynamicsIntegrationEstimator(
        n_samples=4000, iterations=model.n_sites * 1000, seed=2, n_points=8
    ).estimate(model, h, J)

    assert np.isclose(exact.entropy, ti.entropy, atol=0.15)
    assert np.isclose(exact.energy, ti.energy, atol=0.15)
    assert np.isclose(exact.free_energy, ti.free_energy, atol=0.1)
    assert np.isclose(exact.heat_capacity, ti.heat_capacity, atol=0.2)


def test_thermodynamic_integration_matches_exact_potts(backend):
    model = PottsModel(n_sites=4, n_states=3, backend=backend)
    h, J = model.random_params(scale_h=0.3, scale_J=0.3, seed=1)

    exact = ExactThermodynamicsEstimator().estimate(model, h, J)
    ti = ThermodynamicsIntegrationEstimator(
        n_samples=4000, iterations=model.n_sites * 1000, seed=2, n_points=8
    ).estimate(model, h, J)

    assert np.isclose(exact.entropy, ti.entropy, atol=0.15)
    assert np.isclose(exact.energy, ti.energy, atol=0.15)
    assert np.isclose(exact.free_energy, ti.free_energy, atol=0.1)
    assert np.isclose(exact.heat_capacity, ti.heat_capacity, atol=0.2)
