import numpy as np
import pytest
from spininfer.models.ising import IsingModel
from spininfer.models.potts import PottsModel
from spininfer.data.generate import generate_data
from spininfer.data.dataset import Dataset
from spininfer.objectives.PLE_ising import PleIsingObjective
from spininfer.objectives.PLE_potts import PlePottsObjective
from spininfer.objectives.moment_matching import MomentMatchingObjective
from spininfer.stats.mcmc_estimator import McmcEstimator
from spininfer.fitters.lbfgs_fitter import LbfgsFitter


def test_flatten_unflatten_roundtrip():
    from spininfer.fitters.lbfgs_fitter import LbfgsFitter
    rng = np.random.default_rng(0)
    h = rng.normal(size=8)
    J = rng.normal(size=(8, 8))

    x = LbfgsFitter.flatten(np, h, J)
    h_back, J_back = LbfgsFitter.unflatten(np, x, h.shape, J.shape)

    assert np.array_equal(h_back, h)
    assert np.array_equal(J_back, J)

def test_lbfgs_recovers_parameters_ising():
    model = IsingModel(n_sites=15, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
    dataset = Dataset(samples=truth.samples, model=model)

    fitter = LbfgsFitter(model=model, dataset=dataset, objective=PleIsingObjective(), maxiter=500)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    assert np.abs(h - truth.h).mean() < 0.05
    assert np.abs(J - truth.J).mean() < 0.05

def test_lbfgs_recovers_parameters_potts():
    model = PottsModel(n_sites=15, n_states=3, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
    dataset = Dataset(samples=truth.samples, model=model)

    fitter = LbfgsFitter(model=model, dataset=dataset, objective=PlePottsObjective(), maxiter=500)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    assert np.abs(h - truth.h).mean() < 0.05
    assert np.abs(J - truth.J).mean() < 0.05

def test_lbfgs_moves_the_cost_in_the_right_direction():
    model = IsingModel(n_sites=15, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
    dataset = Dataset(samples=truth.samples, model=model)
    objective = PleIsingObjective()

    h_init, J_init = model.random_params(seed=2)
    value_before, _ = objective.compute_value_and_gradient(model, h_init, J_init, dataset)

    fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective, maxiter=500)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J
    value_after, _ = objective.compute_value_and_gradient(model, h, J, dataset)

    assert value_after > value_before

def test_lbfgs_rejects_objective_without_value():
    model = IsingModel(n_sites=8, backend="numpy")
    h, J = model.random_params(seed=0)
    truth = generate_data(model, n_samples=500, iterations=200, seed=1)
    dataset = Dataset(samples=truth.samples, model=model)

    objective = MomentMatchingObjective(estimator=McmcEstimator(n_samples=500))
    fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective, maxiter=5)

    with pytest.raises(NotImplementedError):
        fitter.fit(h, J)