import importlib.util
import itertools
import numpy as np
import pytest
from scipy.special import logsumexp

from spininfer.models.ising import IsingModel
from spininfer.data.generate import generate_data
from spininfer.data.dataset import Dataset
from spininfer.stats.moments import Moments
from spininfer.objectives.moment_matching import MomentMatchingObjective
from spininfer.stats.exact_estimator import ExactEstimator
from spininfer.fitters.lbfgs_fitter import LbfgsFitter


def _setup(n_sites=6, seed=0):
    model = IsingModel(n_sites=n_sites, backend="numpy")
    truth = generate_data(model, n_samples=2000, iterations=200, seed=seed)
    dataset = Dataset(samples=truth.samples, model=model)
    h, J = model.random_params(seed=seed + 1)
    return model, dataset, h, J


def test_value_matches_mean_log_probability():
    model, dataset, h, J = _setup()
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=model.n_sites)))
    log_Z = logsumexp(-model.compute_energy(h, J, states))
    expected = np.mean(-model.compute_energy(h, J, dataset.samples)) - log_Z

    value = MomentMatchingObjective(ExactEstimator()).compute_value(model, h, J, dataset)
    assert np.isclose(value, expected, rtol=1e-10, atol=1e-10)


def test_gradient_matches_finite_differences():
    model, dataset, h, J = _setup()
    objective = MomentMatchingObjective(ExactEstimator())
    grad = objective.compute_gradient(model, h, J, dataset)
    eps = 1e-6

    for i in range(model.n_sites):
        dh = np.zeros_like(h); dh[i] = eps
        fd = (objective.compute_value(model, h + dh, J, dataset)
              - objective.compute_value(model, h - dh, J, dataset)) / (2 * eps)
        assert np.isclose(fd, grad.grad_h[i], atol=1e-6)

    for i, j in itertools.combinations(range(model.n_sites), 2):
        dJ = np.zeros_like(J); dJ[i, j] = dJ[j, i] = eps 
        fd = (objective.compute_value(model, h, J + dJ, dataset)
              - objective.compute_value(model, h, J - dJ, dataset)) / (2 * eps)
        assert np.isclose(fd, grad.grad_J[i, j] + grad.grad_J[j, i], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax not installed"))])
def test_lbfgs_recovers_parameters_from_exact_moments(backend):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    dataset = Dataset(model=model, moments=Moments(mean_s=mean_s, mean_ss=mean_ss))

    fitter = LbfgsFitter(model=model, objective=MomentMatchingObjective(ExactEstimator()),
                         dataset=dataset, tol=1e-10, maxiter=1000)
    result = fitter.fit(*model.random_params(seed=4))

    assert np.allclose(np.asarray(result.h), np.asarray(h_true), atol=1e-4)
    assert np.allclose(np.asarray(result.J), np.asarray(J_true), atol=1e-4)
