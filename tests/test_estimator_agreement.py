import numpy as np
from models.ising import IsingModel
from models.potts import PottsModel
from stats.exact_estimator import ExactEstimator
from stats.mcmc_estimator import McmcEstimator


def test_ising_exact_matches_mcmc():
    model = IsingModel(n_sites=8, backend="numba")
    h, J = model.random_params(scale_h=0.2, scale_J=0.2, seed=0)

    exact = ExactEstimator().estimate(model, h, J)
    mcmc = McmcEstimator(n_samples=20_000, iterations=8 * 2_000, seed=1).estimate(model, h, J)

    assert np.allclose(exact.mean_s, mcmc.mean_s, atol=0.02)
    assert np.allclose(exact.mean_ss, mcmc.mean_ss, atol=0.02)


def test_potts_exact_matches_mcmc():
    model = PottsModel(n_sites=6, n_states=3, backend="numba")
    h, J = model.random_params(scale_h=0.2, scale_J=0.2, seed=0)

    exact = ExactEstimator().estimate(model, h, J)
    mcmc = McmcEstimator(n_samples=20_000, iterations=6 * 2_000, seed=1).estimate(model, h, J)

    assert np.allclose(exact.mean_s, mcmc.mean_s, atol=0.02)
    sites = model.n_sites
    off_diag = ~np.eye(sites, dtype=bool)
    assert np.allclose(exact.mean_ss[off_diag], mcmc.mean_ss[off_diag], atol=0.02)