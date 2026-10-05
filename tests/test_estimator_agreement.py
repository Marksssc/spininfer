import numpy as np
from spininfer.models.ising import IsingModel
from spininfer.models.potts import PottsModel
from spininfer.stats.exact_estimator import ExactEstimator
from spininfer.stats.mcmc_estimator import McmcEstimator


def test_ising_exact_matches_mcmc(backend, to_backend):
    model = IsingModel(n_sites=8, backend=backend)
    h, J = model.random_params(scale_h=0.2, scale_J=0.2, seed=0)

    exact = ExactEstimator().estimate(model, h, J)
    mcmc = McmcEstimator(n_samples=20_000, iterations=8 * 2_000, seed=1).estimate(model, h, J)

    assert np.allclose(to_backend(exact.mean_s), to_backend(mcmc.mean_s), atol=0.02)
    assert np.allclose(to_backend(exact.mean_ss), to_backend(mcmc.mean_ss), atol=0.02)


def test_potts_exact_matches_mcmc(backend, to_numpy):
    model = PottsModel(n_sites=6, n_states=3, backend=backend)
    h, J = model.random_params(scale_h=0.2, scale_J=0.2, seed=0)

    exact = ExactEstimator().estimate(model, h, J)
    mcmc = McmcEstimator(n_samples=20_000, iterations=6 * 2_000, seed=1).estimate(model, h, J)

    assert np.allclose(to_numpy(exact.mean_s), to_numpy(mcmc.mean_s), atol=0.02)
    sites = model.n_sites
    off_diag = ~np.eye(sites, dtype=bool)
    assert np.allclose(to_numpy(exact.mean_ss)[off_diag], to_numpy(mcmc.mean_ss)[off_diag], atol=0.02)