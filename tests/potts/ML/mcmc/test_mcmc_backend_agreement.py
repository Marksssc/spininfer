import numpy as np
import pytest

from models.potts import PottsModel
from stats.exact_estimator import ExactEstimator
from stats.mcmc_estimator import McmcEstimator

N_SITES = 5
N_STATES = 3

SCALE_H = 0.3
SCALE_J = 0.3
PARAM_SEED = 0

N_SAMPLES = 4000
ITERATIONS = 4000
MCMC_SEED = 1

MEAN_S_ATOL = 0.07
MEAN_SS_ATOL = 0.07

BACKENDS = ["numpy", "numba", "cupy", "jax"]


def _make_model(backend):
    try:
        return PottsModel(n_sites=N_SITES, n_states=N_STATES, backend=backend)
    except ValueError:
        pytest.skip(f"'{backend}' backend is not available in this environment")


def _to_numpy(x):
    module = type(x).__module__
    if module.startswith("cupy"):
        import cupy as cp
        return cp.asnumpy(x)
    return np.asarray(x)


def _off_diagonal_mask(n_sites):
    return ~np.eye(n_sites, dtype=bool)


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def test_mcmc_matches_exact_statistics(backend):
    model = _make_model(backend)
    h, J = model.random_params(scale_h=SCALE_H, scale_J=SCALE_J, seed=PARAM_SEED)

    exact = ExactEstimator().estimate(model, h, J)
    mcmc = McmcEstimator(
        n_samples=N_SAMPLES, iterations=ITERATIONS, seed=MCMC_SEED
    ).estimate(model, h, J)

    exact_mean_s = _to_numpy(exact.mean_s)
    mcmc_mean_s = _to_numpy(mcmc.mean_s)
    exact_mean_ss = _to_numpy(exact.mean_ss)
    mcmc_mean_ss = _to_numpy(mcmc.mean_ss)


    assert np.allclose(mcmc_mean_s, exact_mean_s, atol=MEAN_S_ATOL), f"MCMC and exact means don't match"
    mask = _off_diagonal_mask(model.n_sites)
    assert np.allclose(mcmc_mean_ss[mask], exact_mean_ss[mask], atol=MEAN_SS_ATOL), f"MCMC and exact correlations don't match"



def test_all_available_backends_agree_with_each_other():
    results = {}
    for name in BACKENDS:
        try:
            model = PottsModel(n_sites=N_SITES, n_states=N_STATES, backend=name)
        except ValueError:
            continue
        h, J = model.random_params(scale_h=SCALE_H, scale_J=SCALE_J, seed=PARAM_SEED)
        mcmc = McmcEstimator(
            n_samples=N_SAMPLES, iterations=ITERATIONS, seed=MCMC_SEED
        ).estimate(model, h, J)
        results[name] = (_to_numpy(mcmc.mean_s), _to_numpy(mcmc.mean_ss))

    if len(results) < 2:
        pytest.skip("fewer than two backends available to cross-check")

    mask = _off_diagonal_mask(N_SITES)
    names = list(results)
    reference = names[0]
    ref_mean_s, ref_mean_ss = results[reference]

    for other in names[1:]:
        other_mean_s, other_mean_ss = results[other]
        assert np.allclose(other_mean_s, ref_mean_s, atol=MEAN_S_ATOL), \
            f"MCMC single-site moments disagree between '{reference}' and '{other}' backends"
        assert np.allclose(other_mean_ss[mask], ref_mean_ss[mask], atol=MEAN_SS_ATOL), \
            f"MCMC pairwise moments disagree between '{reference}' and '{other}' backends"