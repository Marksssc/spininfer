import numpy as np
from objectives import PLE_ising_numpy, PLE_ising_numba

def test_consistency():
    rng = np.random.default_rng(0)
    sites = 10
    h = rng.normal(scale=0.2, size=sites)
    J = rng.normal(scale=0.3, size=(sites, sites))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    data = rng.choice([-1, 1], size=(500, sites)).astype(np.float64)
    empirical_mean_s = data.mean(axis=0)
    empirical_mean_ss = (data.T @ data) / data.shape[0]

    val_np, hg_np, Jg_np = PLE_ising_numpy.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    val_nb, hg_nb, Jg_nb = PLE_ising_numba.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)

    assert np.isclose(val_np, val_nb)
    assert np.allclose(hg_np, hg_nb)
    assert np.allclose(Jg_np, Jg_nb)