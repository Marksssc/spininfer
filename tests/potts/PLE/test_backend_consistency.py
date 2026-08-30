import numpy as np
from objectives import PLE_potts_numpy, PLE_potts_numba
from models.potts import PottsModel

def test_consistency():
    rng = np.random.default_rng(0)
    sites = 10
    states = 3
    potts = PottsModel(sites, states)
    h, J = potts.random_params()
    data = rng.choice([0, 1, 2], size=(500, sites)).astype(np.float64)
    empirical_mean_s = data.mean(axis=0)
    empirical_mean_ss = (data.T @ data) / data.shape[0]

    val_np, hg_np, Jg_np = PLE_potts_numpy.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    val_nb, hg_nb, Jg_nb = PLE_potts_numba.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)

    assert np.isclose(val_np, val_nb)
    assert np.allclose(hg_np, hg_nb)
    assert np.allclose(Jg_np, Jg_nb)