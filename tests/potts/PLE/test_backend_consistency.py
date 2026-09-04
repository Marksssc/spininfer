import numpy as np
import cupy as cp
from objectives import PLE_potts_numpy, PLE_potts_numba, PLE_potts_cupy, PLE_potts_jax
from models.potts import PottsModel

def test_consistency():
    rng = np.random.default_rng(0)
    sites = 10
    states = 3
    potts = PottsModel(sites, states)
    h, J = potts.random_params()
    data = rng.choice([0, 1, 2], size=(500, sites)).astype(np.int64)
    empirical_mean_s = data.mean(axis=0)
    empirical_mean_ss = (data.T @ data) / data.shape[0]

    val_np, hg_np, Jg_np = PLE_potts_numpy.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    val_nb, hg_nb, Jg_nb = PLE_potts_numba.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    val_cp, hg_cp, Jg_cp = PLE_potts_cupy.value_and_gradient(
        cp.asarray(h), cp.asarray(J), cp.asarray(data), empirical_mean_s, empirical_mean_ss
    )
    val_jx, hg_jx, Jg_jx = PLE_potts_jax.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)

    val_cp, hg_cp, Jg_cp = cp.asnumpy(val_cp), cp.asnumpy(hg_cp), cp.asnumpy(Jg_cp)
    val_jx, hg_jx, Jg_jx = np.asarray(val_jx), np.asarray(hg_jx), np.asarray(Jg_jx)

    for val, hg, Jg, name in [
        (val_nb, hg_nb, Jg_nb, "numba"),
        (val_cp, hg_cp, Jg_cp, "cupy"),
        (val_jx, hg_jx, Jg_jx, "jax"),
    ]:
        assert np.isclose(val_np, val), f"value mismatch: numpy vs {name}"
        assert np.allclose(hg_np, hg), f"h gradient mismatch: numpy vs {name}"
        assert np.allclose(Jg_np, Jg), f"J gradient mismatch: numpy vs {name}"