import numpy as np
import cupy as cp
from objectives import PLE_ising_numpy, PLE_ising_numba, PLE_ising_cupy, PLE_ising_jax

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
    val_cp, hg_cp, Jg_cp = PLE_ising_cupy.value_and_gradient(
        cp.asarray(h), cp.asarray(J), cp.asarray(data),
        cp.asarray(empirical_mean_s), cp.asarray(empirical_mean_ss)
    )
    val_jx, hg_jx, Jg_jx = PLE_ising_jax.value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)

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