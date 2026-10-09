import importlib

import numpy as np
import pytest
from spininfer.models.ising import IsingModel
from spininfer.objectives import ple_ising_numpy


def test_consistency(backend, to_numpy):
    if backend == "numpy":
        pytest.skip("numpy is the reference backend")

    rng = np.random.default_rng(0)
    sites = 10
    h = rng.normal(scale=0.2, size=sites)
    J = rng.normal(scale=0.3, size=(sites, sites))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    data = rng.choice([-1, 1], size=(500, sites)).astype(np.float64)
    mean_s = data.mean(axis=0)
    mean_ss = (data.T @ data) / data.shape[0]

    val_np, hg_np, Jg_np = ple_ising_numpy.value_and_gradient(h, J, data, mean_s, mean_ss)

    kernel = importlib.import_module(f"spininfer.objectives.ple_ising_{backend}")
    xp = IsingModel(n_sites=2, backend=backend).array_backend.xp
    val, hg, Jg = kernel.value_and_gradient(*map(xp.asarray, (h, J, data, mean_s, mean_ss)))

    rtol, atol = (1e-3, 1e-5) if backend == "jax" else (1e-5, 1e-8)  # jax may use lower-precision matmuls on some devices
    assert np.isclose(val_np, float(val), rtol=rtol, atol=atol), f"value mismatch: numpy vs {backend}"
    assert np.allclose(hg_np, to_numpy(hg), rtol=rtol, atol=atol), f"h gradient mismatch: numpy vs {backend}"
    assert np.allclose(Jg_np, to_numpy(Jg), rtol=rtol, atol=atol), f"J gradient mismatch: numpy vs {backend}"
