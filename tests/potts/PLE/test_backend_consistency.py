import importlib

import numpy as np
import pytest
from spininfer.models.potts import PottsModel
from spininfer.objectives import ple_potts_numpy


def test_consistency(backend, to_numpy):
    if backend == "numpy":
        pytest.skip("numpy is the reference backend")

    rng = np.random.default_rng(0)
    sites, states = 10, 3
    h, J = PottsModel(sites, states, backend="numpy").random_params()
    data = rng.choice(np.arange(states), size=(500, sites)).astype(np.int64)
    mean_s = data.mean(axis=0)
    mean_ss = (data.T @ data) / data.shape[0]

    val_np, hg_np, Jg_np = ple_potts_numpy.value_and_gradient(h, J, data, mean_s, mean_ss)

    kernel = importlib.import_module(f"spininfer.objectives.ple_potts_{backend}")
    xp = PottsModel(sites, states, backend=backend).array_backend.xp
    val, hg, Jg = kernel.value_and_gradient(*map(xp.asarray, (h, J, data, mean_s, mean_ss)))

    rtol, atol = (1e-3, 1e-5) if backend == "jax" else (1e-5, 1e-8)  # jax may use lower-precision matmuls on some devices
    assert np.isclose(val_np, float(val), rtol=rtol, atol=atol), f"value mismatch: numpy vs {backend}"
    assert np.allclose(hg_np, to_numpy(hg), rtol=rtol, atol=atol), f"h gradient mismatch: numpy vs {backend}"
    assert np.allclose(Jg_np, to_numpy(Jg), rtol=rtol, atol=atol), f"J gradient mismatch: numpy vs {backend}"
