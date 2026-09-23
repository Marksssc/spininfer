import numpy as np
import pytest
from models.ising import IsingModel
from mean_field.ising import naive_mean_field


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.1, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.1, 1.0, 0.1, id="moderate_coupling"),
])
def test_naive_mean_field_matches_exact_moments(scale_h, scale_J, max_err):
    """nMF should recover (h, J) accurately from *exact* moments, isolating the inversion
    formula itself from any MCMC sampling noise."""
    model = IsingModel(n_sites=10, backend="numpy")
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    h_err = np.abs(h_mf - h_true).mean()
    J_err = np.abs(J_mf - J_true).mean()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


def test_naive_mean_field_accuracy_degrades_with_coupling_strength():
    """As coupling strength grows, the mean-field
    approximation's error grows with it."""
    model = IsingModel(n_sites=10, backend="numpy")
    errors = []
    for scale_J in (0.05, 1.0, 2.0):
        h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.1, loc_J=0.0,
                                              scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
        mean_s, mean_ss = model.exact_statistics(h_true, J_true)
        h_mf, _ = naive_mean_field(model, mean_s, mean_ss)
        errors.append(np.abs(h_mf - h_true).mean())

    assert errors[0] < errors[1] < errors[2], f"expected monotonically increasing error, got {errors}"


def test_naive_mean_field_J_has_zero_diagonal_and_is_symmetric():
    model = IsingModel(n_sites=8, backend="numpy")
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_mf = naive_mean_field(model, mean_s, mean_ss)

    assert np.allclose(np.diag(J_mf), 0.0)
    assert np.allclose(J_mf, J_mf.T)
