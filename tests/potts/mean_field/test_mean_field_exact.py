import numpy as np
import pytest
from models.ising import IsingModel
from models.potts import PottsModel
from mean_field.ising import naive_mean_field as ising_naive_mean_field
from mean_field.potts import naive_mean_field


def _exact_moments(scale_J, n_sites=4, n_states=3, seed=1):
    model = PottsModel(n_sites=n_sites, n_states=n_states, backend="numpy")
    h_true, J_true = model.random_params(scale_h=0.3, scale_J=scale_J, seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    return model, h_true, J_true, mean_s, mean_ss


@pytest.mark.parametrize("scale_J, max_err", [
    pytest.param(0.05, 1e-3, id="weak_coupling"),
    pytest.param(0.3, 0.05, id="moderate_coupling"),
])
def test_naive_mean_field_matches_exact_moments(scale_J, max_err):
    """nMF should recover (h, J) accurately from moments at weak coupling."""
    model, h_true, J_true, mean_s, mean_ss = _exact_moments(scale_J)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    h_err = np.abs(h_mf - h_true).max()
    J_err = np.abs(J_mf - J_true).max()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


def test_naive_mean_field_accuracy_degrades_with_coupling_strength():
    """As coupling strength grows, the mean-field approximation's error grows with it."""
    errors = []
    for scale_J in (0.05, 0.3, 0.6):
        model, _, J_true, mean_s, mean_ss = _exact_moments(scale_J)
        _, J_mf = naive_mean_field(model, mean_s, mean_ss)
        errors.append(np.abs(J_mf - J_true).max())

    assert errors[0] < errors[1] < errors[2], f"expected monotonically increasing error, got {errors}"


def test_naive_mean_field_is_exact_for_independent_sites():
    """With J = 0 the sites are independent, so nMF gives J = 0 and h = gauge-fixed log P."""
    model, h_true, _, mean_s, mean_ss = _exact_moments(scale_J=0.0)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    log_p = np.log(mean_s)
    assert np.allclose(J_mf, 0.0, atol=1e-10)
    assert np.allclose(h_mf, log_p - log_p.mean(axis=1, keepdims=True), atol=1e-10)
    assert np.allclose(h_mf, h_true, atol=1e-10)


def test_naive_mean_field_output_is_in_zero_sum_gauge():
    """The returned (h, J) satisfy the model's gauge: zero-sum over states, symmetric, no self-couplings."""
    model, _, _, mean_s, mean_ss = _exact_moments(scale_J=0.3)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    assert np.allclose(h_mf.sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(J_mf.sum(axis=2), 0.0, atol=1e-10)
    assert np.allclose(J_mf.sum(axis=3), 0.0, atol=1e-10)
    assert np.allclose(J_mf, J_mf.transpose(1, 0, 3, 2), atol=1e-10)
    idx = np.arange(model.n_sites)
    assert np.allclose(J_mf[idx, idx], 0.0)


def test_two_state_potts_matches_ising_naive_mean_field():
    """For q = 2, Potts nMF on one-hot moments must equal Ising nMF on the matching ±1 moments."""
    ising = IsingModel(n_sites=5, backend="numpy")
    h_true, J_true = ising.random_params(scale_h=0.3, scale_J=0.2, seed=2)
    m, mss = ising.exact_statistics(h_true, J_true)
    h_ising, J_ising = ising_naive_mean_field(ising, m, mss)

    sigma = np.array([1.0, -1.0])
    mean_s = (1 + m[:, None] * sigma) / 2
    mean_ss = (1 + m[:, None, None, None] * sigma[None, None, :, None]
               + m[None, :, None, None] * sigma[None, None, None, :]
               + mss[:, :, None, None] * sigma[None, None, :, None] * sigma[None, None, None, :]) / 4
    idx = np.arange(ising.n_sites)
    mean_ss[idx, idx] = mean_s[:, :, None] * np.eye(2)

    potts = PottsModel(n_sites=5, n_states=2, backend="numpy")
    h_potts, J_potts = naive_mean_field(potts, mean_s, mean_ss)

    assert np.allclose(h_potts[:, 0], h_ising, atol=1e-8)
    assert np.allclose(J_potts[:, :, 0, 0], J_ising, atol=1e-8)
