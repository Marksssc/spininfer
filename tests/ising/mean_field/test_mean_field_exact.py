import numpy as np
import pytest
from models.ising import IsingModel
from mean_field.ising import naive_mean_field, TAP_mean_field


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.1, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.1, 1.0, 0.1, id="moderate_coupling"),
])
def test_naive_mean_field_matches_exact_moments(scale_h, scale_J, max_err):
    """nMF should recover (h, J) accurately from *exact* moments."""
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


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.1, id="moderate_coupling"),
])
def test_tap_matches_exact_moments(scale_h, scale_J, max_err):
    """TAP should recover (h, J) from *exact* moments."""
    model = IsingModel(n_sites=10, backend="numpy")
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_tap, J_tap = TAP_mean_field(model, mean_s, mean_ss)

    assert np.all(np.isfinite(h_tap)) and np.all(np.isfinite(J_tap))
    h_err = np.abs(h_tap - h_true).mean()
    J_err = np.abs(J_tap - J_true).mean()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tap_beats_naive_mean_field_at_moderate_coupling(seed):
    """With non-zero magnetisations the Onsager correction should make TAP more accurate than nMF."""
    model = IsingModel(n_sites=10, backend="numpy")
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=1.0 / np.sqrt(model.n_sites), seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_n, J_n = naive_mean_field(model, mean_s, mean_ss)
    h_t, J_t = TAP_mean_field(model, mean_s, mean_ss)

    assert np.abs(J_t - J_true).mean() < np.abs(J_n - J_true).mean()
    assert np.abs(h_t - h_true).mean() < np.abs(h_n - h_true).mean()


def test_tap_reduces_to_naive_mean_field_at_zero_magnetisation():
    """With m = 0 all corrections vanish: J = -C^-1 and h = 0."""
    model = IsingModel(n_sites=8, backend="numpy")
    _, J_true = model.random_params(seed=3)
    h_zero = np.zeros(model.n_sites)
    mean_s, mean_ss = model.exact_statistics(h_zero, J_true)

    h_n, J_n = naive_mean_field(model, mean_s, mean_ss)
    h_t, J_t = TAP_mean_field(model, mean_s, mean_ss)

    assert np.allclose(J_t, J_n)
    assert np.allclose(h_t, h_n, atol=1e-8)


def test_tap_J_has_zero_diagonal_and_is_symmetric():
    model = IsingModel(n_sites=8, backend="numpy")
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_tap = TAP_mean_field(model, mean_s, mean_ss)

    assert np.allclose(np.diag(J_tap), 0.0)
    assert np.allclose(J_tap, J_tap.T)


def test_tap_satisfies_its_own_self_consistency_equation():
    """Plugging the inferred (h, J) back into the TAP equation must reproduce the input m."""
    model = IsingModel(n_sites=10, backend="numpy")
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=0.5 / np.sqrt(model.n_sites), seed=2)
    m, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = TAP_mean_field(model, m, mean_ss)
    onsager = m * np.sum(J**2 * (1.0 - m[None, :]**2), axis=1)
    m_tap = np.tanh(h + J @ m - onsager)

    assert np.allclose(m_tap, m, atol=1e-6)
