import numpy as np
import pytest
from spininfer.models.ising import IsingModel
from spininfer.mean_field.ising import naive_mean_field, tap_mean_field, independent_pair_approximation, sessak_monasson_approximation, bethe_approximation


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.1, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.1, 1.0, 0.1, id="moderate_coupling"),
])
def test_naive_mean_field_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    """nMF should recover (h, J) accurately from *exact* moments."""
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    h_err = np.abs(to_numpy(h_mf) - to_numpy(h_true)).mean()
    J_err = np.abs(to_numpy(J_mf) - to_numpy(J_true)).mean()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


def test_naive_mean_field_accuracy_degrades_with_coupling_strength(backend, to_numpy):
    """As coupling strength grows, the mean-field
    approximation's error grows with it."""
    model = IsingModel(n_sites=10, backend=backend)
    errors = []
    for scale_J in (0.05, 1.0, 2.0):
        h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.1, loc_J=0.0,
                                              scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
        mean_s, mean_ss = model.exact_statistics(h_true, J_true)
        h_mf, _ = naive_mean_field(model, mean_s, mean_ss)
        errors.append(np.abs(to_numpy(h_mf) - to_numpy(h_true)).mean())

    assert errors[0] < errors[1] < errors[2], f"expected monotonically increasing error, got {errors}"


def test_naive_mean_field_J_has_zero_diagonal_and_is_symmetric(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_mf = naive_mean_field(model, mean_s, mean_ss)

    assert np.allclose(np.diag(to_numpy(J_mf)), 0.0)
    assert np.allclose(to_numpy(J_mf), to_numpy(J_mf.T))


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.1, id="moderate_coupling"),
])
def test_tap_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    """TAP should recover (h, J) from *exact* moments."""
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_tap, J_tap = tap_mean_field(model, mean_s, mean_ss)

    assert np.all(np.isfinite(to_numpy(h_tap))) and np.all(np.isfinite(to_numpy(J_tap)))
    h_err = np.abs(to_numpy(h_tap - h_true)).mean()
    J_err = np.abs(to_numpy(J_tap - J_true)).mean()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tap_beats_naive_mean_field_at_moderate_coupling(seed, backend, to_numpy):
    """With non-zero magnetisations the Onsager correction should make TAP more accurate than nMF."""
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=1.0 / np.sqrt(model.n_sites), seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h_n, J_n = naive_mean_field(model, mean_s, mean_ss)
    h_t, J_t = tap_mean_field(model, mean_s, mean_ss)

    assert np.abs(to_numpy(J_t - J_true)).mean() < np.abs(to_numpy(J_n - J_true)).mean()
    assert np.abs(to_numpy(h_t - h_true)).mean() < np.abs(to_numpy(h_n - h_true)).mean()


def test_tap_reduces_to_naive_mean_field_at_zero_magnetisation(backend, to_numpy):
    """With m = 0 all corrections vanish: J = -C^-1 and h = 0."""
    model = IsingModel(n_sites=8, backend=backend)
    xp = model.array_backend.xp
    _, J_true = model.random_params(seed=3)
    h_zero = xp.zeros(model.n_sites)
    mean_s, mean_ss = model.exact_statistics(h_zero, J_true)

    h_n, J_n = naive_mean_field(model, mean_s, mean_ss)
    h_t, J_t = tap_mean_field(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(J_t), to_numpy(J_n))
    assert np.allclose(to_numpy(h_t), to_numpy(h_n), atol=1e-6)


def test_tap_J_has_zero_diagonal_and_is_symmetric(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_tap = tap_mean_field(model, mean_s, mean_ss)

    assert np.allclose(np.diag(to_numpy(J_tap)), 0.0)
    assert np.allclose(to_numpy(J_tap), to_numpy(J_tap.T))


def test_tap_satisfies_its_own_self_consistency_equation(backend, to_numpy):
    """Plugging the inferred (h, J) back into the TAP equation must reproduce the input m."""
    model = IsingModel(n_sites=10, backend=backend)
    xp = model.array_backend.xp
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=0.5 / np.sqrt(model.n_sites), seed=2)
    m, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = tap_mean_field(model, m, mean_ss)
    onsager = m * xp.sum(J**2 * (1.0 - m[None, :]**2), axis=1)
    m_tap = xp.tanh(h + J @ m - onsager)

    assert np.allclose(to_numpy(m_tap), to_numpy(m), atol=1e-6)


def test_ipa_independent_spins_gives_zero_coupling_and_single_site_field(backend, to_numpy):
    """With C_ij = 0 for i != j the pair corrections vanish: J = 0 and h = atanh(m)."""
    model = IsingModel(n_sites=6, backend=backend)
    xp = model.array_backend.xp

    m = xp.array([0.3, -0.5, 0.0, 0.7, -0.1, 0.2])
    mean_ss = xp.outer(m, m)
    mean_ss = model.array_backend.zero_diagonal(mean_ss) + xp.eye(model.n_sites)

    h, J = independent_pair_approximation(model, m, mean_ss)

    assert np.allclose(to_numpy(J), 0.0, atol=1e-6)
    assert np.allclose(to_numpy(h), np.arctanh(to_numpy(m)), atol=1e-6)


def test_ipa_is_exact_for_two_sites(backend, to_numpy):
    """For N = 2 the pair inversion is the full inversion, so (h, J) are recovered exactly."""
    model = IsingModel(n_sites=2, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0, scale_J=1.0, seed=4)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h), to_numpy(h_true), atol=1e-8)
    assert np.allclose(to_numpy(J), to_numpy(J_true), atol=1e-8)


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.15, id="moderate_coupling"),
])
def test_ipa_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.all(np.isfinite(h)) and np.all(np.isfinite(J))
    assert np.abs(to_numpy(h - h_true)).mean() < max_err
    assert np.abs(to_numpy(J - J_true)).mean() < max_err


def test_ipa_J_has_zero_diagonal_and_is_symmetric(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.allclose(np.diag(to_numpy(J)), 0.0)
    assert np.allclose(to_numpy(J), to_numpy(J.T))


def test_ipa_stays_finite_when_a_pair_state_is_never_observed(backend, to_numpy):
    """A never-seen pair state makes the true coupling diverge; the clamp must keep it finite."""
    model = IsingModel(n_sites=3, backend=backend)
    xp = model.array_backend.xp

    m = xp.array([0.0, 0.0, 0.0])
    mean_ss = xp.array([[1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])

    h, J = independent_pair_approximation(model, m, mean_ss)

    assert np.all(np.isfinite(to_numpy(h))) and np.all(np.isfinite(to_numpy(J)))
    assert J[0, 1] > 1.0


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.1, id="moderate_coupling"),
])
def test_sessak_monasson_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = sessak_monasson_approximation(model, mean_s, mean_ss)

    assert np.all(np.isfinite(to_numpy(h))) and np.all(np.isfinite(to_numpy(J)))
    assert np.abs(to_numpy(h - h_true)).mean() < max_err
    assert np.abs(to_numpy(J - J_true)).mean() < max_err


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sessak_monasson_J_beats_naive_and_ipa_at_moderate_coupling(seed, backend, to_numpy):
    """The second-order corrections should make J more accurate than nMF or IPA alone."""
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=1.0 / np.sqrt(model.n_sites), seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_sm = sessak_monasson_approximation(model, mean_s, mean_ss)
    _, J_n = naive_mean_field(model, mean_s, mean_ss)
    _, J_ipa = independent_pair_approximation(model, mean_s, mean_ss)

    err_sm = np.abs(to_numpy(J_sm - J_true)).mean()
    assert err_sm < np.abs(to_numpy(J_n - J_true)).mean()
    assert err_sm < np.abs(to_numpy(J_ipa - J_true)).mean()


def test_sessak_monasson_J_has_zero_diagonal_and_is_symmetric(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J = sessak_monasson_approximation(model, mean_s, mean_ss)

    assert np.allclose(np.diag(to_numpy(J)), 0.0)
    assert np.allclose(to_numpy(J), to_numpy(J.T))


def test_sessak_monasson_independent_spins_gives_zero_coupling(backend, to_numpy):
    """With C_ij = 0 for i != j, all coupling terms vanish and h = atanh(m)."""
    model = IsingModel(n_sites=6, backend=backend)
    xp = model.array_backend.xp

    m = xp.array([0.3, -0.5, 0.1, 0.7, -0.1, 0.2])
    mean_ss = xp.outer(m, m)
    mean_ss = model.array_backend.zero_diagonal(mean_ss) + xp.eye(model.n_sites)

    h, J = sessak_monasson_approximation(model, m, mean_ss)

    assert np.allclose(to_numpy(J), 0.0, atol=1e-6)
    assert np.allclose(to_numpy(h), np.arctanh(to_numpy(m)), atol=1e-6)


def test_sessak_monasson_no_warnings_on_diagonal(backend):
    """The diagonal denominator is masked, so no divide-by-zero warnings should be raised."""
    import warnings
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sessak_monasson_approximation(model, mean_s, mean_ss)


def _chain_params(n_sites, coupling, fields, xp):
    """Nearest-neighbour chain: a tree, so the Bethe approximation is exact on it."""
    J = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        J[i, i + 1] = J[i + 1, i] = coupling
    return xp.asarray(fields, dtype=float), xp.asarray(J)


@pytest.mark.parametrize("coupling", [0.3, 0.8, 1.5])
def test_bethe_is_exact_on_a_tree(coupling, backend, to_numpy):
    """On a chain with exact moments Bethe recovers (h, J) exactly, including zero couplings
    between non-neighbours, even at strong coupling where the other methods fail."""
    model = IsingModel(n_sites=8, backend=backend)
    xp = model.array_backend.xp

    h_true, J_true = _chain_params(8, coupling, xp.linspace(-0.6, 0.6, 8), xp)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = bethe_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(J), to_numpy(J_true), atol=1e-6)
    assert np.allclose(to_numpy(h), to_numpy(h_true), atol=1e-6)


def test_bethe_beats_other_methods_on_a_strongly_coupled_tree(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    xp = model.array_backend.xp

    h_true, J_true = _chain_params(8, 1.0, xp.linspace(-0.6, 0.6, 8), xp)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    err_bethe = np.abs(to_numpy(bethe_approximation(model, mean_s, mean_ss)[1]) - to_numpy(J_true)).mean()
    for method in (naive_mean_field, tap_mean_field, independent_pair_approximation,
                   sessak_monasson_approximation):
        err = np.abs(to_numpy(method(model, mean_s, mean_ss)[1]) - to_numpy(J_true)).mean()
        assert err_bethe < err, f"{method.__name__} error {err} not above Bethe {err_bethe}"


def test_bethe_is_exact_for_two_sites(backend, to_numpy):
    model = IsingModel(n_sites=2, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0, scale_J=1.0, seed=4)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = bethe_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h), to_numpy(h_true), atol=1e-8)
    assert np.allclose(to_numpy(J), to_numpy(J_true), atol=1e-8)


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.15, id="moderate_coupling"),
])
def test_bethe_matches_exact_moments_on_dense_model(scale_h, scale_J, max_err, backend, to_numpy):
    model = IsingModel(n_sites=10, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = bethe_approximation(model, mean_s, mean_ss)

    assert np.all(np.isfinite(to_numpy(h))) and np.all(np.isfinite(to_numpy(J)))
    assert np.abs(to_numpy(h - h_true)).mean() < max_err
    assert np.abs(to_numpy(J - J_true)).mean() < max_err


def test_bethe_reduces_to_naive_mean_field_for_small_correlations(backend,to_numpy):
    """For tiny couplings, C_tilde ~ -x*a, so J ~ -(C^-1)_ij, the nMF result."""
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.3, loc_J=0.0, scale_J=0.002, seed=2)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_bethe = bethe_approximation(model, mean_s, mean_ss)
    _, J_nmf = naive_mean_field(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(J_bethe), to_numpy(J_nmf), atol=1e-4)


def test_bethe_J_has_zero_diagonal_and_is_symmetric(backend, to_numpy):
    model = IsingModel(n_sites=8, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J = bethe_approximation(model, mean_s, mean_ss)

    assert np.allclose(np.diag(to_numpy(J)), 0.0)
    assert np.allclose(to_numpy(J), to_numpy(J.T))


def test_bethe_independent_spins_gives_zero_coupling(backend, to_numpy):
    model = IsingModel(n_sites=6, backend=backend)
    xp = model.array_backend.xp

    m = xp.array([0.3, -0.5, 0.1, 0.7, -0.1, 0.2])
    mean_ss = xp.outer(m, m)
    mean_ss = model.array_backend.zero_diagonal(mean_ss) + xp.eye(model.n_sites)

    h, J = bethe_approximation(model, m, mean_ss)

    assert np.allclose(to_numpy(J), 0.0, atol=1e-6)
    assert np.allclose(to_numpy(h), np.arctanh(to_numpy(m)), atol=1e-6)
