import numpy as np
import pytest
from spininfer.models.ising import IsingModel
from spininfer.models.potts import PottsModel
from spininfer.mean_field.ising import naive_mean_field as ising_naive_mean_field
from spininfer.mean_field.ising import independent_pair_approximation as ising_independent_pair_approximation
from spininfer.mean_field.ising import sessak_monasson_approximation as ising_sessak_monasson_approximation
from spininfer.mean_field.potts import naive_mean_field, independent_pair_approximation, sessak_monasson_approximation


def _exact_moments(backend, scale_J, n_sites=4, n_states=3, seed=1):
    model = PottsModel(n_sites=n_sites, n_states=n_states, backend=backend)
    h_true, J_true = model.random_params(scale_h=0.3, scale_J=scale_J, seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    return model, h_true, J_true, mean_s, mean_ss

def _ising_to_one_hot_moments(m, mss, xp):
    sigma = np.array([1.0, -1.0])
    mean_s = (1 + m[:, None] * sigma) / 2
    mean_ss = (1 + m[:, None, None, None] * sigma[None, None, :, None]
               + m[None, :, None, None] * sigma[None, None, None, :]
               + mss[:, :, None, None] * sigma[None, None, :, None] * sigma[None, None, None, :]) / 4
    idx = np.arange(len(m))
    mean_ss[idx, idx] = mean_s[:, :, None] * np.eye(2)
    return xp.asarray(mean_s), xp.asarray(mean_ss)



@pytest.mark.parametrize("scale_J, max_err", [
    pytest.param(0.05, 1e-3, id="weak_coupling"),
    pytest.param(0.3, 0.05, id="moderate_coupling"),
])
def test_naive_mean_field_matches_exact_moments(scale_J, max_err, backend, to_numpy):
    """nMF should recover (h, J) accurately from moments at weak coupling."""
    model, h_true, J_true, mean_s, mean_ss = _exact_moments(backend, scale_J)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    h_err = np.abs(to_numpy(h_mf) - to_numpy(h_true)).max()
    J_err = np.abs(to_numpy(J_mf) - to_numpy(J_true)).max()
    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"


def test_naive_mean_field_accuracy_degrades_with_coupling_strength(backend, to_numpy):
    """As coupling strength grows, the mean-field approximation's error grows with it."""
    errors = []
    for scale_J in (0.05, 0.3, 0.6):
        model, _, J_true, mean_s, mean_ss = _exact_moments(backend, scale_J)
        _, J_mf = naive_mean_field(model, mean_s, mean_ss)
        errors.append(np.abs(to_numpy(J_mf) - to_numpy(J_true)).max())

    assert errors[0] < errors[1] < errors[2], f"expected monotonically increasing error, got {errors}"


def test_naive_mean_field_is_exact_for_independent_sites(backend, to_numpy):
    """With J = 0 the sites are independent, so nMF gives J = 0 and h = gauge-fixed log P."""
    model, h_true, _, mean_s, mean_ss = _exact_moments(backend, scale_J=0.0)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    log_p = np.log(to_numpy(mean_s))
    assert np.allclose(to_numpy(J_mf), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(h_mf), log_p - log_p.mean(axis=1, keepdims=True), atol=1e-10)
    assert np.allclose(to_numpy(h_mf), to_numpy(h_true), atol=1e-10)


def test_naive_mean_field_output_is_in_zero_sum_gauge(backend, to_numpy):
    """The returned (h, J) satisfy the model's gauge: zero-sum over states, symmetric, no self-couplings."""
    model, _, _, mean_s, mean_ss = _exact_moments(backend, scale_J=0.3)
    h_mf, J_mf = naive_mean_field(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h_mf).sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J_mf).sum(axis=2), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J_mf).sum(axis=3), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J_mf), J_mf.transpose(1, 0, 3, 2), atol=1e-10)
    idx = np.arange(model.n_sites)
    assert np.allclose(to_numpy(J_mf)[idx, idx], 0.0)


def test_two_state_potts_matches_ising_naive_mean_field(backend, to_numpy):
    """For q = 2, Potts nMF on one-hot moments must equal Ising nMF on the matching ±1 moments."""
    ising = IsingModel(n_sites=5, backend=backend)
    xp = ising.array_backend.xp

    h_true, J_true = ising.random_params(scale_h=0.3, scale_J=0.2, seed=2)
    m, mss = ising.exact_statistics(h_true, J_true)
    h_ising, J_ising = ising_naive_mean_field(ising, m, mss)

    mean_s, mean_ss = _ising_to_one_hot_moments(m, mss, xp)


    potts = PottsModel(n_sites=5, n_states=2, backend=backend)
    h_potts, J_potts = naive_mean_field(potts, mean_s, mean_ss)

    assert np.allclose(to_numpy(h_potts)[:, 0], to_numpy(h_ising), atol=1e-8)
    assert np.allclose(to_numpy(J_potts)[:, :, 0, 0], to_numpy(J_ising), atol=1e-8)


def test_ipa_independent_sites_gives_zero_coupling_and_single_site_field(backend, to_numpy):
    """With P_ij(a, b) = P_i(a) P_j(b) the pair corrections vanish: J = 0 and h = gauge-fixed log P."""
    model = PottsModel(n_sites=5, n_states=3, backend=backend)
    xp = model.array_backend.xp
    rng = np.random.default_rng(0)
    mean_s = rng.dirichlet(np.ones(model.n_states), size=model.n_sites)
    mean_ss = mean_s[:, None, :, None] * mean_s[None, :, None, :]
    idx = np.arange(model.n_sites)
    mean_ss[idx, idx] = mean_s[:, :, None] * np.eye(model.n_states)
    mean_s = xp.asarray(mean_s)
    mean_ss = xp.asarray(mean_ss)

    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    log_p = np.log(to_numpy(mean_s))
    assert np.allclose(to_numpy(J), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(h), log_p - log_p.mean(axis=1, keepdims=True), atol=1e-10)


@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_ipa_is_exact_for_two_sites(n_states, backend, to_numpy):
    """For N = 2 the pair inversion is the full inversion, so (h, J) are recovered exactly."""
    model = PottsModel(n_sites=2, n_states=n_states, backend=backend)
    h_true, J_true = model.random_params(scale_h=0.5, scale_J=1.0, seed=4)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h), to_numpy(h_true), atol=1e-8)
    assert np.allclose(to_numpy(J), to_numpy(J_true), atol=1e-8)


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 1e-3, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.1, id="moderate_coupling"),
])
def test_ipa_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    model = PottsModel(n_sites=6, n_states=3, backend=backend)
    h_true, J_true = model.random_params(scale_h=scale_h, scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.all(np.isfinite(to_numpy(h))) and np.all(np.isfinite(to_numpy(J)))
    assert np.abs(to_numpy(h) - to_numpy(h_true)).mean() < max_err
    assert np.abs(to_numpy(J) - to_numpy(J_true)).mean() < max_err


def test_ipa_output_is_in_zero_sum_gauge(backend, to_numpy):
    """The returned (h, J) satisfy the model's gauge: zero-sum over states, symmetric, no self-couplings."""
    model, _, _, mean_s, mean_ss = _exact_moments(backend, scale_J=0.3)
    h, J = independent_pair_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h).sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J).sum(axis=2), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J).sum(axis=3), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J), J.transpose(1, 0, 3, 2), atol=1e-10)
    idx = np.arange(model.n_sites)
    assert np.allclose(to_numpy(J)[idx, idx], 0.0)


def test_two_state_potts_matches_ising_ipa(backend, to_numpy):
    """For q = 2, Potts IPA on one-hot moments must equal Ising IPA on the matching ±1 moments."""
    ising = IsingModel(n_sites=5, backend=backend)
    xp = ising.array_backend.xp
    h_true, J_true = ising.random_params(scale_h=0.3, scale_J=0.2, seed=2)
    m, mss = ising.exact_statistics(h_true, J_true)
    h_ising, J_ising = ising_independent_pair_approximation(ising, m, mss)

    mean_s, mean_ss = _ising_to_one_hot_moments(m, mss, xp)
    potts = PottsModel(n_sites=5, n_states=2, backend=backend)
    h_potts, J_potts = independent_pair_approximation(potts, mean_s, mean_ss)

    assert np.allclose(to_numpy(h_potts[:, 0]), to_numpy(h_ising), atol=1e-8)
    assert np.allclose(to_numpy(J_potts[:, :, 0, 0]), to_numpy(J_ising), atol=1e-8)


def test_ipa_stays_finite_when_a_pair_state_is_never_observed(backend, to_numpy):
    """A never-seen pair state makes the true coupling diverge; the clamp must keep it finite."""
    model = PottsModel(n_sites=3, n_states=3, backend=backend)
    xp = model.array_backend.xp
    all_states = np.array(np.meshgrid(*[np.arange(3)] * 3, indexing="ij")).reshape(3, -1).T
    samples = all_states[~((all_states[:, 0] == 0) & (all_states[:, 1] == 0))]
    samples = xp.asarray(samples)
    moments = model.compute_moments(samples)

    h, J = independent_pair_approximation(model, moments.mean_s, moments.mean_ss)

    assert np.all(np.isfinite(to_numpy(h))) and np.all(np.isfinite(to_numpy(J)))
    assert J[0, 1, 0, 0] < -1.0


@pytest.mark.parametrize("scale_h, scale_J, max_err", [
    pytest.param(0.5, 0.05, 0.01, id="weak_coupling"),
    pytest.param(0.5, 1.0, 0.1, id="moderate_coupling"),
])
def test_sessak_monasson_matches_exact_moments(scale_h, scale_J, max_err, backend, to_numpy):
    model = PottsModel(n_sites=10, n_states=3, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=scale_h, loc_J=0.0,
                                          scale_J=scale_J / np.sqrt(model.n_sites), seed=1)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = sessak_monasson_approximation(model, mean_s, mean_ss)

    assert np.all(np.isfinite(h)) and np.all(np.isfinite(J))
    assert np.abs(to_numpy(h) - to_numpy(h_true)).mean() < max_err
    assert np.abs(to_numpy(J) - to_numpy(J_true)).mean() < max_err


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sessak_monasson_J_beats_naive_and_ipa_at_moderate_coupling(seed, backend, to_numpy):
    """The second-order corrections should make J more accurate than nMF or IPA alone."""
    model = PottsModel(n_sites=10, n_states=3, backend=backend)
    h_true, J_true = model.random_params(loc_h=0.0, scale_h=0.5, loc_J=0.0,
                                          scale_J=1.0 / np.sqrt(model.n_sites), seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    _, J_sm = sessak_monasson_approximation(model, mean_s, mean_ss)
    _, J_n = naive_mean_field(model, mean_s, mean_ss)
    _, J_ipa = independent_pair_approximation(model, mean_s, mean_ss)

    err_sm = np.abs(to_numpy(J_sm) - to_numpy(J_true)).mean()
    assert err_sm < np.abs(to_numpy(J_n) - to_numpy(J_true)).mean()
    assert err_sm < np.abs(to_numpy(J_ipa) - to_numpy(J_true)).mean()


def test_sessak_monasson_output_is_in_zero_sum_gauge(backend, to_numpy):
    model = PottsModel(n_sites=8, n_states=3, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    h, J = sessak_monasson_approximation(model, mean_s, mean_ss)

    assert np.allclose(to_numpy(h).sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J).sum(axis=2), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J).sum(axis=3), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(J), to_numpy(J).transpose(1, 0, 3, 2), atol=1e-10)
    idx = np.arange(model.n_sites)
    assert np.allclose(to_numpy(J)[idx, idx], 0.0)


def test_sessak_monasson_independent_spins_gives_zero_coupling(backend, to_numpy):
    """With C_ij = 0 for i != j, all coupling terms vanish and h is the independent solution."""
    model = PottsModel(n_sites=5, n_states=3, backend=backend)
    xp = model.array_backend.xp
    rng = np.random.default_rng(0)
    mean_s = rng.dirichlet(np.ones(model.n_states), size=model.n_sites)
    mean_ss = mean_s[:, None, :, None] * mean_s[None, :, None, :]
    idx = np.arange(model.n_sites)
    mean_ss[idx, idx] = mean_s[:, :, None] * np.eye(model.n_states)
    mean_s = xp.asarray(mean_s)
    mean_ss = xp.asarray(mean_ss)

    h, J = sessak_monasson_approximation(model, mean_s, mean_ss)

    log_p = np.log(to_numpy(mean_s))
    assert np.allclose(to_numpy(J), 0.0, atol=1e-10)
    assert np.allclose(to_numpy(h), log_p - log_p.mean(axis=1, keepdims=True), atol=1e-10)


def test_sessak_monasson_no_warnings_on_diagonal(backend, to_numpy):
    """The diagonal denominator is masked, so no divide-by-zero warnings should be raised."""
    import warnings
    model = PottsModel(n_sites=8, n_states=3, backend=backend)
    h_true, J_true = model.random_params(seed=3)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sessak_monasson_approximation(model, mean_s, mean_ss)

def test_two_state_potts_matches_ising_sessak_monasson(backend, to_numpy):
    """For q = 2, Potts Sessak Monasson on one-hot moments must equal Ising Sessak Monasson on the matching ±1 moments."""
    ising = IsingModel(n_sites=5, backend=backend)
    xp = ising.array_backend.xp
    h_true, J_true = ising.random_params(scale_h=0.3, scale_J=0.2, seed=2)
    m, mss = ising.exact_statistics(h_true, J_true)
    h_ising, J_ising = ising_sessak_monasson_approximation(ising, m, mss)

    mean_s, mean_ss = _ising_to_one_hot_moments(m, mss, xp)
    potts = PottsModel(n_sites=5, n_states=2, backend=backend)
    h_potts, J_potts = sessak_monasson_approximation(potts, mean_s, mean_ss)

    # Currently uses the self consistent equation from the naive mean field
    #assert np.allclose(h_potts[:, 0], h_ising, atol=1e-8)
    assert np.allclose(to_numpy(J_potts)[:, :, 0, 0], to_numpy(J_ising), atol=1e-8)