import numpy as np
import pytest
from objectives import PLE_ising_numpy, PLE_ising_numba


def _make_problem(rng, sites=8, n_samples=200):
    h = rng.normal(scale=0.2, size=sites)
    J = rng.normal(scale=0.2, size=(sites, sites))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    data = rng.choice([-1.0, 1.0], size=(n_samples, sites))
    empirical_mean_s = data.mean(axis=0)
    empirical_mean_ss = (data.T @ data) / n_samples
    return h, J, data, empirical_mean_s, empirical_mean_ss


def _random_direction(rng, h, J):
    dh = rng.normal(size=h.shape)
    dJ = rng.normal(size=J.shape)
    dJ = (dJ + dJ.T) / 2
    np.fill_diagonal(dJ, 0.0)
    return dh, dJ


def _central_diff_directional_derivative(value_fn, h, J, dh, dJ, data,
                                           empirical_mean_s, empirical_mean_ss,
                                           eps=1e-5):
    v_plus, _, _ = value_fn(h + eps * dh, J + eps * dJ, data,
                             empirical_mean_s, empirical_mean_ss)
    v_minus, _, _ = value_fn(h - eps * dh, J - eps * dJ, data,
                              empirical_mean_s, empirical_mean_ss)
    return (v_plus - v_minus) / (2 * eps)


@pytest.mark.parametrize("kernel", [PLE_ising_numpy, PLE_ising_numba],
                          ids=["numpy", "numba"])
def test_gradient_matches_finite_difference(kernel):
    rng = np.random.default_rng(0)
    h, J, data, mean_s, mean_ss = _make_problem(rng)

    value, grad_h, grad_J = kernel.value_and_gradient(h, J, data, mean_s, mean_ss)

    for trial in range(5):
        dh, dJ = _random_direction(np.random.default_rng(trial), h, J)

        numerical = _central_diff_directional_derivative(
            kernel.value_and_gradient, h, J, dh, dJ, data, mean_s, mean_ss
        )

        analytic = np.sum(grad_h * dh) + np.sum(grad_J * dJ)

        assert np.isclose(numerical, analytic, rtol=1e-3, atol=1e-5), (
            f"trial {trial}: numerical={numerical:.6f} vs analytic={analytic:.6f}"
        )