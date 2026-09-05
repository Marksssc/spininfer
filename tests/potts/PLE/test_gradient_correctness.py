import numpy as np
import pytest
from objectives import PLE_potts_numpy, PLE_potts_numba


def _make_problem(rng, sites=6, n_states=3, n_samples=200):
    h = rng.normal(scale=0.2, size=(sites, n_states))
    J = rng.normal(scale=0.2, size=(sites, sites, n_states, n_states))
    J = (J + J.transpose(1, 0, 3, 2)) / 2
    idx = np.arange(sites)
    J[idx, idx, :, :] = 0.0

    data = rng.integers(0, n_states, size=(n_samples, sites))
    empirical_mean_s = data.mean(axis=0)    
    empirical_mean_ss = (data.T @ data) / n_samples 
    return h, J, data, empirical_mean_s, empirical_mean_ss


def _random_direction(rng, h, J):
    dh = rng.normal(size=h.shape)
    dJ = rng.normal(size=J.shape)
    dJ = (dJ + dJ.transpose(1, 0, 3, 2)) / 2
    idx = np.arange(J.shape[0])
    dJ[idx, idx, :, :] = 0.0
    return dh, dJ


def _central_diff_directional_derivative(value_fn, h, J, dh, dJ, data,
                                           empirical_mean_s, empirical_mean_ss,
                                           eps=1e-5):
    v_plus, _, _ = value_fn(h + eps * dh, J + eps * dJ, data,
                             empirical_mean_s, empirical_mean_ss)
    v_minus, _, _ = value_fn(h - eps * dh, J - eps * dJ, data,
                              empirical_mean_s, empirical_mean_ss)
    return (v_plus - v_minus) / (2 * eps)


@pytest.mark.parametrize("kernel", [PLE_potts_numpy, PLE_potts_numba],
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