import numpy as np
import pytest
from models.ising import IsingModel
from data.generate import generate_data
from data.dataset import Dataset
from fitters.mean_field_fitter import NaiveMeanFieldFitter, TAPMeanFieldFitter


@pytest.mark.parametrize("loc_h, scale_h, loc_J, scale_J, h_max_err, J_max_err", [
    pytest.param(0.0, 0.1, 0.0, 0.5, 0.05, 0.03, id="weak_coupling"),
])
def test_naive_mean_field_recovery(loc_h, scale_h, loc_J, scale_J, h_max_err, J_max_err):
    """NaiveMeanFieldFitter should recover weak-coupling ground truth from MCMC-sampled data,
    in one closed-form step."""
    model = IsingModel(n_sites=10, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=2000, seed=1,
                           loc_h=loc_h, scale_h=scale_h, loc_J=loc_J,
                           scale_J=scale_J / np.sqrt(model.n_sites))
    dataset = Dataset(samples=truth.samples, model=model)

    fitter = NaiveMeanFieldFitter(model=model, dataset=dataset)
    result = fitter.fit()

    h_err = np.abs(result.h - truth.h).mean()
    J_err = np.abs(result.J - truth.J).mean()

    assert h_err < h_max_err, f"h recovery error too high: {h_err}"
    assert J_err < J_max_err, f"J recovery error too high: {J_err}"
    assert result.converged is True
    assert result.n_steps == 1


def test_tap_recovery():
    """TAPMeanFieldFitter should recover weak-coupling ground truth from MCMC-sampled data
    in one closed-form step."""
    model = IsingModel(n_sites=10, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=2000, seed=1,
                           loc_h=0.0, scale_h=0.3, loc_J=0.0,
                           scale_J=0.5 / np.sqrt(model.n_sites))
    dataset = Dataset(samples=truth.samples, model=model)

    result = TAPMeanFieldFitter(model=model, dataset=dataset).fit()

    assert np.abs(result.h - truth.h).mean() < 0.05
    assert np.abs(result.J - truth.J).mean() < 0.03
    assert result.converged is True
    assert result.n_steps == 1
