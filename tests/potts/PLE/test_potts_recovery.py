import numpy as np
import pytest
from models.potts import PottsModel
from data.generate import generate_data
from data.dataset import Dataset
from objectives.PLE_potts import PlePottsObjective
from optimizers.adam import Adam
from fitters.inverse_fitter import InverseFitter
from fitters.lbfgs_fitter import LbfgsFitter
from convergence.criteria import GradientNormConvergence, MomentMatchConvergence

@pytest.mark.parametrize("loc_h, scale_h, loc_J, scale_J, n_states, max_err", [
    pytest.param(0.0, 0.2, 0.0, 0.2, 3, 0.05, id="easy_symmetric_3"),
    pytest.param(0.0, 0.8, 0.0, 0.5, 3, 0.3, id="hard_symmetric_3"),
    pytest.param(0.0, 0.2, 0.0, 0.2, 5, 0.05, id="easy_symmetric_5"),
    pytest.param(0.0, 0.8, 0.0, 0.5, 5, 0.3, id="hard_symmetric_5"),
    pytest.param(0.5, 0.5, 0.0, 0.5, 3, 0.3, id="skewed_h"),
    pytest.param(0.0, 0.5, 0.5, 0.5, 3, 0.3, id="skewed_J"),
])

def test_ple_recovery(loc_h, scale_h, loc_J, scale_J, n_states, max_err):
    model = PottsModel(n_sites=30, n_states=n_states, backend="numba")
    truth = generate_data(model, n_samples=10_000, iterations=5000, seed=1,
                           loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J)
    dataset = Dataset(samples=truth.samples, model=model)

    objective = PlePottsObjective()
    convergence = GradientNormConvergence(model)
    fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective,
                           convergence=convergence, maxiter=1000)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    h_err = np.abs(h - truth.h).mean()
    J_err = np.abs(J - truth.J).mean()

    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"