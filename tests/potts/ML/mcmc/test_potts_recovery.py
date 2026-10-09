import numpy as np
import pytest
from spininfer.models.potts import PottsModel
from spininfer.data.generate import generate_data
from spininfer.data.dataset import Dataset
from spininfer.objectives.moment_matching import MomentMatchingObjective
from spininfer.stats.mcmc_estimator import McmcEstimator
from spininfer.optimizers.adam import Adam
from spininfer.fitters.inverse_fitter import InverseFitter
from spininfer.convergence.criteria import GradientNormConvergence, MomentMatchConvergence
import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("loc_h, scale_h, loc_J, scale_J, n_states, max_err", [
    pytest.param(0.0, 0.2, 0.0, 0.2, 3, 0.05, id="easy_symmetric_3"),
    pytest.param(0.0, 0.8, 0.0, 0.5, 3, 0.15, id="hard_symmetric_3"),
    pytest.param(0.0, 0.2, 0.0, 0.2, 5, 0.05, id="easy_symmetric_5"),
    pytest.param(0.0, 0.5, 0.0, 0.5, 5, 0.2,  id="hard_symmetric_5"),
    pytest.param(0.5, 0.5, 0.0, 0.5, 3, 0.15, id="skewed_h"),
    pytest.param(0.0, 0.5, 0.5, 0.5, 3, 0.15, id="skewed_J"),
])

def test_moment_matching_recovery(loc_h, scale_h, loc_J, scale_J, n_states, max_err, backend, to_numpy):
    model = PottsModel(n_sites=30, n_states=n_states, backend=backend)
    truth = generate_data(model, n_samples=20_000, iterations=10000, seed=1,
                           loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J)
    dataset = Dataset(samples=truth.samples, model=model)

    objective = MomentMatchingObjective(estimator=McmcEstimator(n_samples=5000))
    convergence = GradientNormConvergence(model)#MomentMatchConvergence(model, dataset)

    fitter = InverseFitter(model=model, dataset=dataset, objective=objective,
                           optimizer=Adam(lr=0.05), convergence=convergence,
                           n_steps=1000, verbose=False)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    h_err = np.abs(to_numpy(h) - to_numpy(truth.h)).mean()
    J_err = np.abs(to_numpy(J) - to_numpy(truth.J)).mean()

    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"