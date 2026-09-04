import numpy as np
import pytest
from models.ising import IsingModel
from data.generate import generate_data
from data.dataset import Dataset
from objectives.moment_matching import MomentMatchingObjective
from stats.exact_estimator import ExactEstimator
from optimizers.adam import Adam
from fitters.inverse_fitter import InverseFitter
from convergence.criteria import GradientNormConvergence, MomentMatchConvergence

@pytest.mark.parametrize("loc_h, scale_h, loc_J, scale_J, max_err", [
    pytest.param(0.0, 0.2, 0.0, 0.2, 0.05, id="easy_symmetric"),
    pytest.param(0.0, 1.0, 0.0, 0.5, 0.15, id="hard_symmetric"),
    pytest.param(0.5, 0.5, 0.0, 0.5, 0.15, id="skewed_h"),
    pytest.param(0.0, 0.5, 1.0, 0.5, 0.15, id="skewed_J"),
])

def test_MLE_recovery(loc_h, scale_h, loc_J, scale_J, max_err):
    model = IsingModel(n_sites=10, backend="numba")
    truth = generate_data(model, n_samples=20_000, iterations=5000, seed=1,
                           loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J)
    dataset = Dataset(samples=truth.samples, model=model)

    objective = MomentMatchingObjective(estimator=ExactEstimator())
    convergence = GradientNormConvergence()#MomentMatchConvergence(model, dataset)
    fitter = InverseFitter(model=model, dataset=dataset, objective=objective,
                            optimizer=Adam(lr=0.05), convergence=convergence, 
                            n_steps=10000, verbose=False)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    h_err = np.abs(h - truth.h).mean()
    J_err = np.abs(J - truth.J).mean()

    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"
    assert result.converged == True, f"Not enough steps for convergence {result.n_steps}, with final gradient norm {result.final_grad_norm}.6f"