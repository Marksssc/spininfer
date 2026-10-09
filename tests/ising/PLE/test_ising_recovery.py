import numpy as np
import pytest
from spininfer.models.ising import IsingModel
from spininfer.data.generate import generate_data
from spininfer.data.dataset import Dataset
from spininfer.objectives.ple_ising import PleIsingObjective
from spininfer.fitters.lbfgs_fitter import LbfgsFitter
from spininfer.convergence.criteria import GradientNormConvergence, MomentMatchConvergence

@pytest.mark.parametrize("loc_h, scale_h, loc_J, scale_J, max_err", [
    pytest.param(0.0, 0.2, 0.0, 0.2, 0.05, id="easy_symmetric"),
    pytest.param(0.0, 1.0, 0.0, 0.5, 0.15, id="hard_symmetric"),
    pytest.param(0.5, 0.5, 0.0, 0.5, 0.15, id="skewed_h"),
    pytest.param(0.0, 0.5, 1.0, 0.5, 0.15, id="skewed_J"),
])

def test_ple_recovery(loc_h, scale_h, loc_J, scale_J, max_err, backend, to_numpy):
    model = IsingModel(n_sites=30, backend=backend)
    truth = generate_data(model, n_samples=20_000, iterations=10000, seed=1,
                           loc_h=loc_h, scale_h=scale_h, loc_J=loc_J, scale_J=scale_J)
    dataset = Dataset(samples=truth.samples, model=model)

    objective = PleIsingObjective()
    convergence = GradientNormConvergence(model)#MomentMatchConvergence(model, dataset)
    fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective,
                            convergence=convergence, maxiter=1000, tol=1e-10)
    h_init, J_init = model.random_params(seed=2)
    result = fitter.fit(h_init, J_init)
    h, J = result.h, result.J

    h_err = np.abs(to_numpy(h) - to_numpy(truth.h)).mean()
    J_err = np.abs(to_numpy(J) - to_numpy(truth.J)).mean()

    assert h_err < max_err, f"h recovery error too high: {h_err}"
    assert J_err < max_err, f"J recovery error too high: {J_err}"