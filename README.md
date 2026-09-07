# spininfer

Inverse Ising / Potts model fitting for neural data given a matrix of data structured as (samples, state). The algorithm can recover the pairwise couplings `J` and fields `h` of the maximum-entropy model that reproduces the observed first and second moments.

This is a setup from statistical physics applied to neuroscience where you have `N` neurons recorded over `T` time bins with an integer number of discrete states, and you want the fewest assumptions possible about the data while the model still reproduces the means and correlations of the data.

Please note that a section with the likelihood, pseudolikelihood and gradient derivations is currently being constructed.

## What's in the repo

- **Two models**: Ising (binary ±1 spins) and Potts (`q`-state categorical).
- **Four array backends**: `numpy`, `numba`, `cupy`, `jax` with the same interface, which can be swapped by an input string. `cupy`/`jax` are optional and the library falls back if these are not installed.
- **Two ways to compute model statistics**: exact enumeration (small systems only) and MCMC (Metropolis, scales to hundreds of sites but can take considerable computing and time).
- **Two fitting objectives**: pseudolikelihood (fast, no sampling needed during fitting) and moment matching (slower, needs a statistics estimator (see above), but is not biased
- **Two optimizers**: Gradient ascent and Adam optimizer.
- **Two fitters**: a self-made gradient loop, and an L-BFGS wrapper around `scipy.optimize.minimize` for when you want faster convergence and don't need a custom step rule.

Everything is built in a modular fashion: a model doesn't know or care which optimizer you're using, an optimizer doesn't know which objective produced its gradient, and so on. You mix and match.

## Installation

```bash
git clone <this-repo>
cd inverse-spin
pip install -e .
```

You'll need `numpy`, `scipy`, and `numba` at minimum. If you have a CUDA GPU and want the `cupy` backend, `pip install cupy-cudaXXX` separately. This is optional and the library will just skip registering that backend if it's missing. Same story for `jax`. These GPU backends do allow for considerably faster MCMC sampling and PLE computation, while exact enumeration is mostly limited by memory.

## Quick example: recovering a known Ising model

As an example, below is some code that generates random parameters (both h and J Gaussians centered around 0). These random parameters are used to generate data and based on this generated data parameters are inferred. These inferred parameters are then compared to the ground truth (see also `scripts`):

```python
import numpy as np

from models.ising import IsingModel
from data.generate import generate_data
from data.dataset import Dataset
from objectives.PLE_ising import PleIsingObjective
from optimizers.adam import Adam
from fitters.inverse_fitter import InverseFitter
from convergence.criteria import GradientNormConvergence

model = IsingModel(n_sites=20, backend="numba")

truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
dataset = Dataset(samples=truth.samples, model=model)

fitter = InverseFitter(
    model=model,
    dataset=dataset,
    objective=PleIsingObjective(),
    optimizer=Adam(lr=0.05),
    convergence=GradientNormConvergence(model, tol=1e-6),
    n_steps=2000,
)

h_init, J_init = model.random_params(seed=2)
result = fitter.fit(h_init, J_init)

print(f"converged: {result.converged} after {result.n_steps} steps")
print(f"mean |h_err| = {np.abs(result.h - truth.h).mean():.4f}")
print(f"mean |J_err| = {np.abs(result.J - truth.J).mean():.4f}")
```

On a 20-site system this converges in well under a second on the numba backend and typically lands within 1-2% mean absolute error of the true parameters.

## Fitting real data

If you already have observed samples, skip `generate_data` and build a `Dataset` directly:

```python
from models.ising import IsingModel
from data.dataset import Dataset

model = IsingModel(n_sites=your_data.shape[1], backend="numba")
dataset = Dataset(samples=your_data, model=model)
```

`Dataset` validates that the sample width matches the model and computes the empirical statistics in the data.

## Using L-BFGS instead

For a lot of problems L-BFGS just converges faster and with less tuning than the custom gradient loops:

```python
from fitters.lbfgs_fitter import LbfgsFitter

fitter = LbfgsFitter(model=model, dataset=dataset, objective=PleIsingObjective(), maxiter=500)
result = fitter.fit(h_init, J_init)
```

Note: this needs an objective that implements `compute_value_and_gradient`, which the PLE objectives do. `MomentMatchingObjective` only produces a gradient, which means there is no scalar likelihood for a moment-matching problem, meaning it can't be used with `LbfgsFitter` and will raise an `AttributeError` if you try. This is currently in development.

## Potts models
Same shapes, different arrays. `h` is `(n_sites, n_states)`, `J` is `(n_sites, n_sites, n_states, n_states)`, and samples are integer state labels. The following snippet shows the generation of a dataset for the Potts model:

```python
from models.potts import PottsModel

model = PottsModel(n_sites=15, n_states=3, backend="numba")
truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
dataset = Dataset(samples=truth.samples, model=model)
```

Everything else works identically to the Ising model case. For example, just swap in `PlePottsObjective` for `PleIsingObjective`.

## Moment matching + exact statistics (small systems)

If your system is small enough to enumerate exactly (roughly `n_sites` up to the low 20s for Ising, fewer for Potts depending on `n_states`), you can fit against exact model statistics instead of an MCMC estimate or the PLE method, which is the most reliable method:

```python
from objectives.moment_matching import MomentMatchingObjective
from stats.exact_estimator import ExactEstimator

objective = MomentMatchingObjective(estimator=ExactEstimator())
fitter = InverseFitter(model=model, dataset=dataset, objective=objective,
                        optimizer=Adam(lr=0.05), n_steps=10_000)
```

Swap `ExactEstimator()` for `McmcEstimator(n_samples=20_000, iterations=5000)` and the exact same code works on systems too large to enumerate.

## Regularization

Wrap any objective in `RegularizedObjective` if you want an L1/L2(or combined) penalty on the fields or couplings. This is especially useful if the data you are working with is especially biased towards a single state or data with little samples. In the former case no regularization can lead to vanishing gradients and multiple parameters sets which can generate the data and in the latter case no regularization can lead to overfitting. Example code of regularization is found below:

```python
from objectives.regularization import L2Regularizer
from objectives.regularized import RegularizedObjective

regularized = RegularizedObjective(
    objective=PleIsingObjective(),
    regularizer=L2Regularizer(strength=0.01, apply_to_J=True),
)
```

## Project layout

```
backend/        array-backend abstraction (numpy/cupy/jax dispatch, random, gauge helpers)
models/         IsingModel, PottsModel — validation, simulate, moments, gauge fixing
stats/
  mcmc/         Metropolis kernels, one file per (model, backend)
  exact/        exact enumeration kernels, one file per (model, backend)
  mcmc_estimator.py / exact_estimator.py   uniform Moments-producing wrappers
data/           Dataset (observed data + cached empirical moments), synthetic data generation
objectives/     PLE and moment-matching gradients, regularization decorators
optimizers/     gradient ascent, Adam
fitters/        custom gradient-loop fitter, L-BFGS fitter
convergence/    stopping criteria (gradient norm, moment-match agreement)
scripts/        standalone recovery demos with plots
tests/          mirrors the structure above
```

## Running the tests

```bash
pytest
```

A couple of tests (`test_backend_consistency.py` under `tests/ising/PLE` and `tests/potts/PLE`) require a CUDA GPU and `cupy` installed, which means they'll fail to collect without one. Everything else runs on CPU. Warning for testing: the full recovery tests (`test_ising_recovery.py`, `test_potts_recovery.py`) run real Metropolis chains and thousands of optimization steps, so the tests can take quite a while to complete.

## Known Limitations
- **Memory limits for exact statistics:** Exact enumeration currently has no size guard. Calling `model.exact_statistics(...)` on a large system will result in an out-of-memory error rather than a safe exit. Stick to MCMC for larger systems.
- **JAX L-BFGS constraints:** The JAX path for `LbfgsFitter` uses `jax.scipy.optimize.minimize`, which has a narrower API than SciPy's (no callbacks, BFGS only). It is operational but less battle-tested than the NumPy/Numba paths.
- **Gauge fixing:** `model.apply_gauge` is applied after every optimizer step so that `h` and `J` stay in a consistent, symmetric, zero-diagonal representation. If comparing recovered parameters against another library, ensure both are evaluated in the same gauge.

## Roadmap
- **Mathematical documentation:** A dedicated section deriving the likelihood, pseudo-likelihood, and gradients for both models.
- **PyTorch backend:** Expanding backend support to include `torch`.
- **Protein modeling:** Extending functionality to support Multiple Sequence Alignments (MSA) for protein analysis.
- **Thermodynamic observables:** Such as native methods for entropy calculations.