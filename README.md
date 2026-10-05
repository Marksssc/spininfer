# spininfer

This repository is a multi-backend library for parameter inference in the Ising and Potts models, featuring mean-field, adaptive cluster expansion, maximum likelihood, and pseudo likelihood methods, along with thermodynamic variable extraction on the CPU (NumPy and Numba) and GPU (CuPy and JAX).

Inverse Ising and Potts inference is used wherever you want a maximum-entropy model that reproduces the means and pairwise correlations of the data and makes the fewest other assumptions. In neuroscience, this means modelling N neurons recorded over T time bins with discrete states. In protein structure analysis, a multiple sequence alignment (MSA) is modelled as a 21-state Potts model, which is the basis of contact prediction.

## What's in the repo

- **Two models**: Ising (binary ±1 spins) and Potts (`q`-state categorical).
- **Four array backends**: `numpy`, `numba`, `cupy`, `jax` with the same interface, which can be swapped by an input string. `cupy`/`jax` are optional and the library falls back if these are not installed. Note that JAX by default has float 32 precision. To get float 64 precision run `jax.config.update("jax_enable_x64", True)`.
- **Three families of inference methods** (details and references in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)):
  - *Likelihood-based:* maximum likelihood by moment matching, and pseudolikelihood.
  - *Closed-form mean-field:* naive mean field, TAP, independent pair, Sessak–Monasson and Bethe. These need only the first and second moments. TAP and Bethe are Ising-only.
  - *Adaptive cluster expansion:* builds the parameters from smaller clusters.
- **Two ways to compute model statistics**: exact enumeration (small systems only) and MCMC (Metropolis, scales to hundreds of sites but can take considerable computing and time).
- **Thermodynamic variables**: calculation of the energy, free energy, entropy, and heat capacity through exact enumeration or thermodynamic integration.
- **Optimizers and fitters**: gradient ascent and Adam, a custom gradient-loop fitter, and an L-BFGS wrapper around `scipy.optimize.minimize` or `optax.lbfgs` for faster convergence.
- **Regularization**: L1 and L2 penalties that can be applied to any objective.
- **Data loading**: raw sample matrices, spike trains and protein multiple sequence alignments transformed to Ising/Potts datasets.

The pieces are written in a modular fashion and can thus be mixed.

## Installation

```bash
git clone https://github.com/Marksssc/spininfer
cd spininfer
pip install -e .
```

You'll need `numpy`, `scipy`, and `numba` at minimum. If you have a CUDA GPU and want to use the significantly faster GPU implementations, you'll need to install these packages to match your CUDA version. You can check this with nvidia-smi in your terminal. For CuPy run

```bash
pip install cupy-cuda12x  # For CUDA 12
pip install cupy-cuda13x  # For CUDA 13
```

and for JAX

```bash
pip install -U "jax[cuda12]"  # For CUDA 12
pip install -U "jax[cuda13]"  # For CUDA 13
```

GPU packages are optional; without them the library uses the numpy and numba backends.

## Quick example: recovering a known Ising model

As an example, below is some code that generates random parameters (both h and J Gaussians centered around 0). These random parameters are used to generate data and based on this generated data parameters are inferred. These inferred parameters are then compared to the ground truth (for more examples see `scripts`):

```python
import numpy as np

from spininfer.models.ising import IsingModel
from spininfer.data.generate import generate_data
from spininfer.data.dataset import Dataset
from spininfer.objectives.PLE_ising import PleIsingObjective
from spininfer.optimizers.adam import Adam
from spininfer.fitters.inverse_fitter import InverseFitter
from spininfer.convergence.criteria import GradientNormConvergence

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
from spininfer.models.ising import IsingModel
from spininfer.data.dataset import Dataset

model = IsingModel(n_sites=your_data.shape[1], backend="numba")
dataset = Dataset(samples=your_data, model=model)
```

`Dataset` validates that the sample width matches the model and computes the empirical statistics in the data.

## Using L-BFGS instead

For a lot of problems L-BFGS converges faster and with less tuning than the custom gradient loops:

```python
from spininfer.fitters.lbfgs_fitter import LbfgsFitter

fitter = LbfgsFitter(model=model, dataset=dataset, objective=PleIsingObjective(), maxiter=500)
result = fitter.fit(h_init, J_init)
```

Note: this needs an objective that implements `compute_value_and_gradient`, which the exact MLE and PLE objectives do. MCMC moment matching is not supported with L-BFGS.

## Using a mean-field method instead
If you think a mean-field method suffices, or should be used as a starting point for the MLE or PLE algorithms this can be implemented as follows:

```python
# Mean-field fitting
from spininfer.fitters.mean_field_fitter import NaiveMeanFieldFitter, TAPMeanFieldFitter, IndependentPairFitter, SessakMonassonFitter, BetheFitter
fitter_nmf = NaiveMeanFieldFitter(model=model, dataset=dataset)
fitter_tap = TAPMeanFieldFitter(model=model, dataset=dataset) #Ising only
fitter_ipa = IndependentPairFitter(model=model, dataset=dataset)
fitter_sm = SessakMonassonFitter(model=model, dataset=dataset)
fitter_bethe = BetheFitter(model=model, dataset=dataset) #Ising only

result = fitter_nmf.fit()
h_nmf, J_nmf = result.h, result.J

# Using mean-field as a starting point
from spininfer.fitters.lbfgs_fitter import LbfgsFitter
h_init = h_nmf
J_init = J_nmf

fitter = LbfgsFitter(model=model, dataset=dataset, objective=PleIsingObjective(), maxiter=500)
result = fitter.fit(h_init, J_init)
```

## Using adaptive cluster expansion
Another method that is good at recovering strong coupling is adaptive cluster expansion, which is implemented as follows:

```python
# Adaptive cluster expansion
from spininfer.fitters.ACE_fitter import ACEFitter

theta = 1e-3
fitter = ACEFitter(model=model, dataset=dataset, threshold=theta)
ACE_result = fitter.fit()
```

## Potts models
Same shapes, different arrays. `h` is `(n_sites, n_states)`, `J` is `(n_sites, n_sites, n_states, n_states)`, and samples are integer state labels. The following code shows the generation of a dataset for the Potts model (for more examples see `scripts`):

```python
from spininfer.models.potts import PottsModel

model = PottsModel(n_sites=15, n_states=3, backend="numba")
truth = generate_data(model, n_samples=20_000, iterations=1000, seed=1)
dataset = Dataset(samples=truth.samples, model=model)
```

Everything else works identically to the Ising model case. For example, just swap in `PlePottsObjective` for `PleIsingObjective`.

## Moment matching + exact statistics (small systems)

If your system is small enough to enumerate exactly (roughly `n_sites` up to the low 20s for Ising, fewer for Potts depending on `n_states`), you can fit against exact model statistics instead of an MCMC estimate or the PLE method, which is the most reliable method:

```python
from spininfer.objectives.moment_matching import MomentMatchingObjective
from spininfer.stats.exact_estimator import ExactEstimator

objective = MomentMatchingObjective(estimator=ExactEstimator())
fitter = InverseFitter(model=model, dataset=dataset, objective=objective,
                        optimizer=Adam(lr=0.05), n_steps=10_000)
```

Swap `ExactEstimator()` for `McmcEstimator(n_samples=20_000, iterations=5000)` and the exact same code works on systems too large to enumerate. However in the case exact enumeration does not work I would recommend PLE.

## Thermodynamic quantities

Given the parameters, you can compute the free energy, energy, entropy and heat capacity, either exactly or by thermodynamic integration over MCMC samples. Units are β = 1. The following code gives an example:

```python
from spininfer.models.ising import IsingModel
from spininfer.stats.exact_thermodynamics_estimator import ExactThermodynamicsEstimator
from spininfer.stats.thermodynamics_integration_estimator import ThermodynamicsIntegrationEstimator

model = IsingModel(n_sites=12, backend="numba")
h, J = model.random_params(seed=1)

exact = ExactThermodynamicsEstimator().estimate(model, h, J)
ti = ThermodynamicsIntegrationEstimator(n_samples=10_000, iterations=1000).estimate(model, h, J)

print(exact.free_energy, exact.energy, exact.entropy, exact.heat_capacity)
print(ti.free_energy, ti.energy, ti.entropy, ti.heat_capacity)

```

Thermodynamic integration only needs MCMC samples so it can be used for large systems. The accuracy is set by `n_samples`, `iterations` and the number for integration poitns `n_points`.


## Regularization

Wrap any objective in `RegularizedObjective` if you want an L1/L2(or combined) penalty on the fields or couplings. This is useful if the data you are working with is biased towards a single state or data with little samples. In the former case no regularization can lead to vanishing gradients and multiple parameters which can generate the data and in the latter case no regularization can lead to overfitting. Example code of regularization is found below:

```python
from spininfer.objectives.regularization import L2Regularizer
from spininfer.objectives.regularized import RegularizedObjective

regularized = RegularizedObjective(
    objective=PleIsingObjective(),
    regularizer=L2Regularizer(strength=0.01, apply_to_J=True),
)
```

## Project layout

```
src/spininfer/
  backend/            array-backend functions and loading
  models/             IsingModel, PottsModel: validation, simulation, moments, gauge fixing
  stats/
    mcmc/             Metropolis kernels, one file per (model, backend)
    exact/            exact enumeration kernels, one file per (model, backend)
    mcmc_estimator.py, exact_estimator.py   uniform wrappers that return model moments
    thermodynamics.py and *_thermodynamics_*estimator.py files   return thermodynamic variables
    moments.py        container for first and second moments
  data/               Dataset (observed samples and empirical moments), synthetic data generation, spike-train and MSA loaders
  objectives/         pseudolikelihood and moment-matching objectives, regularization wrappers
  optimizers/         gradient ascent, Adam
  fitters/            gradient-loop fitter, L-BFGS fitter, mean-field fitters, adaptive cluster expansion fitter
  mean_field/         closed-form estimators: naive mean field, TAP, independent pair, Sessak–Monasson, Bethe
  cluster_expansion/  adaptive cluster expansion (cluster fitting, expansion and assembly)
  convergence/        stopping criteria (gradient norm, moment-match agreement)
docs/                 method overview and derivations (PDF)
scripts/              standalone recovery demos
tests/                tests, mirroring the package structure
```


## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

Warning for testing: the full recovery tests (`test_ising_recovery.py`, `test_potts_recovery.py`) run real Metropolis chains and thousands of optimization steps, so the tests can take quite a while to complete.

## Notes
- **Gauge fixing:** `model.apply_gauge` is applied after every optimizer step so that `h` and `J` stay in a consistent, symmetric, zero-diagonal representation. If comparing recovered parameters against another library, ensure both are evaluated in the same gauge.

## Roadmap
- **Minimum probability flow:** The minimum probability flow algorithm will be implemented.
- **Interaction screening:** Interaction screening will be implemented.