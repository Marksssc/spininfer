import numpy as np
import matplotlib.pyplot as plt

from models.ising import IsingModel
from data.generate import generate_data
from data.dataset import Dataset
from objectives.PLE_ising import PleIsingObjective
from optimizers.adam import Adam
from fitters.lbfgs_fitter import LbfgsFitter
from fitters.inverse_fitter import InverseFitter

N_SITES = 50
N_SAMPLES = 200000
SEED = 1

model = IsingModel(n_sites=N_SITES, backend="numba")
truth = generate_data(model, n_samples=N_SAMPLES, iterations=1000, seed=SEED, loc_J=1.5)
dataset = Dataset(samples=truth.samples, model=model)
objective = PleIsingObjective()

# pick ONE of these two
fitter = InverseFitter(model=model, dataset=dataset, objective=objective, optimizer=Adam(lr=0.05), n_steps=300)
#fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective, maxiter=50)

h_init, J_init = model.random_params(seed=SEED + 1)
result = fitter.fit(h_init, J_init)
h, J = result.h, result.J

h_err = np.abs(h - truth.h).mean()
J_err = np.abs(J - truth.J).mean()
print(f"\nmean |h_recovered - h_true| = {h_err:.4f}")
print(f"mean |J_recovered - J_true| = {J_err:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- h recovery ---
ax = axes[0]
sc = ax.scatter(truth.h, h, c=np.arange(len(h)), cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
lims = [min(truth.h.min(), h.min()), max(truth.h.max(), h.max())]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="perfect recovery")
ax.set_xlabel("true h")
ax.set_ylabel("recovered h")
ax.set_title(f"h recovery  (MAE = {h_err:.4f})")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(sc, ax=ax, label="site index")

# --- J recovery---
iu = np.triu_indices_from(J, k=1)
true_J_flat, rec_J_flat = truth.J[iu], J[iu]

ax = axes[1]
sc = ax.scatter(true_J_flat, rec_J_flat, c=np.abs(true_J_flat - rec_J_flat),
                 cmap="magma_r", s=25, alpha=0.7, edgecolor="none")
lims = [min(true_J_flat.min(), rec_J_flat.min()), max(true_J_flat.max(), rec_J_flat.max())]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="perfect recovery")
ax.set_xlabel("true J")
ax.set_ylabel("recovered J")
ax.set_title(f"J recovery  (MAE = {J_err:.4f})")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(sc, ax=ax, label="|error|")

fig.suptitle(f"Parameter recovery")
fig.tight_layout()
plt.show()