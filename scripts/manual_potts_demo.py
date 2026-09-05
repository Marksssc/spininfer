import numpy as np
import matplotlib.pyplot as plt

from models.potts import PottsModel
from data.generate import generate_data
from data.dataset import Dataset
from objectives.PLE_potts import PlePottsObjective
from optimizers.adam import Adam
from fitters.lbfgs_fitter import LbfgsFitter
from fitters.inverse_fitter import InverseFitter

N_SITES = 50
N_SAMPLES = 20_000
SEED = 1

model = PottsModel(n_sites=N_SITES, backend="numba", n_states=3)
truth = generate_data(model, n_samples=N_SAMPLES, iterations=1000, seed=SEED)
dataset = Dataset(samples=truth.samples, model=model)
objective = PlePottsObjective()

# pick ONE of these two
# fitter = InverseFitter(model=model, dataset=dataset, objective=objective, optimizer=Adam(lr=0.05), n_steps=300)
fitter = LbfgsFitter(model=model, dataset=dataset, objective=objective, maxiter=500)

h_init, J_init = model.random_params(seed=SEED + 1)
result = fitter.fit(h_init, J_init)
h, J = result.h, result.J

h_err = np.abs(h - truth.h).mean()
J_err = np.abs(J - truth.J).mean()
print(f"\nmean |h_recovered - h_true| = {h_err:.4f}")
print(f"mean |J_recovered - J_true| = {J_err:.4f}")

n_states = model.n_states

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- h recovery ---
ax = axes[0]
state_idx = np.tile(np.arange(n_states), model.n_sites) 
cmap = plt.get_cmap("tab10", n_states) 
sc = ax.scatter(truth.h.ravel(), h.ravel(), c=state_idx, cmap=cmap, s=40,
                 edgecolor="k", linewidth=0.3,
                 vmin=-0.5, vmax=n_states - 0.5)
lims = [min(truth.h.min(), h.min()), max(truth.h.max(), h.max())]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="perfect recovery")
ax.set_xlabel("true h")
ax.set_ylabel("recovered h")
ax.set_title(f"h recovery  (MAE = {h_err:.4f})")
ax.legend(loc="upper left", fontsize=8)
cbar = fig.colorbar(sc, ax=ax, label="state", ticks=range(n_states))

# --- J recovery ---
iu = np.triu_indices(model.n_sites, k=1)
true_J_flat = truth.J[iu[0], iu[1], :, :].ravel()
rec_J_flat = J[iu[0], iu[1], :, :].ravel()

ax = axes[1]
sc = ax.scatter(true_J_flat, rec_J_flat, c=np.abs(true_J_flat - rec_J_flat),
                 cmap="magma_r", s=20, alpha=0.6, edgecolor="none")
lims = [min(true_J_flat.min(), rec_J_flat.min()), max(true_J_flat.max(), rec_J_flat.max())]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="perfect recovery")
ax.set_xlabel("true J")
ax.set_ylabel("recovered J")
ax.set_title(f"J recovery  (MAE = {J_err:.4f})")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(sc, ax=ax, label="|error|")

fig.suptitle(f"Potts parameter recovery, n_samples={N_SAMPLES}", fontsize=12)
fig.tight_layout()
plt.show()