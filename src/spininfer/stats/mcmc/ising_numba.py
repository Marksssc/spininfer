import numpy as np
from numba import njit, prange

_CHAIN_SEED_STRIDE = 1_000_003

@njit(inline='always', fastmath=True)
def delta_energy(h, J, spins, site_change):
    s_i = spins[site_change]
    delta_mag_energy = s_i * h[site_change]
    delta_cor_energy = s_i*np.dot(J[site_change, :], spins)
    return 2*(delta_mag_energy + delta_cor_energy)

@njit(parallel=True, fastmath=True)
def simulate(h, J, samples, iterations=1000, seed=0):
    sites = len(h)
    lattice = np.empty((samples, sites))

    for ii in prange(samples):
        np.random.seed(seed * _CHAIN_SEED_STRIDE + ii)
        lattice_row = lattice[ii, :]
        for kk in range(sites):
            lattice_row[kk] = 1.0 if np.random.random() < 0.5 else -1.0

        local_field = h + (J@lattice_row)
        for jj in range(iterations):
            site_to_flip = np.random.randint(sites)
            s_i = lattice_row[site_to_flip]

            energy_dif = 2 * s_i * local_field[site_to_flip]
            if energy_dif <= 0 or np.random.random() < np.exp(-energy_dif):
                lattice_row[site_to_flip] *= -1
                local_field -= 2 * s_i * J[site_to_flip, :]

    return lattice