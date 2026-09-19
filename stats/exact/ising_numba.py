import numpy as np
from numba import njit, prange

@njit(inline='always')
def _state_and_energy(ii, sites, h, J, state):
    for j in range(sites):
        bit = (ii >> j) & 1
        state[j] = 1.0 if bit else -1.0
    return h @ state + 0.5 * (state @ J @ state)

@njit(parallel=True, fastmath=True)
def get_exact_statistics(h, J):
    sites = len(h)
    num_states = 1 << sites

    total_z = 0.0
    sum_means = np.zeros(sites)
    sum_corrs = np.zeros((sites, sites))

    for ii in prange(num_states):
        state = np.empty(sites)
        energy = _state_and_energy(ii, sites, h, J, state)
        weight = np.exp(energy)

        total_z += weight
        sum_means += state * weight
        sum_corrs += np.outer(state, state) * weight

    return sum_means / total_z, sum_corrs / total_z

@njit(parallel=True, fastmath=True)
def get_exact_thermodynamics(h, J):
    sites = len(h)
    num_states = 1 << sites

    total_z = 0.0
    sum_energy = 0.0
    sum_energy2 = 0.0

    for ii in prange(num_states):
        state = np.empty(sites)
        energy = _state_and_energy(ii, sites, h, J, state)
        weight = np.exp(energy)

        total_z += weight
        sum_energy += weight * energy
        sum_energy2 += weight * energy * energy

    enthalpy = -sum_energy / total_z
    entropy = np.log(total_z) + enthalpy
    mean_e2 = sum_energy2 / total_z
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity
