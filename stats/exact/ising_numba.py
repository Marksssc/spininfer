import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def get_exact_statistics(h, J):
    sites = len(h)
    num_states = 1 << sites

    total_z = 0.0
    sum_means = np.zeros(sites)
    sum_corrs = np.zeros((sites, sites))

    for ii in prange(num_states):
        state = np.empty(sites)
        for j in range(sites):
            bit = (ii >> j) & 1
            state[j] = 1 if bit else -1
        
        energy = -(h @ state + (state @ J @ state)*0.5)
        weight = np.exp(-energy)

        total_z += weight
        sum_means += state * weight
        sum_corrs += np.outer(state, state) * weight

    return sum_means / total_z, sum_corrs / total_z