import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def get_exact_statistics(h, J):
    sites, states = np.shape(h)
    num_states = states**sites

    total_z = 0.0
    sum_means = np.zeros((sites, states))
    sum_corrs = np.zeros((sites, sites, states, states))
    state = np.zeros(sites, dtype=np.int32)

    for ii in range(num_states):
        temp = ii

        for jj in range(sites):
            state[jj] = temp % states
            temp = temp//states
        
        energy = 0.0
        for site in range(sites):
            energy -= h[site, state[site]]
            for site2 in range(sites):
                if site == site2:
                    continue
                energy -= 0.5*J[site, site2, state[site], state[site2]]
    
        weight = np.exp(-energy)
        total_z += weight

        for site in range(sites):
            sum_means[site, state[site]] += weight
            for site2 in range(sites):
                if site == site2:
                    continue
                sum_corrs[site, site2, state[site], state[site2]] += weight
    
    return sum_means/total_z, sum_corrs/total_z