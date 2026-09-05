import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id

@njit(fastmath=True, parallel=True)
def get_exact_statistics(h, J):
    sites, states = h.shape
    num_states = states**sites
    n_threads = get_num_threads()

    sum_means_t = np.zeros((n_threads, sites, states))
    sum_corrs_t = np.zeros((n_threads, sites, sites, states, states))
    total_z_t = np.zeros(n_threads)
    state_t = np.zeros((n_threads, sites), dtype=np.int32)

    powers = np.zeros(sites, dtype=np.int64)
    for jj in range(sites):
        powers[jj] = states ** jj

    for ii in prange(num_states):
        tid = get_thread_id()
        
        for jj in range(sites):
            state_t[tid, jj] = (ii // powers[jj]) % states

        energy = 0.0
        for site in range(sites):
            energy -= h[site, state_t[tid, site]]
            for site2 in range(sites):
                if site == site2:
                    continue
                energy -= 0.5 * J[site, site2, state_t[tid, site], state_t[tid, site2]]

        weight = np.exp(-energy)
        total_z_t[tid] += weight
        for site in range(sites):
            sum_means_t[tid, site, state_t[tid, site]] += weight
            for site2 in range(sites):
                sum_corrs_t[tid, site, site2, state_t[tid, site], state_t[tid, site2]] += weight

    total_z = np.sum(total_z_t)
    sum_means = np.sum(sum_means_t, axis=0)
    sum_corrs = np.sum(sum_corrs_t, axis=0)
    return sum_means / total_z, sum_corrs / total_z