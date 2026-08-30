import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    measurements, sites = data.shape
    states = h.shape[1]
    inv_m = 1.0 / measurements

    value_per_site = np.zeros(sites)
    h_gradient = np.zeros((sites, states))
    J_gradient = np.zeros((sites, sites, states, states))

    for site in prange(sites):
        energy = np.zeros(states)
        cond_prob = np.zeros(states)
        terms = np.zeros(states)
        sum_state = np.zeros(states)
        site_log_lik = 0.0

        for measurement in range(measurements):
            actual_state1 = int(data[measurement, site])

            for state in range(states):
                energy[state] = h[site, state]

            for site2 in range(sites):
                if site == site2:
                    continue
                actual_state2 = int(data[measurement, site2])
                for state in range(states):
                    energy[state] += J[site, site2, state, actual_state2]

            max_val = energy[0]
            for state in range(states):
                if energy[state] > max_val:
                    max_val = energy[state]

            sum_exp = 0.0
            for state in range(states):
                e = np.exp(energy[state] - max_val)
                cond_prob[state] = e
                sum_exp += e
            site_log_lik += (energy[actual_state1] - max_val) - np.log(sum_exp)

            for state in range(states):
                prob = cond_prob[state] / sum_exp
                term = (1.0 if state == actual_state1 else 0.0) - prob
                terms[state] = term
                sum_state[state] += term

            for site2 in range(sites):
                if site == site2:
                    continue
                actual_state2 = int(data[measurement, site2])
                for state1 in range(states):
                    J_gradient[site, site2, state1, actual_state2] += terms[state1]

        value_per_site[site] = site_log_lik
        h_gradient[site, :] = sum_state * inv_m

    value = np.sum(value_per_site) * inv_m
    J_gradient *= inv_m

    J_sym = np.zeros_like(J_gradient)
    for i in range(sites):
        for j in range(sites):
            for a in range(states):
                for b in range(states):
                    J_sym[i, j, a, b] = (J_gradient[i, j, a, b] + J_gradient[j, i, b, a]) / 2.0

    return value, h_gradient, J_sym

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    _, h_grad, J_grad = value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    return h_grad, J_grad