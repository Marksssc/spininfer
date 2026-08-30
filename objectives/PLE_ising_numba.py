import numpy as np
from numba import njit

@njit(fastmath=True)
def value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    measurements, sites = data.shape
    inv_m = 1.0 / measurements

    local_fields = data @ J + h
    theta = np.tanh(local_fields)

    h_grad = empirical_mean_s - np.sum(theta, axis=0) * inv_m
    J_grad = empirical_mean_ss - (theta.T @ data) * inv_m
    np.fill_diagonal(J_grad, 0.0)
    J_grad = (J_grad + J_grad.T) / 2

    log_probs = -np.log1p(np.exp(-2.0 * data * local_fields))
    value = np.sum(log_probs) * inv_m

    return value, h_grad, J_grad

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    _, h_grad, J_grad = value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    return h_grad, J_grad