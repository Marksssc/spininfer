import cupy as cp

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    measurements, sites = data.shape
    inv_m = 1.0 / measurements

    local_fields = data @ J + h
    theta = cp.tanh(local_fields)

    h_grad = empirical_mean_s - cp.sum(theta, axis=0) * inv_m
    J_grad = empirical_mean_ss - (theta.T @ data) * inv_m
    cp.fill_diagonal(J_grad, 0.0)
    J_grad = (J_grad + J_grad.T) / 2

    return h_grad, J_grad