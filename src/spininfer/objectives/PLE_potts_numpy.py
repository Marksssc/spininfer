import numpy as np

def value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    measurements, sites = data.shape
    n_states = h.shape[1]
    inv_m = 1.0 / measurements
    one_hot = np.eye(n_states)[data]

    energy = h[None, :, :] + np.einsum('ijab,njb->nia', J, one_hot)

    max_energy = energy.max(axis=2, keepdims=True) 
    exp_energy = np.exp(energy - max_energy)             
    sum_exp = exp_energy.sum(axis=2, keepdims=True)                   
    cond_prob = exp_energy / sum_exp

    observed_energy = np.einsum('nia,nia->ni', energy, one_hot)
    log_lik = observed_energy - max_energy[:, :, 0] - np.log(sum_exp[:, :, 0])
    value = log_lik.sum() * inv_m
    
    terms = one_hot - cond_prob
    h_gradient = terms.sum(axis=0) * inv_m

    J_gradient = np.einsum('nia,njb->ijab', terms, one_hot) * inv_m
    idx = np.arange(sites)
    J_gradient[idx, idx, :, :] = 0.0
    J_sym = (J_gradient + J_gradient.transpose(1, 0, 3, 2)) / 2.0

    return value, h_gradient, J_sym

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    _, h_grad, J_grad = value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    return h_grad, J_grad