import jax 
import jax.numpy as jnp
import jax.nn

def _scalar_log_pseudolikelihood(h, J, data):
    measurements, sites = data.shape
    n_states = h.shape[1]
    inv_m = 1.0 / measurements

    one_hot = jnp.eye(n_states)[data]

    energy = h[None, :, :] + jnp.einsum('ijab,njb->nia', J, one_hot)

    observed_energy = jnp.einsum('nia,nia->ni', energy, one_hot)

    log_z = jax.nn.logsumexp(energy, axis=2)

    log_lik = observed_energy - log_z                

    return log_lik.sum() * inv_m

@jax.jit
def _value_and_grad(h, J, data):
    return jax.value_and_grad(_scalar_log_pseudolikelihood, argnums=(0, 1))(h, J, data)

def value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    value, (h_grad, J_grad) = _value_and_grad(h, J, data)

    n_sites = J_grad.shape[0]
    J_grad = J_grad.at[jnp.diag_indices(n_sites)].set(0.0)
    J_sym = (J_grad + J_grad.transpose(1, 0, 3, 2)) / 2.0

    return value, h_grad, J_sym

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    _, h_grad, J_grad = value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    return h_grad, J_grad