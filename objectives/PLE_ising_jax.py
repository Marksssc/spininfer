import jax.numpy as jnp
import jax

def _scalar_log_pseudolikelihood(h, J, data):
    local_fields = jnp.dot(data, J) + h
    log_probs = -jnp.log1p(jnp.exp(-2.0 * data * local_fields))
    measurements = data.shape[0]
    return jnp.sum(log_probs) / measurements

@jax.jit
def _value_and_grad(h, J, data):
    return jax.value_and_grad(_scalar_log_pseudolikelihood, argnums=(0, 1))(h, J, data)

def value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    value, (h_grad, J_grad) = _value_and_grad(h, J, data)

    n_sites = J_grad.shape[0]
    J_grad = J_grad.at[jnp.diag_indices(n_sites)].set(0.0)
    J_grad = (J_grad + J_grad.T) / 2.0

    return value, h_grad, J_grad

def gradient(h, J, data, empirical_mean_s, empirical_mean_ss):
    _, h_grad, J_grad = value_and_gradient(h, J, data, empirical_mean_s, empirical_mean_ss)
    return h_grad, J_grad