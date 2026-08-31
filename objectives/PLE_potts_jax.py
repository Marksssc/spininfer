import jax 
import jax.numpy as jnp

def _scalar_log_pseudolikelihood(h, J, data):
    n_states = h.shape(axis=1)
    one_hot = jnp.eye(n_states)[data]

    energies = h[None]