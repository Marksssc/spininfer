from functools import lru_cache
import jax.numpy as jnp
import jax
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int):
    states = jnp.array(list(itertools.product([-1.0, 1.0], repeat=n_sites)))
    return states

def _state_energy(h, J, state):
    return -jnp.dot(h, state) - 0.5 * jnp.dot(state, jnp.dot(J, state))

def get_exact_statistics(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites) 

    energies = jax.vmap(_state_energy, in_axes=(None, None, 0))(h, J, all_states)
    weights = jnp.exp(-energies)
    Z = jnp.sum(weights)
    probs = weights / Z

    mean_s = jnp.sum(probs[:, None] * all_states, axis=0)
    mean_ss = jnp.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss