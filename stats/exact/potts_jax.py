import jax
import jax.numpy as jnp
import itertools
from functools import lru_cache

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    states = list(itertools.product(range(n_states), repeat=n_sites))
    return jnp.array(states)

def get_exact_statistics(h, J):
    n_sites, n_states = h.shape
    
    all_states = _enumerate_all_states(n_sites, n_states) 
    one_hot = jnp.eye(n_states, dtype=h.dtype)[all_states]
    
    e_h = jnp.einsum('nia,ia->n', one_hot, h)
    e_J = jnp.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    energies = e_h + e_J
    
    max_e = energies.max()
    weights = jnp.exp(energies - max_e)
    Z = jnp.sum(weights)
    probs = weights / Z
    
    mean_s = jnp.einsum('n,nia->ia', probs, one_hot)
    mean_ss = jnp.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    
    return mean_s, mean_ss