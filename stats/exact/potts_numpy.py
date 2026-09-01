from functools import lru_cache
import numpy as np
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    states = np.array(list(itertools.product(np.arange(n_states), repeat=n_sites)))
    return states

def get_exact_statistics(h, J):
    n_sites, n_states = h.shape
    all_states = _enumerate_all_states(n_sites, n_states) 
    one_hot = np.eye(n_states)[all_states]

    e_h = np.einsum('nia,ia->n', one_hot, h)
    e_J = np.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    total_energy = e_h + e_J 

    max_e = total_energy.max()
    weights = np.exp(total_energy - max_e)
    Z = np.sum(weights)
    probs = weights / Z 

    mean_s = np.einsum('n,nia->ia', probs, one_hot)
    mean_ss = np.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    
    return mean_s, mean_ss