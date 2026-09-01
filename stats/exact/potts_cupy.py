from functools import lru_cache
import cupy as cp
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    states = cp.array(list(itertools.product(range(n_states), repeat=n_sites)))
    return states

def get_exact_statistics(h, J):
    n_sites, n_states = h.shape
    all_states = _enumerate_all_states(n_sites, n_states) 
    one_hot = cp.eye(n_states)[all_states]

    e_h = cp.einsum('nia,ia->n', one_hot, h)
    e_J = cp.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    total_energy = e_h + e_J 

    max_e = total_energy.max()
    weights = cp.exp(total_energy - max_e)
    Z = cp.sum(weights)
    probs = weights / Z 

    mean_s = cp.einsum('n,nia->ia', probs, one_hot)
    mean_ss = cp.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    
    return mean_s, mean_ss