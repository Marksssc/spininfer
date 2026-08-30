from functools import lru_cache
import cupy as cp
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int):
    states = cp.array(list(itertools.product([-1.0, 1.0], repeat=n_sites)))
    return states

def get_exact_statistics(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites) 

    energies = - cp.sum(h * all_states, axis=1) - 0.5 * cp.einsum('ni,ij,nj->n', all_states, J, all_states)
    weights = cp.exp(-energies)
    Z = cp.sum(weights)
    probs = weights / Z

    mean_s = cp.sum(probs[:, None] * all_states, axis=0)
    mean_ss = cp.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss