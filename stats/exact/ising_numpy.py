from functools import lru_cache
import numpy as np
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int):
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n_sites)))
    return states

def get_exact_statistics(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites) 

    energies = - np.sum(h * all_states, axis=1) - 0.5 * np.einsum('ni,ij,nj->n', all_states, J, all_states)
    weights = np.exp(-energies)
    Z = np.sum(weights)
    probs = weights / Z

    mean_s = np.sum(probs[:, None] * all_states, axis=0)
    mean_ss = np.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss