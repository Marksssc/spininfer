from functools import lru_cache
import numpy as np
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int):
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n_sites)))
    return states

def _exact_distribution(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites)

    energies = np.sum(h * all_states, axis=1) + 0.5 * np.einsum('ni,ij,nj->n', all_states, J, all_states)
    max_exp = energies.max()
    weights = np.exp(energies - max_exp)

    Z = np.sum(weights)
    probs = weights / Z
    log_Z = np.log(Z) + max_exp
    return all_states, probs, energies, log_Z

def get_exact_statistics(h, J):
    all_states, probs, _, _ = _exact_distribution(h, J)

    mean_s = np.sum(probs[:, None] * all_states, axis=0)
    mean_ss = np.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss

def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = -np.sum(probs * energies)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
    mean_e2 = np.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity