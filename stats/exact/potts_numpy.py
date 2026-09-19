from functools import lru_cache
import numpy as np
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    states = np.array(list(itertools.product(np.arange(n_states), repeat=n_sites)))
    return states

def _exact_distribution(h, J):
    n_sites, n_states = h.shape
    all_states = _enumerate_all_states(n_sites, n_states)
    one_hot = np.eye(n_states)[all_states]

    e_h = np.einsum('nia,ia->n', one_hot, h)
    e_J = np.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    energies = e_h + e_J

    max_e = energies.max()
    weights = np.exp(energies - max_e)
    Z = np.sum(weights)
    probs = weights / Z
    log_Z = np.log(Z) + max_e

    return one_hot, probs, energies, log_Z

def get_exact_statistics(h, J):
    one_hot, probs, _, _ = _exact_distribution(h, J)

    mean_s = np.einsum('n,nia->ia', probs, one_hot)
    mean_ss = np.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    return mean_s, mean_ss

def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = -np.sum(probs * energies)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
    mean_e2 = np.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity
