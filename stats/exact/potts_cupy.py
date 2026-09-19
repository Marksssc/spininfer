from functools import lru_cache
import cupy as cp

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    num_configs = n_states ** n_sites
    idx = cp.arange(num_configs, dtype=cp.int64)[:, None]
    exponents = cp.arange(n_sites - 1, -1, -1, dtype=cp.int64)[None, :]
    powers = n_states ** exponents
    return (idx // powers) % n_states

def _exact_distribution(h, J):
    n_sites, n_states = h.shape
    all_states = _enumerate_all_states(n_sites, n_states)
    one_hot = cp.eye(n_states)[all_states]

    e_h = cp.einsum('nia,ia->n', one_hot, h)
    e_J = cp.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    energies = e_h + e_J

    max_e = energies.max()
    weights = cp.exp(energies - max_e)
    Z = cp.sum(weights)
    probs = weights / Z
    log_Z = cp.log(Z) + max_e

    return one_hot, probs, energies, log_Z

def get_exact_statistics(h, J):
    one_hot, probs, _, _ = _exact_distribution(h, J)

    mean_s = cp.einsum('n,nia->ia', probs, one_hot)
    mean_ss = cp.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    return mean_s, mean_ss

def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = -cp.sum(probs * energies)
    entropy = -cp.sum(cp.where(probs > 0, probs * cp.log(probs), 0.0))
    mean_e2 = cp.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity