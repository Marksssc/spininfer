from functools import lru_cache
import cupy as cp

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites):
    n_configs = 1 << n_sites
    idx = cp.arange(n_configs, dtype=cp.int64)[:, None]
    shifts = cp.arange(n_sites, dtype=cp.int64)[None, :]
    bits = (idx >> shifts) & 1
    return bits.astype(cp.float64) * 2 - 1


def _exact_distribution(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites)

    energies = cp.sum(h * all_states, axis=1) + 0.5 * cp.einsum('ni,ij,nj->n', all_states, J, all_states)
    max_exp = energies.max()
    weights = cp.exp(energies - max_exp)

    Z = cp.sum(weights)
    probs = weights / Z
    log_Z = cp.log(Z) + max_exp
    return all_states, probs, energies, log_Z

def get_exact_statistics(h, J):
    all_states, probs, _, _ = _exact_distribution(h, J)

    mean_s = cp.sum(probs[:, None] * all_states, axis=0)
    mean_ss = cp.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss

def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = -cp.sum(probs * energies)
    entropy = -cp.sum(cp.where(probs > 0, probs * cp.log(probs), 0.0))
    mean_e2 = cp.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity