from functools import lru_cache
import jax.numpy as jnp
import jax
import itertools

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int):
    n_configs = 1 << n_sites
    idx = jnp.arange(n_configs, dtype=jnp.int64)[:, None]
    shifts = jnp.arange(n_sites, dtype=jnp.int64)[None, :]
    bits = (idx >> shifts) & 1
    return bits.astype(jnp.float64) * 2 - 1

def _state_energy(h, J, state):
    return -jnp.dot(h, state) - 0.5 * jnp.dot(state, jnp.dot(J, state))


def _exact_distribution(h, J):
    n_sites = h.shape[0]
    all_states = _enumerate_all_states(n_sites) 

    energies = jax.vmap(_state_energy, in_axes=(None, None, 0))(h, J, all_states)
    exp = -energies
    max_exp = exp.max()
    weights = jnp.exp(exp - max_exp)

    Z = jnp.sum(weights)
    probs = weights / Z
    log_Z = jnp.log(Z) + max_exp
    return all_states, probs, energies, log_Z


def get_exact_statistics(h, J):
    all_states, probs, _, _ = _exact_distribution(h, J)

    mean_s = jnp.sum(probs[:, None] * all_states, axis=0)
    mean_ss = jnp.einsum('n,ni,nj->ij', probs, all_states, all_states)
    return mean_s, mean_ss

def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = jnp.sum(probs * energies)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))
    mean_e2 = jnp.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity