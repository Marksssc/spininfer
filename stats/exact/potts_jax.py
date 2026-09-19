import jax
import jax.numpy as jnp
from functools import lru_cache

@lru_cache(maxsize=None)
def _enumerate_all_states(n_sites: int, n_states: int):
    num_configs = n_states ** n_sites
    idx = jnp.arange(num_configs, dtype=jnp.int64)[:, None]
    exponents = jnp.arange(n_sites - 1, -1, -1, dtype=jnp.int64)[None, :]
    powers = n_states ** exponents
    return (idx // powers) % n_states

@jax.jit
def _exact_distribution(h, J):
    n_sites, n_states = h.shape
    all_states = _enumerate_all_states(n_sites, n_states)
    one_hot = jnp.eye(n_states)[all_states]

    e_h = jnp.einsum('nia,ia->n', one_hot, h)
    e_J = jnp.einsum('nia,njb,ijab->n', one_hot, one_hot, J) / 2.0
    energies = e_h + e_J

    max_e = energies.max()
    weights = jnp.exp(energies - max_e)
    Z = jnp.sum(weights)
    probs = weights / Z
    log_Z = jnp.log(Z) + max_e

    return one_hot, probs, energies, log_Z

@jax.jit
def get_exact_statistics(h, J):
    one_hot, probs, _, _ = _exact_distribution(h, J)

    mean_s = jnp.einsum('n,nia->ia', probs, one_hot)
    mean_ss = jnp.einsum('n,nia,njb->ijab', probs, one_hot, one_hot)
    return mean_s, mean_ss

@jax.jit
def get_exact_thermodynamics(h, J):
    _, probs, energies, log_Z = _exact_distribution(h, J)

    enthalpy = -jnp.sum(probs * energies)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))
    mean_e2 = jnp.sum(probs * energies**2)
    heat_capacity = mean_e2 - enthalpy**2
    return entropy, enthalpy, heat_capacity