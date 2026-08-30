import jax
import jax.numpy as jnp
from jax import random
from functools import partial

def _metropolis_step(carry, key):
    lattice, h, J = carry
    site_key, accept_key = random.split(key)

    site = random.randint(site_key, (), 0, h.shape[0])
    local_field = jnp.dot(J[site], lattice) + h[site]
    delta_E = 2.0 * lattice[site] * local_field
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta_E))
    accept = random.uniform(accept_key) < accept_prob

    new_spin = jnp.where(accept, -lattice[site], lattice[site])
    lattice = lattice.at[site].set(new_spin)
    return (lattice, h, J), None

def _simulate_one_chain(h, J, key, iterations):
    n_sites = h.shape[0]
    init_key, run_key = random.split(key)
    lattice = random.choice(init_key, jnp.array([-1.0, 1.0]), shape=(n_sites,))

    step_keys = random.split(run_key, iterations)
    (final_lattice, _, _), _ = jax.lax.scan(_metropolis_step, (lattice, h, J), step_keys)
    return final_lattice

@partial(jax.jit, static_argnames=['samples', 'iterations'])
def simulate(h, J, samples, iterations=1000, key=None):
    if key is None:
        key = random.PRNGKey(0)
    chain_keys = random.split(key, samples)

    batched_simulate = jax.vmap(_simulate_one_chain, in_axes=(None, None, 0, None))
    return batched_simulate(h, J, chain_keys, iterations)