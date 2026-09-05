from functools import partial

import jax
import jax.numpy as jnp

def get_energy_dif(h, J, lattice, chain_idx, sites_to_flip, flip_to, samples_idx, sites_idx):
    old_states = lattice[chain_idx, sites_to_flip]
    mag_dif = h[sites_to_flip, flip_to] - h[sites_to_flip, old_states]

    J_diff = J[sites_to_flip, :, flip_to, :] - J[sites_to_flip, :, old_states, :]

    int_dif = J_diff[samples_idx, sites_idx, lattice].sum(axis=1)

    return mag_dif + int_dif

@partial(jax.jit, static_argnames=("samples", "iterations"))
def simulate(h, J, samples, iterations=1000, seed=0):
    sites, states = h.shape
    
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    lattice = jax.random.randint(subkey, shape=(samples, sites), minval=0, maxval=states)
    chain_idx = jnp.arange(samples)

    samples_idx = jnp.arange(len(chain_idx))[:, None]
    sites_idx = jnp.arange(J.shape[0])[None, :]

    def _step(carry, _):
        lattice, key = carry

        key, k_site, k_shift, k_accept = jax.random.split(key, 4)

        sites_to_flip = jax.random.randint(k_site, shape=(samples,), minval=0, maxval=sites)
        shift = jax.random.randint(k_shift, shape=(samples,), minval=1, maxval=states)

        old_states = lattice[chain_idx, sites_to_flip]
        flip_to = (old_states + shift) % states

        energy_dif = -get_energy_dif(h, J, lattice, chain_idx, sites_to_flip, flip_to, samples_idx, sites_idx)
        
        p_accept = jax.random.uniform(k_accept, shape=(samples,))
        mask = (energy_dif <= 0.0) | (p_accept < jnp.exp(-energy_dif))
        
        new_states = jnp.where(mask, flip_to, old_states)
        lattice = lattice.at[chain_idx, sites_to_flip].set(new_states)
        
        return (lattice, key), None

    (final_lattice, _), _ = jax.lax.scan(_step, (lattice, key), None, length=iterations)
    
    return final_lattice