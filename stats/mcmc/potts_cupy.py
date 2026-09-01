import cupy as cp


def get_energy_dif(h, J, initial_lattice, chain_idx, sites_to_flip, flip_to):
    old_states = initial_lattice[chain_idx, sites_to_flip]
    mag_dif = h[sites_to_flip, flip_to] - h[sites_to_flip, old_states]

    J_diff = J[sites_to_flip, :, flip_to, :] - J[sites_to_flip, :, old_states, :]

    samples_idx = cp.arange(len(chain_idx))[:, None]
    sites_idx = cp.arange(J.shape[0])[None, :]

    int_dif = J_diff[samples_idx, sites_idx, initial_lattice].sum(axis=1)

    return mag_dif + int_dif


def simulate(h, J, samples, iterations=1000, seed=0):
    cp.random.seed(seed)
    # Get the initial random lattice
    sites, states = cp.shape(h)
    lattice = cp.random.randint(0, states, size=(samples, sites))
    chain_idx = cp.arange(samples)

    for ii in range(iterations):
        sites_to_flip = cp.random.randint(0, sites, size=samples)
        shift = cp.random.randint(0, states, size=samples)
        flip_to = (lattice[chain_idx, sites_to_flip] + shift) % states

        energy_dif = get_energy_dif(h, J, lattice, chain_idx, sites_to_flip, flip_to)
        mask = (energy_dif <= 0.0) | (cp.random.random(samples) < cp.exp(-energy_dif))
        
        accepted_chains = chain_idx[mask]
        accepted_sites = sites_to_flip[mask]
        lattice[accepted_chains, accepted_sites] = flip_to[mask]

    return lattice


'''def get_energy_dif(h, J, initial_lattice, chain_idx, sites_to_flip, flip_to):
    old_states = initial_lattice[chain_idx, sites_to_flip]

    mag_dif = h[sites_to_flip, flip_to]-h[sites_to_flip, old_states]
    J_dif = J[sites_to_flip, :, flip_to, :]-J[sites_to_flip, :, old_states, :]

    one_hot_lattice = cp.eye(h.shape[1])[initial_lattice]
    int_dif = cp.einsum('mjc,mjc->m', J_dif, one_hot_lattice)

    return mag_dif + int_dif'''