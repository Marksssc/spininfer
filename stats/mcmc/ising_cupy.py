import cupy as cp

def delta_energy(h, J, data, site_change):
    n_chains = data.shape[0]
    chain_idx = cp.arange(n_chains)

    s_i = data[chain_idx, site_change]
    delta_mag_energy = s_i * h[site_change]

    delta_cor_energy = s_i * cp.einsum('ij,ij->i', J[site_change], data)

    return 2 * (delta_mag_energy + delta_cor_energy)


def simulate(h, J, samples, iterations=1000, seed=0):
    cp.random.seed(seed)
    sites = len(h)
    lattice = (cp.random.randint(0, 2, size=(samples, sites)) * 2 - 1).astype(cp.float64)
    chain_idx = cp.arange(samples)

    for ii in range(iterations):
        change_sites = cp.random.randint(0, sites, size=samples)
        energy_dif = delta_energy(h, J, lattice, change_sites)
        mask = (energy_dif <= 0.0) | (cp.random.random(samples) < cp.exp(-energy_dif))

        accepted_chains = chain_idx[mask]
        accepted_sites = change_sites[mask]
        lattice[accepted_chains, accepted_sites] *= -1

    return lattice