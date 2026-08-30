import numpy as np
from numba import njit, prange

@njit(inline='always', fastmath=True)
def get_energy_dif(h, J, spin, site, newstate, sites):
    delta_E = 0
    current_spin = spin[site]
    delta_E = -h[site, newstate] - (-h[site, current_spin])
    for ii in range(sites):
        if ii != site:
            delta_E += -J[site, ii, newstate, spin[ii]] - (-J[site, ii, current_spin, spin[ii]])
    return delta_E


@njit(parallel=True, fastmath=True)
def simulate(h, J, samples, iterations=1000, seed=0):
    np.random.seed(seed)
    # Get the initial random lattice
    sites, states = np.shape(h)
    initial_lattice = np.random.randint(0, states, size=(samples, sites))

    for ii in prange(samples):
        lattice_row = initial_lattice[ii, :]
        for jj in range(iterations):
            # Choosing which spin to flip and what that value is
            spinflip = np.random.randint(sites)
            current_state = lattice_row[spinflip]

            # Getting new thing and making sure it is not the same
            shift = np.random.randint(1, states)
            new_state = (current_state + shift) % states

            # Get the energy change
            energy_change = get_energy_dif(h, J, lattice_row, spinflip, new_state, sites)

            # Calculating whether to change my values
            if energy_change <= 0 or np.random.random() < np.exp(-energy_change):
                lattice_row[spinflip] = new_state
    return initial_lattice