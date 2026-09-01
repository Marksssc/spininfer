import numpy as np

def get_energy_dif(h, J, initial_lattice, chain_idx, sites_to_flip, flip_to):
    old_states = initial_lattice[chain_idx, sites_to_flip]
    mag_dif = h[sites_to_flip, flip_to] - h[sites_to_flip, old_states]

    J_diff = J[sites_to_flip, :, flip_to, :] - J[sites_to_flip, :, old_states, :]

    samples_idx = np.arange(len(chain_idx))[:, None]
    sites_idx = np.arange(J.shape[0])[None, :]

    int_dif = J_diff[samples_idx, sites_idx, initial_lattice].sum(axis=1)

    return mag_dif + int_dif


def simulate(h, J, samples, iterations=1000, seed=0):
    np.random.seed(seed)
    # Get the initial random lattice
    sites, states = np.shape(h)
    initial_lattice = np.random.randint(0, states, size=(samples, sites))

    for ii in range(samples):
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