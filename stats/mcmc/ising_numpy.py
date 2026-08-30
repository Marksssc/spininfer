import numpy as np

def delta_energy(h, J, spins, site_change):
    s_i = spins[site_change]
    delta_mag_energy = s_i * h[site_change]
    delta_cor_energy = s_i*np.dot(J[site_change, :], spins)
    return 2*(delta_mag_energy + delta_cor_energy)

def simulate(h, J, samples, iterations=1000, seed=0):
    np.random.seed(seed)
    sites = len(h)
    lattice = (np.random.randint(0, 2, size=(samples, sites))*2 - 1).astype(np.float64)

    for ii in range(samples):
        lattice_row = lattice[ii, :]
        local_field = h + (J@lattice_row)
        for jj in range(iterations):
            site_to_flip = np.random.randint(sites)
            s_i = lattice_row[site_to_flip]

            energy_dif = 2 * s_i * local_field[site_to_flip]
            if energy_dif <= 0 or np.random.random() < np.exp(-energy_dif):
                lattice_row[site_to_flip] *= -1
                local_field -= 2 * s_i * J[site_to_flip, :]

    return lattice