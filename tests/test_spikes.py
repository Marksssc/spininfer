import numpy as np

from models.ising import IsingModel
from models.potts import PottsModel
from data.spikes import (
    spikes_to_ising,
    spikes_to_potts,
    bin_ising,
    bin_potts,
    spike_counts,
    ising_dataset_from_spikes,
    potts_dataset_from_spikes,
)


def test_spikes_to_ising_shape_and_values():
    spike_indices = [np.array([0, 5]), np.array([2])]
    out = spikes_to_ising(spike_indices, n_timepoints=8)

    assert out.shape == (8, 2)
    assert np.all(np.isin(out, [-1.0, 1.0]))
    assert out[0, 0] == 1.0 and out[5, 0] == 1.0
    assert out[2, 1] == 1.0
    assert out[1, 0] == -1.0


def test_bin_ising_marks_bin_active_if_any_spike():
    samples = -np.ones((6, 1))
    samples[1, 0] = 1.0
    binned = bin_ising(samples, bin_width=3)

    assert binned.shape == (2, 1)
    assert binned[0, 0] == 1.0
    assert binned[1, 0] == -1.0


def test_spike_counts_matches_manual_count():
    samples = -np.ones((6, 1))
    samples[[0, 1, 4], 0] = 1.0
    counts = spike_counts(samples, bin_width=3)

    assert counts.tolist() == [[2], [1]]


def test_spikes_to_potts_complex_takes_precedence():
    simple = [np.array([0, 1])]
    complex_ = [np.array([1])]
    out = spikes_to_potts(simple, complex_, n_timepoints=4)

    assert out[0, 0] == 1
    assert out[1, 0] == 2
    assert out[2, 0] == 0


def test_bin_potts_precedence_within_a_window():
    samples = np.zeros((4, 1), dtype=np.int64)
    samples[0, 0] = 1
    samples[1, 0] = 2
    binned = bin_potts(samples, bin_width=2)

    assert binned.shape == (2, 1)
    assert binned[0, 0] == 2
    assert binned[1, 0] == 0


def test_ising_dataset_from_spikes_builds_valid_dataset():
    n_sites = 4
    model = IsingModel(n_sites=n_sites, backend="numpy")
    spike_indices = [np.array([0, 4]), np.array([1]), np.array([]), np.array([2, 3])]

    dataset = ising_dataset_from_spikes(model, spike_indices, n_timepoints=10, bin_width=5)

    assert dataset.samples.shape == (2, n_sites)
    assert dataset.moments.mean_s.shape == (n_sites,)


def test_potts_dataset_from_spikes_builds_valid_dataset():
    n_sites, n_states = 3, 3
    model = PottsModel(n_sites=n_sites, n_states=n_states, backend="numpy")
    simple = [np.array([0]), np.array([]), np.array([2])]
    complex_ = [np.array([]), np.array([1]), np.array([])]

    dataset = potts_dataset_from_spikes(model, simple, complex_, n_timepoints=4)

    assert dataset.samples.shape == (4, n_sites)
    assert dataset.moments.mean_s.shape == (n_sites, n_states)
