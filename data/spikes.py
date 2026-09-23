from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

from data.dataset import Dataset
from models import Model

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a pickled pandas DataFrame from `file_path`."""
    return pd.read_pickle(file_path)


def select_subset(df: pd.DataFrame, column: str, value, reset_index: bool = True) -> pd.DataFrame:
    """Return rows of `df` where `column == value`, optionally resetting the index."""
    out = df[df[column] == value]
    if reset_index:
        out = out.reset_index(drop=True)
    return out


def spikes_to_ising(spike_indices: Sequence[np.ndarray], n_timepoints: int) -> np.ndarray:
    """Build a (n_timepoints, n_sites) +-1 array: +1 at each site's spike times, -1 elsewhere."""
    n_sites = len(spike_indices)
    out = -np.ones((n_timepoints, n_sites))
    for site, times in enumerate(spike_indices):
        out[np.asarray(times, dtype=int), site] = 1.0
    return out


def spikes_to_potts(simple_spike_indices: Sequence[np.ndarray],
                     complex_spike_indices: Sequence[np.ndarray],
                     n_timepoints: int) -> np.ndarray:
    """Build a (n_timepoints, n_sites) {0,1,2} array: 1 at simple-spike times, 2 at complex-spike times, 0 otherwise."""
    n_sites = len(simple_spike_indices)
    out = np.zeros((n_timepoints, n_sites), dtype=np.int64)
    for site, times in enumerate(simple_spike_indices):
        out[np.asarray(times, dtype=int), site] = 1
    for site, times in enumerate(complex_spike_indices):
        out[np.asarray(times, dtype=int), site] = 2
    return out


def _windows(samples: np.ndarray, bin_width: int) -> np.ndarray:
    """Reshape `samples` into non-overlapping (n_bins, bin_width, n_sites) windows, dropping any leftover timepoints."""
    n_timepoints, n_sites = samples.shape
    n_bins = n_timepoints // bin_width
    return samples[: n_bins * bin_width].reshape(n_bins, bin_width, n_sites)


def bin_ising(samples: np.ndarray, bin_width: int) -> np.ndarray:
    """Bin +-1 samples into +-1 per window: +1 if any spike occurred in the window."""
    spiked = _windows(samples, bin_width) == 1
    return spiked.any(axis=1) * 2 - 1


def spike_counts(samples: np.ndarray, bin_width: int) -> np.ndarray:
    """Count spikes per site within each bin_width window."""
    spiked = _windows(samples, bin_width) == 1
    return spiked.sum(axis=1)


def bin_potts(samples: np.ndarray, bin_width: int) -> np.ndarray:
    """Bin {0,1,2} samples per window, preferring complex (2) over simple (1) over none (0)."""
    windows = _windows(samples, bin_width)
    has_complex = (windows == 2).any(axis=1)
    has_simple = (windows == 1).any(axis=1)
    return np.where(has_complex, 2, np.where(has_simple, 1, 0)).astype(np.int32)


def ising_dataset_from_spikes(model: Model, spike_indices: Sequence[np.ndarray],
                               n_timepoints: int, bin_width: int | None = None) -> Dataset:
    """Convert spike times to +-1 samples (optionally binned) and wrap them as a Dataset for `model`."""
    samples = spikes_to_ising(spike_indices, n_timepoints)
    if bin_width is not None:
        samples = bin_ising(samples, bin_width)
    return Dataset(samples=samples, model=model)


def potts_dataset_from_spikes(model: Model, simple_spike_indices: Sequence[np.ndarray],
                               complex_spike_indices: Sequence[np.ndarray],
                               n_timepoints: int, bin_width: int | None = None) -> Dataset:
    """Convert simple/complex spike times to {0,1,2} samples (optionally binned) and wrap them as a Dataset for `model`."""
    samples = spikes_to_potts(simple_spike_indices, complex_spike_indices, n_timepoints)
    if bin_width is not None:
        samples = bin_potts(samples, bin_width)
    return Dataset(samples=samples, model=model)
