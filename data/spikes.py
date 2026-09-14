from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

from data.dataset import Dataset

def load_dataframe(file_path: str) -> pd.DataFrame:
    return pd.read_pickle(file_path)


def select_subset(df: pd.DataFrame, column: str, value, reset_index: bool = True) -> pd.DataFrame:
    out = df[df[column] == value]
    if reset_index:
        out = out.reset_index(drop=True)
    return out


def spikes_to_ising(spike_indices: Sequence[np.ndarray], n_timepoints: int) -> np.ndarray:
    n_sites = len(spike_indices)
    out = -np.ones((n_timepoints, n_sites))
    for site, times in enumerate(spike_indices):
        out[np.asarray(times, dtype=int), site] = 1.0
    return out


def spikes_to_potts(simple_spike_indices: Sequence[np.ndarray],
                     complex_spike_indices: Sequence[np.ndarray],
                     n_timepoints: int) -> np.ndarray:
    n_sites = len(simple_spike_indices)
    out = np.zeros((n_timepoints, n_sites), dtype=np.int64)
    for site, times in enumerate(simple_spike_indices):
        out[np.asarray(times, dtype=int), site] = 1
    for site, times in enumerate(complex_spike_indices):
        out[np.asarray(times, dtype=int), site] = 2
    return out


def _windows(samples: np.ndarray, bin_width: int) -> np.ndarray:
    n_timepoints, n_sites = samples.shape
    n_bins = n_timepoints // bin_width
    return samples[: n_bins * bin_width].reshape(n_bins, bin_width, n_sites)


def bin_ising(samples: np.ndarray, bin_width: int) -> np.ndarray:
    spiked = _windows(samples, bin_width) == 1
    return spiked.any(axis=1) * 2 - 1


def spike_counts(samples: np.ndarray, bin_width: int) -> np.ndarray:
    spiked = _windows(samples, bin_width) == 1
    return spiked.sum(axis=1)


def bin_potts(samples: np.ndarray, bin_width: int) -> np.ndarray:
    windows = _windows(samples, bin_width)
    has_complex = (windows == 2).any(axis=1)
    has_simple = (windows == 1).any(axis=1)
    return np.where(has_complex, 2, np.where(has_simple, 1, 0)).astype(np.int32)


def ising_dataset_from_spikes(model, spike_indices: Sequence[np.ndarray],
                               n_timepoints: int, bin_width: int | None = None) -> Dataset:
    samples = spikes_to_ising(spike_indices, n_timepoints)
    if bin_width is not None:
        samples = bin_ising(samples, bin_width)
    return Dataset(samples=samples, model=model)


def potts_dataset_from_spikes(model, simple_spike_indices: Sequence[np.ndarray],
                               complex_spike_indices: Sequence[np.ndarray],
                               n_timepoints: int, bin_width: int | None = None) -> Dataset:
    samples = spikes_to_potts(simple_spike_indices, complex_spike_indices, n_timepoints)
    if bin_width is not None:
        samples = bin_potts(samples, bin_width)
    return Dataset(samples=samples, model=model)
