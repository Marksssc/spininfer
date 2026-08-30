import numpy as np
import pytest
from models.ising import IsingModel
from models.potts import PottsModel
from data.dataset import Dataset

def test_dataset_rejects_mismatched_site_count():
    model = IsingModel(n_sites=10, backend="numpy")
    wrong_samples = np.ones((100, 7))

    with pytest.raises(ValueError):
        Dataset(samples=wrong_samples, model=model)

def test_dataset_rejects_mismatched_site_count_potts():
    model = PottsModel(n_sites=10, n_states=3, backend="numpy")
    wrong_samples = np.zeros((100, 7), dtype=int) 

    with pytest.raises(ValueError):
        Dataset(samples=wrong_samples, model=model)

def test_dataset_accepts_matching_site_count():
    model = IsingModel(n_sites=10, backend="numpy")
    samples = np.random.default_rng(0).choice([-1, 1], size=(100, 10)).astype(np.float64)

    dataset = Dataset(samples=samples, model=model)
    assert dataset.moments is not None

def test_dataset_accepts_matching_site_count_potts():
    model = PottsModel(n_sites=10, n_states=3, backend="numpy")
    samples = np.random.default_rng(0).integers(0, model.n_states, size=(100, 10))

    dataset = Dataset(samples=samples, model=model)
    assert dataset.moments is not None

def test_dataset_moments_match_direct_computation():
    model = IsingModel(n_sites=5, backend="numpy")
    samples = np.random.default_rng(0).choice([-1, 1], size=(200, 5)).astype(np.float64)

    dataset = Dataset(samples=samples, model=model)
    expected = model.compute_moments(samples)

    assert np.allclose(dataset.moments.mean_s, expected.mean_s)
    assert np.allclose(dataset.moments.mean_ss, expected.mean_ss)

def test_dataset_moments_match_direct_computation_potts():
    model = PottsModel(n_sites=5, n_states=3, backend="numpy")
    samples = np.random.default_rng(0).integers(0, model.n_states, size=(200, 5))

    dataset = Dataset(samples=samples, model=model)
    expected = model.compute_moments(samples)

    assert np.allclose(dataset.moments.mean_s, expected.mean_s)
    assert np.allclose(dataset.moments.mean_ss, expected.mean_ss)

def test_compute_moments_matches_hand_computation():
    model = IsingModel(n_sites=3, backend="numpy")
    samples = np.array([
        [ 1,  1, -1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [ 1,  1,  1],
    ], dtype=np.float64)

    moments = model.compute_moments(samples)
    expected_mean_s = np.array([0.5, 0.5, 0.0])
    expected_mean_ss = samples.T @ samples /4

    assert np.allclose(moments.mean_s, expected_mean_s)
    assert np.allclose(moments.mean_ss, expected_mean_ss)
    assert np.isclose(moments.mean_ss[0, 0], 1.0)

def test_compute_moments_matches_hand_computation_potts():
    model = PottsModel(n_sites=3, n_states=3, backend="numpy")
    samples = np.array([
        [0, 1, 2],
        [0, 0, 0],
        [0, 1, 1],
        [1, 2, 1],
    ])

    moments = model.compute_moments(samples)

    expected_mean_s = np.array([
        [0.75, 0.25, 0.0],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
    ])

    site1_site2_ss = np.array([
            [0.25, 0.5, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0],
        ])
    
    site1_site3_ss = np.array([
        [0.25, 0.25, 0.25],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.0],
    ])

    site2_site3_ss = np.array([
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.25],
            [0.0, 0.25, 0.0],
    ])

    assert np.allclose(moments.mean_s, expected_mean_s)
    assert np.allclose(moments.mean_ss[0, 1, :, :], site1_site2_ss)
    assert np.allclose(moments.mean_ss[0, 2, :, :], site1_site3_ss)
    assert np.allclose(moments.mean_ss[1, 2, :, :], site2_site3_ss)

    for site in range(3):
        for state in range(3):
            assert np.isclose(moments.mean_ss[site, site, state, state], expected_mean_s[site, state])

def test_dataset_moments_are_cached_not_recomputed(monkeypatch):
    model = IsingModel(n_sites=5, backend="numpy")
    samples = np.random.default_rng(0).choice([-1, 1], size=(200, 5)).astype(np.float64)

    call_count = {"n": 0}
    original = model.compute_moments
    def counting_compute_moments(s):
        call_count["n"] += 1
        return original(s)
    monkeypatch.setattr(model, "compute_moments", counting_compute_moments)

    dataset = Dataset(samples=samples, model=model)
    _ = dataset.moments
    _ = dataset.moments
    _ = dataset.moments

    assert call_count["n"] == 1