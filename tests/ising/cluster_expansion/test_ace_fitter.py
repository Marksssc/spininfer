import numpy as np

from spininfer.models.ising import IsingModel
from spininfer.data.dataset import Dataset
from spininfer.stats.moments import Moments
from spininfer.convergence.criteria import FitResult
from spininfer.cluster_expansion._cluster_fit import fit_cluster
from spininfer.fitters.ACE_fitter import ACEfitter


def _exact_dataset(n_sites=5, seed=0):
    model = IsingModel(n_sites=n_sites, backend="numpy")
    h_true, J_true = model.random_params(seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    dataset = Dataset(model=model, moments=Moments(mean_s=mean_s, mean_ss=mean_ss))
    return model, dataset


def test_zero_threshold_reproduces_full_fit():
    model, dataset = _exact_dataset()
    result = ACEfitter(model, dataset, threshold=0.0, regularizer=None, max_size=model.n_sites).fit()
    full = fit_cluster(model, dataset, sites=range(model.n_sites))

    assert result.converged
    assert {k: len(level) for k, level in result.kept.items()} == {1: 5, 2: 10, 3: 10, 4: 5, 5: 1}
    assert np.allclose(result.h, full.h, atol=1e-8)
    assert np.allclose(result.J, full.J, atol=1e-8)
    assert np.isclose(result.entropy, full.entropy, atol=1e-8)


def test_huge_threshold_gives_independent_sites():
    model, dataset = _exact_dataset()
    result = ACEfitter(model, dataset, threshold=1e6, regularizer=None).fit()

    assert result.kept[1] == {(i,) for i in range(model.n_sites)}
    assert not result.kept.get(2)
    assert np.allclose(result.J, 0.0)
    assert np.allclose(result.h, np.arctanh(dataset.moments.mean_s), atol=1e-5)


def test_max_size_limits_cluster_size():
    model, dataset = _exact_dataset()
    result = ACEfitter(model, dataset, threshold=0.0, regularizer=None, max_size=2).fit()

    assert max(result.kept) == 2
    assert all(len(cluster) <= 2 for cluster in result.delta_S)


def test_positive_threshold_keeps_fewer_clusters():
    model, dataset = _exact_dataset()
    loose = ACEfitter(model, dataset, threshold=0.0, regularizer=None).fit()
    strict = ACEfitter(model, dataset, threshold=0.01, regularizer=None).fit()

    n_loose = sum(len(level) for level in loose.kept.values())
    n_strict = sum(len(level) for level in strict.kept.values())
    assert n_strict < n_loose
    kept_strict = set().union(*strict.kept.values())
    assert all(abs(strict.delta_S[c]) > 0.01 for c in kept_strict if len(c) > 1)


def test_result_is_a_fit_result():
    model, dataset = _exact_dataset()
    result = ACEfitter(model, dataset, threshold=0.01).fit()

    assert isinstance(result, FitResult)
    assert result.h.shape == (model.n_sites,)
    assert result.J.shape == (model.n_sites, model.n_sites)
    assert np.allclose(result.J, result.J.T)
