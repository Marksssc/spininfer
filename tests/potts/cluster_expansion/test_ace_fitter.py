import numpy as np

from spininfer.models.potts import PottsModel
from spininfer.data.dataset import Dataset
from spininfer.stats.moments import Moments
from spininfer.convergence.criteria import FitResult
from spininfer.cluster_expansion._cluster_fit import fit_cluster
from spininfer.fitters.ACE_fitter import ACEFitter


def _exact_dataset(backend, n_sites=5, n_states=3, seed=0):
    model = PottsModel(n_sites=n_sites, n_states=n_states, backend=backend)
    h_true, J_true = model.random_params(seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    dataset = Dataset(model=model, moments=Moments(mean_s=mean_s, mean_ss=mean_ss))
    return model, dataset


def test_zero_threshold_reproduces_full_fit(backend, to_numpy):
    model, dataset = _exact_dataset(backend)
    result = ACEFitter(model, dataset, threshold=0.0, regularizer=None, max_size=model.n_sites).fit()
    full = fit_cluster(model, dataset, sites=range(model.n_sites))

    assert np.allclose(to_numpy(result.h), to_numpy(full.h), atol=1e-8)
    assert np.allclose(to_numpy(result.J), to_numpy(full.J), atol=1e-8)
    assert np.isclose(to_numpy(result.entropy), to_numpy(full.entropy), atol=1e-8)


def test_huge_threshold_gives_independent_sites(backend, to_numpy):
    model, dataset = _exact_dataset(backend)
    result = ACEFitter(model, dataset, threshold=1e6, regularizer=None).fit()
    log_p = np.log(dataset.moments.mean_s)


    assert result.kept[1] == {(i,) for i in range(model.n_sites)}
    assert not result.kept.get(2)
    assert np.allclose(to_numpy(result.J), 0.0)
    assert np.allclose(to_numpy(result.h), to_numpy(log_p) - to_numpy(log_p).mean(axis=1, keepdims=True), atol=1e-5)



def test_max_size_limits_cluster_size(backend):
    model, dataset = _exact_dataset(backend)
    result = ACEFitter(model, dataset, threshold=0.0, regularizer=None, max_size=2).fit()

    assert max(result.kept) == 2
    assert all(len(cluster) <= 2 for cluster in result.delta_S)


def test_positive_threshold_keeps_fewer_clusters(backend):
    model, dataset = _exact_dataset(backend)
    loose = ACEFitter(model, dataset, threshold=0.0, regularizer=None).fit()
    strict = ACEFitter(model, dataset, threshold=0.01, regularizer=None).fit()

    n_loose = sum(len(level) for level in loose.kept.values())
    n_strict = sum(len(level) for level in strict.kept.values())
    assert n_strict < n_loose
    kept_strict = set().union(*strict.kept.values())
    assert all(abs(strict.delta_S[c]) > 0.01 for c in kept_strict if len(c) > 1)


def test_result_is_a_fit_result(backend, to_numpy):
    model, dataset = _exact_dataset(backend)
    result = ACEFitter(model, dataset, threshold=0.01).fit()

    assert isinstance(result, FitResult)
    assert result.h.shape == (model.n_sites, model.n_states)
    assert result.J.shape == (model.n_sites, model.n_sites, model.n_states, model.n_states)
    assert np.allclose(to_numpy(result.J), np.transpose(to_numpy(result.J), (1, 0, 3, 2)))
