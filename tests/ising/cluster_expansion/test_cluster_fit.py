import numpy as np

from spininfer.models.ising import IsingModel
from spininfer.data.dataset import Dataset
from spininfer.stats.moments import Moments
from spininfer.objectives.regularization import L2Regularizer
from spininfer.cluster_expansion._cluster_fit import fit_cluster


def _exact_dataset(backend, n_sites=6, seed=0):
    model = IsingModel(n_sites=n_sites, backend=backend)
    h_true, J_true = model.random_params(seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    dataset = Dataset(model=model, moments=Moments(mean_s=mean_s, mean_ss=mean_ss))
    return model, dataset, h_true, J_true


def test_full_cluster_recovers_truth_and_entropy(backend, to_numpy):
    model, dataset, h_true, J_true = _exact_dataset(backend)
    fit = fit_cluster(model, dataset, sites=range(model.n_sites))

    assert np.allclose(to_numpy(fit.h), to_numpy(h_true), atol=1e-4)
    assert np.allclose(to_numpy(fit.J), to_numpy(J_true), atol=1e-4)
    true_entropy, _, _ = model.exact_thermodynamics(h_true, J_true)
    assert np.isclose(to_numpy(fit.entropy), to_numpy(true_entropy), atol=1e-6)


def test_sub_cluster_matches_its_moments_and_entropy(backend, to_numpy):
    model, dataset, _, _ = _exact_dataset(backend)
    sites = (1, 3, 4)
    fit = fit_cluster(model, dataset, sites=sites)

    cluster_model = IsingModel(n_sites=len(sites), backend=backend)
    mean_s, mean_ss = cluster_model.exact_statistics(fit.h, fit.J)
    idx = np.array(sites)
    assert np.allclose(to_numpy(mean_s), to_numpy(dataset.moments.mean_s[idx]), atol=1e-4)
    assert np.allclose(to_numpy(mean_ss), to_numpy(dataset.moments.mean_ss[np.ix_(idx, idx)]), atol=1e-4)

    entropy, _, _ = cluster_model.exact_thermodynamics(fit.h, fit.J) 
    assert np.isclose(to_numpy(fit.entropy), to_numpy(entropy), atol=1e-6)


def test_l2_regularization_raises_entropy_and_shrinks_couplings(backend, to_numpy):
    model, dataset, _, _ = _exact_dataset(backend)
    sites = range(model.n_sites)
    plain = fit_cluster(model, dataset, sites=sites)
    regularized = fit_cluster(model, dataset, sites=sites, regularizer=L2Regularizer(strength=0.1))

    assert regularized.entropy > plain.entropy
    assert np.linalg.norm(to_numpy(regularized.J)) < np.linalg.norm(to_numpy(plain.J))
