import numpy as np

from models.potts import PottsModel
from data.dataset import Dataset
from stats.moments import Moments
from objectives.regularization import L2Regularizer
from cluster_expansion._cluster_fit import fit_cluster


def _exact_dataset(n_sites=6, n_states=3, seed=0):
    model = PottsModel(n_sites=n_sites, n_states=n_states, backend="numpy")
    h_true, J_true = model.random_params(seed=seed)
    mean_s, mean_ss = model.exact_statistics(h_true, J_true)
    dataset = Dataset(model=model, moments=Moments(mean_s=mean_s, mean_ss=mean_ss))
    return model, dataset, h_true, J_true


def test_full_cluster_recovers_truth_and_entropy():
    model, dataset, h_true, J_true = _exact_dataset()
    fit = fit_cluster(model, dataset, sites=range(model.n_sites))

    assert fit.converged
    assert np.allclose(fit.h, h_true, atol=1e-4)
    assert np.allclose(fit.J, J_true, atol=1e-4)
    true_entropy, _, _ = model.exact_thermodynamics(h_true, J_true)
    assert np.isclose(fit.entropy, true_entropy, atol=1e-6)


def test_sub_cluster_matches_its_moments_and_entropy():
    model, dataset, _, _ = _exact_dataset()
    sites = (1, 3, 4)
    fit = fit_cluster(model, dataset, sites=sites)

    cluster_model = PottsModel(n_sites=len(sites), n_states=3, backend="numpy")
    mean_s, mean_ss = cluster_model.exact_statistics(fit.h, fit.J)
    idx = np.array(sites)
    assert np.allclose(mean_s, dataset.moments.mean_s[idx], atol=1e-4)
    assert np.allclose(mean_ss, dataset.moments.mean_ss[np.ix_(idx, idx)], atol=1e-4)

    entropy, _, _ = cluster_model.exact_thermodynamics(fit.h, fit.J) 
    assert np.isclose(fit.entropy, entropy, atol=1e-6)


def test_l2_regularization_raises_entropy_and_shrinks_couplings():
    model, dataset, _, _ = _exact_dataset()
    sites = range(model.n_sites)
    plain = fit_cluster(model, dataset, sites=sites)
    regularized = fit_cluster(model, dataset, sites=sites, regularizer=L2Regularizer(strength=0.1))

    assert regularized.entropy > plain.entropy
    assert np.linalg.norm(regularized.J) < np.linalg.norm(plain.J)
