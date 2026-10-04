import numpy as np
import pytest

from cluster_expansion.expansion import next_level, compute_delta_S


def test_next_level_keeps_only_candidates_with_all_subsets_kept():
    kept = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
    assert next_level(kept, n_sites=5) == [(0, 1, 2), (1, 2, 3)]


def test_next_level_from_singletons_gives_all_pairs_in_order():
    pairs = next_level({(i,) for i in range(5)}, n_sites=5)
    assert pairs == [(i, j) for i in range(5) for j in range(i + 1, 5)]


def test_next_level_of_empty_level_is_empty():
    assert next_level(set(), n_sites=5) == []


def test_compute_delta_S_subtracts_all_proper_subsets():
    delta_S = {(0,): 0.6, (1,): 0.5, (2,): 0.4, (0, 1): -0.1, (0, 2): -0.2, (1, 2): -0.3}
    assert compute_delta_S((0,), 0.6, delta_S) == pytest.approx(0.6)
    assert compute_delta_S((0, 1, 2), 1.0, delta_S) == pytest.approx(1.0 - 1.5 + 0.6)

