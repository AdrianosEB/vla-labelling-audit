"""The fast path must agree with the naive path exactly, not approximately."""

from __future__ import annotations

import numpy as np
import pytest

from vla_label_audit.agreement import (
    cosine_distance_matrix,
    exact_match_distance_matrix,
    krippendorff_alpha,
)
from vla_label_audit.scalable import alpha_nominal, alpha_semantic, bootstrap_alpha_semantic


def corpus(n_units: int, per_unit: int, dim: int = 12, seed: int = 0, spread: float = 0.4):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_units, dim))
    emb = np.repeat(base, per_unit, axis=0) + spread * rng.normal(size=(n_units * per_unit, dim))
    return np.repeat(np.arange(n_units), per_unit), emb


@pytest.mark.parametrize("n_units,per_unit,spread", [(30, 3, 0.4), (50, 2, 1.0), (12, 5, 0.05)])
def test_semantic_matches_the_naive_implementation(n_units, per_unit, spread):
    units, emb = corpus(n_units, per_unit, spread=spread, seed=n_units)
    fast = alpha_semantic(units, emb)
    slow = krippendorff_alpha(units, cosine_distance_matrix(emb))
    assert fast.alpha == pytest.approx(slow.alpha, abs=1e-10)
    assert fast.observed == pytest.approx(slow.observed, abs=1e-10)
    assert fast.expected == pytest.approx(slow.expected, abs=1e-10)
    assert (fast.n_units, fast.n_pairable) == (slow.n_units, slow.n_pairable)


def test_semantic_matches_with_ragged_group_sizes():
    """Uneven annotation counts are the realistic case and the easy thing to get wrong."""
    rng = np.random.default_rng(1)
    units = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4])  # unit 4 is a singleton
    emb = rng.normal(size=(units.size, 9))
    fast = alpha_semantic(units, emb)
    slow = krippendorff_alpha(units, cosine_distance_matrix(emb))
    assert fast.alpha == pytest.approx(slow.alpha, abs=1e-10)
    assert fast.n_units == 4 and fast.n_pairable == 11


def test_semantic_is_unaffected_by_input_scaling():
    units, emb = corpus(25, 3, seed=2)
    scaled = emb * np.linspace(0.1, 10.0, emb.shape[0])[:, None]
    assert alpha_semantic(units, emb).alpha == pytest.approx(
        alpha_semantic(units, scaled).alpha, abs=1e-10
    )


def test_semantic_handles_unsorted_unit_ids():
    """Real annotation files are not sorted; grouping must not assume it."""
    units, emb = corpus(20, 3, seed=3)
    perm = np.random.default_rng(0).permutation(units.size)
    assert alpha_semantic(units, emb).alpha == pytest.approx(
        alpha_semantic(units[perm], emb[perm]).alpha, abs=1e-10
    )


@pytest.mark.parametrize("n_units,per_unit", [(40, 3), (25, 2)])
def test_nominal_matches_the_naive_implementation(n_units, per_unit):
    rng = np.random.default_rng(n_units)
    vocab = [f"task-{i}" for i in range(6)]
    units = np.repeat(np.arange(n_units), per_unit)
    labels = np.array([vocab[i] for i in rng.integers(0, len(vocab), units.size)], dtype=object)
    fast = alpha_nominal(units, labels)
    slow = krippendorff_alpha(units, exact_match_distance_matrix(labels))
    assert fast.alpha == pytest.approx(slow.alpha, abs=1e-10)
    assert fast.observed == pytest.approx(slow.observed, abs=1e-10)


def test_nominal_perfect_and_random_extremes():
    units = np.repeat(np.arange(30), 3)
    identical = np.array([f"task-{u}" for u in units], dtype=object)
    assert alpha_nominal(units, identical).alpha == pytest.approx(1.0, abs=1e-10)

    rng = np.random.default_rng(7)
    scrambled = np.array([f"task-{i}" for i in rng.integers(0, 8, units.size)], dtype=object)
    assert abs(alpha_nominal(units, scrambled).alpha) < 0.15


def test_semantic_perfect_agreement():
    base = np.random.default_rng(4).normal(size=(25, 10))
    emb = np.repeat(base, 3, axis=0)
    assert alpha_semantic(np.repeat(np.arange(25), 3), emb).alpha == pytest.approx(1.0, abs=1e-9)


def test_scales_past_what_the_naive_version_could_hold():
    """15,000 annotations. The naive path would need a 15k x 15k matrix."""
    units, emb = corpus(5_000, 3, dim=64, seed=9, spread=0.5)
    res = alpha_semantic(units, emb)
    assert 0.0 < res.alpha < 1.0
    assert res.n_units == 5_000 and res.n_pairable == 15_000


def test_bootstrap_brackets_the_point_estimate_and_matches_it():
    units, emb = corpus(200, 3, seed=5)
    point, lo, hi = bootstrap_alpha_semantic(units, emb, n_boot=300, seed=0)
    assert point == pytest.approx(alpha_semantic(units, emb).alpha, abs=1e-10)
    assert lo <= point <= hi and hi > lo


def test_bootstrap_narrows_with_more_episodes():
    def width(n_units: int) -> float:
        units, emb = corpus(n_units, 3, seed=6)
        _, lo, hi = bootstrap_alpha_semantic(units, emb, n_boot=300, seed=1)
        return hi - lo
    assert width(600) < width(60)


def test_errors_are_explicit():
    units, emb = corpus(10, 3)
    with pytest.raises(ValueError, match="disagree on N"):
        alpha_semantic(units[:-1], emb)
    with pytest.raises(ValueError, match="zero-norm"):
        alpha_semantic(np.array([0, 0]), np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="2\\+ annotations"):
        alpha_semantic(np.array([0, 1, 2]), np.eye(3))
    with pytest.raises(ValueError, match="2\\+ annotations"):
        alpha_nominal(np.array([0, 1]), np.array(["a", "b"], dtype=object))


def test_bootstrap_memory_does_not_scale_with_episode_count():
    """Regression guard for the 59 GB bug.

    The first implementation cached one d x d Gram per episode. At 2,000
    episodes x 128 dims that is already 262 MB; at DROID scale it is 59 GB.
    This asserts peak memory stays near the size of the embeddings.
    """
    import tracemalloc

    units, emb = corpus(2_000, 3, dim=128, seed=11)
    tracemalloc.start()
    bootstrap_alpha_semantic(units, emb, n_boot=5, seed=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    embedding_bytes = emb.nbytes
    assert peak < 8 * embedding_bytes, f"peak {peak/1e6:.0f} MB vs data {embedding_bytes/1e6:.0f} MB"


def test_bootstrap_replicates_vary_but_stay_in_range():
    units, emb = corpus(150, 3, seed=12)
    point, lo, hi = bootstrap_alpha_semantic(units, emb, n_boot=200, seed=3)
    assert -1.0 <= lo <= point <= hi <= 1.0
    assert hi - lo > 1e-6, "interval collapsed; counts are probably not varying"
