"""Cross-modal audit, tested on data whose alignment is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from vla_label_audit.crossmodal import (
    cca_alignment,
    effective_rank,
    gaussian_mi_from_cca,
    instruction_space_report,
    knn_indices,
    mutual_information_ksg,
    neighborhood_disagreement,
    neighborhood_overlap,
    normalize,
    rank_correlation_across_views,
)
from vla_label_audit.noise import fit_degradation_curve, inject_label_noise, predicted_cost


def aligned_views(n=200, d=16, noise=0.05, seed=0):
    """Two views generated from one shared latent: alignment is real."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 6))
    a = z @ rng.normal(size=(6, d)) + noise * rng.normal(size=(n, d))
    b = z @ rng.normal(size=(6, d)) + noise * rng.normal(size=(n, d))
    return a, b


def unrelated_views(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(n, d))


def test_knn_excludes_self_and_returns_sorted_neighbours():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8))
    nn = knn_indices(x, k=5)
    assert nn.shape == (50, 5)
    assert all(i not in nn[i] for i in range(50))
    xn = normalize(x)
    sims = np.einsum("nd,nkd->nk", xn, xn[nn])
    assert np.all(np.diff(sims, axis=1) <= 1e-9), "neighbours not in descending similarity"


def test_knn_rejects_out_of_range_k():
    x = np.random.default_rng(0).normal(size=(10, 4))
    for bad in (0, 10, 11):
        with pytest.raises(ValueError, match="out of range"):
            knn_indices(x, k=bad)


def test_neighborhood_overlap_separates_aligned_from_unrelated():
    """The headline diagnostic must actually discriminate."""
    a, b = aligned_views()
    u, v = unrelated_views()
    assert neighborhood_overlap(a, b, k=10).mean() > 0.3
    assert neighborhood_overlap(u, v, k=10).mean() < 0.05


def test_neighborhood_disagreement_flags_a_planted_mislabel():
    """Give one episode a label unlike its behavioural neighbours' labels."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(120, 5))
    vis = z @ rng.normal(size=(5, 24))
    lang = z @ rng.normal(size=(5, 24))
    lang[7] = -lang[7]  # episode 7 now says the opposite of what it did
    scores = neighborhood_disagreement(vis, lang, k=10)
    assert scores.argmax() == 7
    assert scores[7] > np.median(scores) * 2


def test_disagreement_is_near_zero_when_labels_are_consistent():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(80, 4))
    view = z @ rng.normal(size=(4, 12))
    assert neighborhood_disagreement(view, view, k=5).mean() < 0.4


def test_rank_correlation_tracks_true_alignment():
    a, b = aligned_views()
    u, v = unrelated_views()
    assert rank_correlation_across_views(a, b, sample=200) > 0.5
    assert abs(rank_correlation_across_views(u, v, sample=200)) < 0.1


def test_cca_finds_the_shared_subspace_and_not_a_phantom_one():
    a, b = aligned_views(n=400, noise=0.02)
    u, v = unrelated_views(n=400)
    strong = cca_alignment(a, b, n_components=5)
    weak = cca_alignment(u, v, n_components=5)
    assert strong.mean_top_k > 0.9
    assert strong.gaussian_mi_nats > weak.gaussian_mi_nats
    assert np.all(np.diff(strong.canonical_correlations) <= 1e-9), "correlations not sorted"
    assert np.all((strong.canonical_correlations >= 0) & (strong.canonical_correlations <= 1))


def test_cca_is_invariant_to_invertible_linear_maps():
    """Canonical correlations depend on the subspace, not on the basis.

    So an encoder that scales its outputs differently gives the same result.
    """
    rng = np.random.default_rng(3)
    a, b = aligned_views(n=300, seed=3)
    m = rng.normal(size=(a.shape[1], a.shape[1]))
    base = cca_alignment(a, b, n_components=4, reg=1e-8)
    mapped = cca_alignment(a @ m, b, n_components=4, reg=1e-8)
    assert np.allclose(base.canonical_correlations, mapped.canonical_correlations, atol=1e-3)


def test_gaussian_mi_is_monotone_and_zero_at_zero():
    assert gaussian_mi_from_cca(np.zeros(5)) == pytest.approx(0.0)
    assert gaussian_mi_from_cca(np.array([0.9])) > gaussian_mi_from_cca(np.array([0.5]))
    assert np.isfinite(gaussian_mi_from_cca(np.array([1.0])))  # must not blow up


def test_ksg_recovers_known_gaussian_mutual_information():
    """For a bivariate Gaussian, I = -0.5*log(1-rho^2). Check against truth."""
    rng = np.random.default_rng(4)
    rho = 0.8
    n = 4000
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    truth = -0.5 * np.log(1 - rho**2)
    est = mutual_information_ksg(x[:, None], y[:, None], k=5)
    assert est == pytest.approx(truth, rel=0.15)


def test_ksg_is_near_zero_for_independent_variables():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2000, 1))
    y = rng.normal(size=(2000, 1))
    assert mutual_information_ksg(x, y, k=5) < 0.05


def test_effective_rank_recovers_a_planted_rank():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(500, 4))
    x = z @ rng.normal(size=(4, 32))
    assert effective_rank(x) == pytest.approx(4.0, abs=1.0)
    assert effective_rank(rng.normal(size=(500, 32))) > 20


def test_instruction_space_report_detects_template_collapse():
    """Many strings, few real directions: the paraphrase-heavy corpus signature."""
    rng = np.random.default_rng(7)
    templates = rng.normal(size=(3, 20))
    emb = np.repeat(templates, 100, axis=0) + 0.01 * rng.normal(size=(300, 20))
    texts = [f"instruction {i}" for i in range(300)]
    rep = instruction_space_report(emb, texts)
    assert rep["unique_fraction"] == 1.0            # every string differs
    assert rep["effective_rank"] < 6                # but only ~3 real directions
    assert rep["mean_nearest_neighbor_similarity"] > 0.95
    assert rep["n_annotations"] == 300


def test_instruction_report_rejects_misaligned_texts():
    with pytest.raises(ValueError, match="align"):
        instruction_space_report(np.eye(4), ["a", "b"])


def test_noise_injection_hits_the_requested_rate():
    labels = np.arange(1000)
    for mode in ("shuffle", "swap"):
        res = inject_label_noise(labels, 0.3, mode=mode, seed=0)
        assert res.realised_rate == pytest.approx(0.3, abs=0.02), mode


def test_zero_noise_changes_nothing_and_full_shuffle_changes_most():
    labels = np.arange(500)
    assert inject_label_noise(labels, 0.0).realised_rate == 0.0
    assert inject_label_noise(labels, 1.0, mode="shuffle", seed=1).realised_rate > 0.95


def test_swap_preserves_the_label_multiset():
    """Swap must not create or destroy labels, only relocate them."""
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 10, size=400)
    res = inject_label_noise(labels, 0.5, mode="swap", seed=2)
    assert np.array_equal(np.sort(labels), np.sort(res.labels))


def test_paraphrase_swaps_in_semantically_close_labels():
    rng = np.random.default_rng(9)
    emb = np.repeat(rng.normal(size=(20, 8)), 5, axis=0) + 0.01 * rng.normal(size=(100, 8))
    labels = np.arange(100)
    res = inject_label_noise(labels, 0.4, mode="paraphrase", embeddings=emb, seed=3)
    changed = res.corrupted_idx
    assert changed.size > 0
    # Replacements should come from the same tight cluster (same block of 5).
    assert np.mean(res.labels[changed] // 5 == changed // 5) > 0.9


def test_paraphrase_requires_embeddings_and_modes_are_validated():
    with pytest.raises(ValueError, match="embeddings"):
        inject_label_noise(np.arange(10), 0.5, mode="paraphrase")
    with pytest.raises(ValueError, match="unknown mode"):
        inject_label_noise(np.arange(10), 0.5, mode="nonsense")
    with pytest.raises(ValueError, match="rate"):
        inject_label_noise(np.arange(10), 1.5)


def test_degradation_curve_recovers_a_planted_slope():
    rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    scores = 0.70 - 0.6 * rates
    curve = fit_degradation_curve(rates, scores)
    assert curve["slope"] == pytest.approx(-0.6, abs=1e-6)
    assert curve["intercept"] == pytest.approx(0.70, abs=1e-6)
    assert predicted_cost(curve, 0.25) == pytest.approx(0.15, abs=1e-6)


def test_degradation_curve_needs_enough_levels():
    with pytest.raises(ValueError, match=">= 3"):
        fit_degradation_curve(np.array([0.0, 0.5]), np.array([0.7, 0.4]))
