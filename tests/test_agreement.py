"""Agreement statistics, pinned against cases where the answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from vla_label_audit.agreement import (
    bootstrap_alpha_ci,
    cosine_distance_matrix,
    exact_match_distance_matrix,
    fleiss_kappa,
    krippendorff_alpha,
    per_unit_disagreement,
)


def _units(n_units: int, per_unit: int) -> np.ndarray:
    return np.repeat(np.arange(n_units), per_unit)


def test_perfect_agreement_gives_alpha_one():
    """Identical annotations within every unit, different across units."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 8))
    emb = np.repeat(base, 3, axis=0)
    d = cosine_distance_matrix(emb)
    res = krippendorff_alpha(_units(20, 3), d)
    assert res.alpha == pytest.approx(1.0, abs=1e-9)
    assert res.observed == pytest.approx(0.0, abs=1e-12)
    assert res.n_units == 20 and res.n_pairable == 60


def test_random_annotation_gives_alpha_near_zero():
    """Labels assigned independently of unit: alpha should sit around zero."""
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(600, 16))
    res = krippendorff_alpha(_units(200, 3), cosine_distance_matrix(emb))
    assert abs(res.alpha) < 0.05


def test_systematic_disagreement_goes_negative():
    """Annotators steered to differ *more* within units than across them.

    Two clusters; each unit gets one annotation from each. Within-unit
    disagreement is then maximal while the pool average is halved, so alpha
    must fall below zero.
    """
    rng = np.random.default_rng(2)
    a = rng.normal(size=(1, 12)) + 0.01 * rng.normal(size=(40, 12))
    b = -a[0] + 0.01 * rng.normal(size=(40, 12))
    emb = np.empty((80, 12))
    emb[0::2], emb[1::2] = a, b
    res = krippendorff_alpha(_units(40, 2), cosine_distance_matrix(emb))
    assert res.alpha < -0.5


def test_semantic_and_exact_match_disagree_on_paraphrase():
    """The gap between the two views is the paraphrase rate.

    Strings differ everywhere, so nominal alpha sees total disagreement. The
    embeddings are near-identical within units, so semantic alpha is near 1.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(30, 8))
    emb = np.repeat(base, 2, axis=0) + 0.001 * rng.normal(size=(60, 8))
    texts = [f"phrasing-{i}" for i in range(60)]  # every string unique
    units = _units(30, 2)

    semantic = krippendorff_alpha(units, cosine_distance_matrix(emb)).alpha
    nominal = krippendorff_alpha(units, exact_match_distance_matrix(texts)).alpha
    assert semantic > 0.95
    assert nominal < 0.05


def test_singleton_units_are_dropped_not_crashed():
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(7, 5))
    units = np.array([0, 0, 1, 1, 2, 3, 4])  # units 2,3,4 have one annotation
    res = krippendorff_alpha(units, cosine_distance_matrix(emb))
    assert res.n_units == 2 and res.n_pairable == 4


def test_alpha_undefined_without_any_pairable_unit():
    emb = np.eye(3)
    with pytest.raises(ValueError, match="2\\+ annotations"):
        krippendorff_alpha(np.array([0, 1, 2]), cosine_distance_matrix(emb))


def test_distance_matrix_must_be_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        krippendorff_alpha(np.array([0, 0]), np.array([[0.0, 1.0], [0.5, 0.0]]))


def test_cosine_distance_rejects_zero_vectors():
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_distance_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_fleiss_matches_the_textbook_worked_example():
    """The canonical worked example: 10 subjects, 14 raters, 5 categories.

    Published values are P_bar = 0.378, P_e = 0.213, kappa = 0.209.
    """
    table = np.array(
        [
            [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
        ]
    )
    assert fleiss_kappa(table) == pytest.approx(0.2099, abs=1e-3)


def test_fleiss_perfect_and_degenerate_cases():
    perfect = np.array([[3, 0], [0, 3], [3, 0], [0, 3]])
    assert fleiss_kappa(perfect) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one category"):
        fleiss_kappa(np.array([[3, 0], [3, 0]]))
    with pytest.raises(ValueError, match="same rater count"):
        fleiss_kappa(np.array([[3, 0], [2, 0]]))


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(40, 10))
    emb = np.repeat(base, 3, axis=0) + 0.4 * rng.normal(size=(120, 10))
    units = _units(40, 3)
    point, lo, hi = bootstrap_alpha_ci(units, cosine_distance_matrix(emb), n_boot=300, seed=0)
    assert lo <= point <= hi
    assert hi - lo > 0


def test_bootstrap_is_reproducible_and_narrows_with_units():
    rng = np.random.default_rng(6)
    def spread(n_units: int) -> float:
        base = rng.normal(size=(n_units, 10))
        emb = np.repeat(base, 3, axis=0) + 0.5 * rng.normal(size=(n_units * 3, 10))
        _, lo, hi = bootstrap_alpha_ci(
            _units(n_units, 3), cosine_distance_matrix(emb), n_boot=300, seed=1
        )
        return hi - lo
    assert spread(120) < spread(20)


def test_per_unit_disagreement_ranks_the_odd_one_out():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(10, 6))
    emb = np.repeat(base, 2, axis=0)
    emb[5] = -emb[4]  # unit 2's second annotation now points the other way
    scores = per_unit_disagreement(_units(10, 2), cosine_distance_matrix(emb))
    assert max(scores, key=scores.get) == 2
    assert len(scores) == 10
