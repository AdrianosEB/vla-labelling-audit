"""Inter-annotator agreement for free-text robot instructions.

DROID has up to three crowdsourced instructions per episode but publishes no
agreement number. Standard agreement statistics assume categorical labels, and
these are free text: "pick up the red mug" and "grab the mug" agree, but an
exact-match statistic scores them as different.

Krippendorff's alpha takes an arbitrary difference function, so using squared
cosine distance between sentence embeddings gives an alpha over meaning
instead of over strings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AgreementResult",
    "cosine_distance_matrix",
    "exact_match_distance_matrix",
    "krippendorff_alpha",
    "fleiss_kappa",
    "bootstrap_alpha_ci",
    "per_unit_disagreement",
]


@dataclass(frozen=True)
class AgreementResult:
    """Alpha plus the observed and expected disagreement it is built from.

    The two terms are kept because a low alpha can come from high observed
    disagreement or from low expected disagreement (every episode gets the same
    generic label). Alpha alone does not tell these apart.
    """

    alpha: float
    observed: float
    expected: float
    n_units: int
    n_pairable: int

    def __str__(self) -> str:
        return (
            f"alpha={self.alpha:.4f} (D_o={self.observed:.4f}, D_e={self.expected:.4f}, "
            f"{self.n_units} units, {self.n_pairable} values)"
        )


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Squared cosine distance between every pair of embeddings.

    Squared because Krippendorff's alpha is defined over a squared difference
    function.
    """
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected [N, d] embeddings, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding: cosine distance is undefined")
    xn = x / norms
    cos = np.clip(xn @ xn.T, -1.0, 1.0)
    return (1.0 - cos) ** 2


def exact_match_distance_matrix(labels: Sequence) -> np.ndarray:
    """Nominal difference: 0 if the strings match exactly, 1 otherwise."""
    arr = np.asarray(labels, dtype=object)
    return (arr[:, None] != arr[None, :]).astype(float)


def krippendorff_alpha(
    unit_ids: Sequence,
    distance_matrix: np.ndarray,
) -> AgreementResult:
    """Krippendorff's alpha over a precomputed difference matrix.

    Args:
        unit_ids: length-N labels saying which episode each annotation belongs
            to. Units with a single annotation are dropped.
        distance_matrix: ``[N, N]`` symmetric, zero-diagonal squared
            differences, e.g. from :func:`cosine_distance_matrix`.

    Returns:
        Alpha with its observed and expected disagreement. 1.0 is perfect
        agreement, 0.0 is chance, and negative means worse than chance.
    """
    unit_ids = np.asarray(unit_ids)
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"distance_matrix must be square, got {d.shape}")
    if d.shape[0] != unit_ids.shape[0]:
        raise ValueError("unit_ids and distance_matrix disagree on N")
    if not np.allclose(d, d.T, atol=1e-8):
        raise ValueError("distance_matrix must be symmetric")

    groups = [np.flatnonzero(unit_ids == u) for u in np.unique(unit_ids)]
    pairable = [g for g in groups if g.size >= 2]
    if not pairable:
        raise ValueError("no unit has 2+ annotations; alpha is undefined")

    idx = np.concatenate(pairable)
    n = idx.size

    # Observed disagreement: mean within-unit difference, weighted so units with
    # more annotations do not dominate purely by having more pairs.
    total = 0.0
    for g in pairable:
        sub = d[np.ix_(g, g)]
        total += sub.sum() / (g.size - 1)
    d_obs = total / n

    # Expected disagreement: mean difference between any two annotations drawn
    # from the whole pool, ignoring which unit they came from.
    pool = d[np.ix_(idx, idx)]
    d_exp = pool.sum() / (n * (n - 1))

    alpha = 1.0 if d_exp == 0 else 1.0 - d_obs / d_exp
    return AgreementResult(float(alpha), float(d_obs), float(d_exp), len(pairable), int(n))


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for the categorical case.

    Args:
        table: ``[n_units, n_categories]`` counts of how many annotators
            assigned each category to each unit.
    """
    t = np.asarray(table, dtype=float)
    if t.ndim != 2:
        raise ValueError(f"expected [units, categories], got {t.shape}")
    n_per_unit = t.sum(axis=1)
    if not np.allclose(n_per_unit, n_per_unit[0]):
        raise ValueError("Fleiss' kappa requires the same rater count per unit")
    n = n_per_unit[0]
    if n < 2:
        raise ValueError("need >= 2 raters per unit")
    p_i = ((t**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_e = ((t.sum(axis=0) / t.sum()) ** 2).sum()
    if np.isclose(p_e, 1.0):
        raise ValueError("all annotations fall in one category; kappa is undefined")
    return float((p_bar - p_e) / (1 - p_e))


def bootstrap_alpha_ci(
    unit_ids: Sequence,
    distance_matrix: np.ndarray,
    *,
    n_boot: int = 2_000,
    alpha_level: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI for alpha, resampling whole units.

    Returns:
        ``(alpha, low, high)``.
    """
    unit_ids = np.asarray(unit_ids)
    d = np.asarray(distance_matrix, dtype=float)
    uniq = np.unique(unit_ids)
    groups = [np.flatnonzero(unit_ids == u) for u in uniq]
    groups = [g for g in groups if g.size >= 2]
    if len(groups) < 2:
        raise ValueError("need >= 2 multiply-annotated units to bootstrap")

    rng = np.random.default_rng(seed)
    point = krippendorff_alpha(unit_ids, d).alpha
    reps = np.empty(n_boot)
    n_g = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        idx = np.concatenate([groups[j] for j in pick])
        # Relabel so repeated draws of one unit stay distinct units.
        labels = np.concatenate([np.full(groups[j].size, k) for k, j in enumerate(pick)])
        reps[b] = krippendorff_alpha(labels, d[np.ix_(idx, idx)]).alpha
    lo, hi = np.percentile(reps, [100 * alpha_level / 2, 100 * (1 - alpha_level / 2)])
    return float(point), float(lo), float(hi)


def per_unit_disagreement(unit_ids: Sequence, distance_matrix: np.ndarray) -> dict:
    """Mean pairwise difference within each unit, for ranking suspect episodes."""
    unit_ids = np.asarray(unit_ids)
    d = np.asarray(distance_matrix, dtype=float)
    out: dict = {}
    for u in np.unique(unit_ids):
        g = np.flatnonzero(unit_ids == u)
        if g.size < 2:
            continue
        sub = d[np.ix_(g, g)]
        out[u] = float(sub.sum() / (g.size * (g.size - 1)))
    return out
