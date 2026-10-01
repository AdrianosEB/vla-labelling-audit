"""Krippendorff's alpha at corpus scale, without an N-by-N distance matrix.

DROID has about 150,000 annotations. The textbook formulation needs the full
pairwise difference matrix, which at that size is 90 GB in float32. Alpha is a
ratio of two terms, and neither needs that matrix:

* Observed disagreement only involves annotations within the same episode.

* Expected disagreement involves every pair, but for squared cosine distance
  on L2-normalised vectors it has a closed form. With ``s_ij = x_i . x_j``:

      sum_ij (1 - s_ij)^2 = n^2 - 2 * ||sum_i x_i||^2 + ||X^T X||_F^2

  since ``sum_ij (x_i . x_j)^2 = trace(M M)`` for ``M = X^T X``. This costs
  O(n d^2) time and O(d^2) memory.

The result is exact. The tests check it against the naive implementation in
``agreement.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .agreement import AgreementResult

__all__ = [
    "alpha_semantic",
    "alpha_nominal",
    "bootstrap_alpha_semantic",
    "per_unit_disagreement",
]


def _group_slices(unit_ids: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(unit_ids, kind="stable")
    sorted_ids = unit_ids[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    return [order[boundaries[i] : boundaries[i + 1]] for i in range(boundaries.size - 1)]


def alpha_semantic(unit_ids: Sequence, embeddings: np.ndarray) -> AgreementResult:
    """Alpha under squared cosine distance, computed in O(n d^2).

    Args:
        unit_ids: ``[N]`` episode identifier per annotation.
        embeddings: ``[N, d]`` sentence embeddings (normalised internally).

    Returns:
        The same :class:`AgreementResult` as the naive implementation.
    """
    unit_ids = np.asarray(unit_ids)
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N, d] embeddings, got {x.shape}")
    if x.shape[0] != unit_ids.shape[0]:
        raise ValueError("unit_ids and embeddings disagree on N")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding: cosine distance is undefined")
    x = x / norms

    groups = [g for g in _group_slices(unit_ids) if g.size >= 2]
    if not groups:
        raise ValueError("no unit has 2+ annotations; alpha is undefined")

    # observed: within-episode pairs only
    total = 0.0
    n = 0
    for g in groups:
        sub = x[g]
        s = sub @ sub.T
        total += ((1.0 - s) ** 2).sum() / (g.size - 1)
        n += g.size
    d_obs = total / n

    # expected: closed form over the pairable pool
    pool = x[np.concatenate(groups)]
    v = pool.sum(axis=0)
    gram = pool.T @ pool
    pair_sum = n * n - 2.0 * float(v @ v) + float((gram * gram).sum())
    d_exp = pair_sum / (n * (n - 1))

    alpha = 1.0 if d_exp == 0 else 1.0 - d_obs / d_exp
    return AgreementResult(float(alpha), float(d_obs), float(d_exp), len(groups), int(n))


def alpha_nominal(unit_ids: Sequence, labels: Sequence) -> AgreementResult:
    """Alpha under exact string identity, in linear time.

    Expected disagreement here is the probability that two annotations drawn
    from the pool differ, which follows from the value counts alone.
    """
    unit_ids = np.asarray(unit_ids)
    lab = np.asarray(labels, dtype=object)
    if lab.shape[0] != unit_ids.shape[0]:
        raise ValueError("unit_ids and labels disagree on N")

    groups = [g for g in _group_slices(unit_ids) if g.size >= 2]
    if not groups:
        raise ValueError("no unit has 2+ annotations; alpha is undefined")

    total = 0.0
    n = 0
    for g in groups:
        counts = Counter(lab[g].tolist())
        same = sum(c * (c - 1) for c in counts.values())   # ordered equal pairs
        total += (g.size * (g.size - 1) - same) / (g.size - 1)
        n += g.size
    d_obs = total / n

    pool_counts = Counter(lab[np.concatenate(groups)].tolist())
    same_pool = sum(c * (c - 1) for c in pool_counts.values())
    d_exp = (n * (n - 1) - same_pool) / (n * (n - 1))

    alpha = 1.0 if d_exp == 0 else 1.0 - d_obs / d_exp
    return AgreementResult(float(alpha), float(d_obs), float(d_exp), len(groups), int(n))


def bootstrap_alpha_semantic(
    unit_ids: Sequence,
    embeddings: np.ndarray,
    *,
    n_boot: int = 1_000,
    alpha_level: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI for the semantic alpha, resampling whole episodes.

    Each replicate recomputes the weighted Gram ``X^T diag(w) X``, where ``w``
    repeats each episode's resample count across its annotations. The other
    terms are weighted sums of per-episode quantities computed once.

    Do not cache one ``d x d`` Gram per episode instead. An earlier version did,
    and at 50,000 episodes x 384 x 384 float64 that is 59 GB.

    Returns:
        ``(alpha, low, high)``.
    """
    unit_ids = np.asarray(unit_ids)
    x = np.asarray(embeddings, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    groups = [g for g in _group_slices(unit_ids) if g.size >= 2]
    if len(groups) < 2:
        raise ValueError("need >= 2 multiply-annotated units to bootstrap")

    idx = np.concatenate(groups)
    pool = np.ascontiguousarray(x[idx])
    owner = np.repeat(np.arange(len(groups)), [g.size for g in groups])
    sizes = np.array([g.size for g in groups], dtype=np.float64)

    obs_terms = np.empty(len(groups))
    sums = np.empty((len(groups), x.shape[1]))
    for i, g in enumerate(groups):
        sub = x[g]
        s = sub @ sub.T
        obs_terms[i] = ((1.0 - s) ** 2).sum() / (g.size - 1)
        sums[i] = sub.sum(axis=0)

    def alpha_from(counts: np.ndarray) -> float:
        n = float(counts @ sizes)
        if n < 2:
            return float("nan")
        d_obs = float(counts @ obs_terms) / n
        v = counts @ sums
        w = counts[owner]
        gram = (pool * w[:, None]).T @ pool
        d_exp = (n * n - 2.0 * float(v @ v) + float((gram * gram).sum())) / (n * (n - 1))
        return 1.0 if d_exp == 0 else 1.0 - d_obs / d_exp

    n_g = len(groups)
    point = alpha_from(np.ones(n_g))
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = alpha_from(np.bincount(rng.integers(0, n_g, n_g), minlength=n_g).astype(float))
    lo, hi = np.percentile(reps, [100 * alpha_level / 2, 100 * (1 - alpha_level / 2)])
    return float(point), float(lo), float(hi)


def per_unit_disagreement(unit_ids: Sequence, embeddings: np.ndarray) -> dict:
    """Mean within-episode squared cosine distance, one score per episode."""
    unit_ids = np.asarray(unit_ids)
    x = np.asarray(embeddings, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    out: dict = {}
    for g in _group_slices(unit_ids):
        if g.size < 2:
            continue
        sub = x[g]
        s = sub @ sub.T
        out[unit_ids[g[0]]] = float(((1.0 - s) ** 2).sum() / (g.size * (g.size - 1)))
    return out
