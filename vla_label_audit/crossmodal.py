"""Cross-view diagnostics: does the language label match the trajectory?

Each episode has several embedding views (camera, action, instruction). The
main tool is neighbourhood disagreement: if an episode's visual neighbours
carry one label and it carries another, either the label is wrong or the
episode is unusual. No ground truth is needed. The other functions measure
how well two views line up across the whole corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps
from scipy.spatial import cKDTree
from scipy.special import digamma

__all__ = [
    "AlignmentResult",
    "normalize",
    "knn_indices",
    "neighborhood_disagreement",
    "neighborhood_overlap",
    "rank_correlation_across_views",
    "cca_alignment",
    "gaussian_mi_from_cca",
    "mutual_information_ksg",
    "effective_rank",
    "instruction_space_report",
]


@dataclass(frozen=True)
class AlignmentResult:
    """How well two embedding views of the same episodes correspond."""

    canonical_correlations: np.ndarray
    mean_top_k: float
    gaussian_mi_nats: float
    n_components: int

    def __str__(self) -> str:
        return (
            f"top-{self.n_components} canonical corr mean={self.mean_top_k:.4f}, "
            f"Gaussian MI={self.gaussian_mi_nats:.4f} nats"
        )


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product equals cosine similarity."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected [N, d], got {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm row cannot be normalised")
    return x / n


def knn_indices(embeddings: np.ndarray, k: int, *, exclude_self: bool = True) -> np.ndarray:
    """Exact k-nearest-neighbour indices by cosine similarity.

    Brute force is used on purpose. At a few hundred thousand vectors it is
    cheap, and it rules out ANN recall as an explanation for any result.
    """
    x = normalize(embeddings)
    n = x.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k={k} out of range for {n} items")
    sim = x @ x.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sim, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def neighborhood_disagreement(
    query_view: np.ndarray,
    label_view: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Score each episode by how much its label differs from its neighbours'.

    Args:
        query_view: ``[N, d1]`` view used to find neighbours (visual or action
            embeddings).
        label_view: ``[N, d2]`` view being checked (instruction embeddings).
        k: neighbourhood size.

    Returns:
        ``[N]`` scores in ``[0, 2]``. Higher means the label is less like the
        labels of similar episodes. A high score can also mean a rare but
        correctly labelled behaviour, so it flags a suspect, not an error.
    """
    q = normalize(query_view)
    lab = normalize(label_view)
    if q.shape[0] != lab.shape[0]:
        raise ValueError("views must cover the same episodes")
    nn = knn_indices(q, k)
    # 1 - mean cosine similarity between each label and its neighbours' labels
    sims = np.einsum("nd,nkd->nk", lab, lab[nn])
    return 1.0 - sims.mean(axis=1)


def neighborhood_overlap(view_a: np.ndarray, view_b: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-episode Jaccard overlap between its k nearest neighbours in two views."""
    na, nb = knn_indices(view_a, k), knn_indices(view_b, k)
    out = np.empty(na.shape[0])
    for i in range(na.shape[0]):
        sa, sb = set(na[i].tolist()), set(nb[i].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def rank_correlation_across_views(
    view_a: np.ndarray, view_b: np.ndarray, *, sample: int = 2_000, seed: int = 0
) -> float:
    """Spearman correlation between pairwise distances in two views.

    Unlike :func:`neighborhood_overlap`, this looks at the whole geometry and
    not only the top-k. It runs on a random subsample of ``sample`` episodes
    because the full pairwise set is quadratic.
    """
    a, b = normalize(view_a), normalize(view_b)
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        raise ValueError("views must cover the same episodes")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    iu = np.triu_indices(idx.size, k=1)
    da = (1 - a[idx] @ a[idx].T)[iu]
    db = (1 - b[idx] @ b[idx].T)[iu]
    return float(sps.spearmanr(da, db).statistic)


def cca_alignment(
    view_a: np.ndarray, view_b: np.ndarray, *, n_components: int = 10, reg: float = 1e-4
) -> AlignmentResult:
    """Canonical correlations between two embedding views.

    Whitens both views and takes the SVD of the cross-covariance. ``reg``
    ridges the covariances, which is needed because sample covariances are
    near-singular at embedding dimensionality.
    """
    a = np.asarray(view_a, dtype=float)
    b = np.asarray(view_b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError("views must cover the same episodes")
    n = a.shape[0]
    k = min(n_components, a.shape[1], b.shape[1], n - 1)
    if k < 1:
        raise ValueError("not enough samples or dimensions for CCA")

    a = a - a.mean(0)
    b = b - b.mean(0)
    ca = a.T @ a / (n - 1) + reg * np.eye(a.shape[1])
    cb = b.T @ b / (n - 1) + reg * np.eye(b.shape[1])
    cab = a.T @ b / (n - 1)

    inv_sqrt = lambda m: np.linalg.inv(_sqrtm_psd(m))
    corr = np.linalg.svd(inv_sqrt(ca) @ cab @ inv_sqrt(cb), compute_uv=False)[:k]
    corr = np.clip(corr, 0.0, 1.0)
    return AlignmentResult(corr, float(corr.mean()), gaussian_mi_from_cca(corr), k)


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return v @ np.diag(np.sqrt(np.clip(w, 1e-12, None))) @ v.T


def gaussian_mi_from_cca(correlations: np.ndarray) -> float:
    """Mutual information in nats under a joint-Gaussian assumption.

    ``I = -0.5 * sum(log(1 - rho_i^2))``. The Gaussian assumption does not hold
    for embeddings, but this estimate is stable in high dimension, unlike KSG.
    """
    r = np.clip(np.asarray(correlations, dtype=float), 0.0, 1 - 1e-9)
    return float(-0.5 * np.log1p(-(r**2)).sum())


def mutual_information_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Kraskov-Stoegbauer-Grassberger mutual information estimator (variant 1).

    The estimator is badly biased above roughly ten dimensions, so reduce both
    views with PCA before calling it. Returns MI in nats, clipped at zero.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    n = x.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k={k} out of range for n={n}")

    joint = np.hstack([x, y])
    # KSG uses the max norm in the joint space, hence Chebyshev throughout.
    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, k]
    tx, ty = cKDTree(x), cKDTree(y)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(0.0, mi))


def effective_rank(embeddings: np.ndarray) -> float:
    """``exp`` of the entropy of the normalised singular-value spectrum.

    For instruction embeddings this estimates how many independent directions
    the language spans.
    """
    x = np.asarray(embeddings, dtype=float)
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def instruction_space_report(instruction_embeddings: np.ndarray, texts: list[str] | None = None) -> dict:
    """Summary statistics for the variety in a set of instruction embeddings."""
    x = np.asarray(instruction_embeddings, dtype=float)
    xn = normalize(x)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    report = {
        "n_annotations": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]),
        "effective_rank": effective_rank(x),
        "mean_nearest_neighbor_similarity": float(sim.max(axis=1).mean()),
        "mean_pairwise_similarity": float(
            (xn @ xn.T)[np.triu_indices(x.shape[0], k=1)].mean()
        ),
    }
    if texts is not None:
        if len(texts) != x.shape[0]:
            raise ValueError("texts and embeddings must align")
        report["n_unique_strings"] = len(set(texts))
        report["unique_fraction"] = len(set(texts)) / len(texts)
    return report
