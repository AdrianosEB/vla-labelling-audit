"""Controlled label-noise injection and the degradation curve fitted from it.

There is no clean version of DROID to compare against, so the cost of label
noise is estimated the other way round: inject known amounts of noise, train
at each level, and fit performance against noise rate.

Noise modes:

* ``swap``       - the labels of two episodes are exchanged. Models indexing
                   bugs. Preserves the label distribution exactly.
* ``shuffle``    - labels are permuted among the corrupted episodes. An upper
                   bound on damage.
* ``paraphrase`` - the label is replaced by its nearest neighbour in embedding
                   space. Models annotators wording the same thing differently,
                   and serves as the control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseResult", "inject_label_noise", "fit_degradation_curve", "predicted_cost"]


@dataclass(frozen=True)
class NoiseResult:
    """Corrupted labels plus the exact record of what was changed."""

    labels: np.ndarray
    corrupted_idx: np.ndarray
    rate: float
    mode: str

    @property
    def realised_rate(self) -> float:
        """Fraction of labels actually changed.

        Can be lower than the requested rate: a swap may pair two episodes that
        already share a label, and the nearest paraphrase may be the same label.
        """
        return len(self.corrupted_idx) / len(self.labels)


def inject_label_noise(
    labels: np.ndarray,
    rate: float,
    *,
    mode: str = "swap",
    embeddings: np.ndarray | None = None,
    seed: int = 0,
) -> NoiseResult:
    """Corrupt a given fraction of labels reproducibly.

    Args:
        labels: ``[N]`` label ids or strings.
        rate: fraction of episodes to corrupt, in ``[0, 1]``.
        mode: ``"swap"``, ``"shuffle"``, or ``"paraphrase"``.
        embeddings: ``[N, d]`` label embeddings, required for ``"paraphrase"``.
        seed: RNG seed.
    """
    lab = np.asarray(labels).copy()
    n = lab.shape[0]
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    n_corrupt = int(round(rate * n))
    if n_corrupt == 0:
        return NoiseResult(lab, np.array([], dtype=int), rate, mode)

    if mode == "shuffle":
        idx = rng.choice(n, size=n_corrupt, replace=False)
        lab[idx] = lab[rng.permutation(idx)]
    elif mode == "swap":
        idx = rng.choice(n, size=n_corrupt - (n_corrupt % 2), replace=False)
        if idx.size == 0:
            return NoiseResult(lab, np.array([], dtype=int), rate, mode)
        a, b = idx[: idx.size // 2], idx[idx.size // 2 :]
        lab[a], lab[b] = lab[b].copy(), lab[a].copy()
    elif mode == "paraphrase":
        if embeddings is None:
            raise ValueError("paraphrase mode needs label embeddings")
        emb = np.asarray(embeddings, dtype=float)
        if emb.shape[0] != n:
            raise ValueError("embeddings and labels must align")
        norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        idx = rng.choice(n, size=n_corrupt, replace=False)
        sim = norm[idx] @ norm.T
        sim[np.arange(idx.size), idx] = -np.inf
        lab[idx] = lab[sim.argmax(axis=1)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    changed = np.flatnonzero(np.asarray(labels) != lab)
    return NoiseResult(lab, changed, rate, mode)


def fit_degradation_curve(rates: np.ndarray, scores: np.ndarray) -> dict:
    """Least-squares line through (noise rate, performance).

    Linear on purpose: with five or six noise levels a richer form would fit
    seed variance.

    Returns slope, intercept, r, the two-sided p-value for the slope, and its
    standard error.
    """
    r = np.asarray(rates, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if r.shape != s.shape:
        raise ValueError("rates and scores must align")
    if r.size < 3:
        raise ValueError("need >= 3 noise levels to fit a curve")
    from scipy import stats as sps

    fit = sps.linregress(r, s)
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "p_value": float(fit.pvalue),
        "stderr": float(fit.stderr),
    }


def predicted_cost(curve: dict, measured_noise_rate: float) -> float:
    """Performance cost implied by a measured noise rate, from the fitted slope.

    This extrapolates from uniform synthetic noise to real noise, so treat it
    as an order-of-magnitude estimate.
    """
    if not 0.0 <= measured_noise_rate <= 1.0:
        raise ValueError("measured_noise_rate must lie in [0, 1]")
    return float(-curve["slope"] * measured_noise_rate)
