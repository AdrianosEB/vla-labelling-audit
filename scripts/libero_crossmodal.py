"""Do LIBERO's 40 instructions describe the trajectories they are attached to?

The detector flags an episode when its visual neighbours carry a different
instruction. This script measures two things that have to hold for that to work.

Visual-neighbourhood task purity: the fraction of an episode's k visual nearest
neighbours that share its ``task_index``, against the exact chance baseline
``sum_t n_t(n_t-1)/(N(N-1))``. At chance purity there is no signal for a wrong
label to contradict. The visual-vs-language kNN Jaccard used on DROID is not the
headline here because LIBERO has 40 instructions for 1,693 episodes, so each
language vector is shared by ~42 episodes and its neighbours are arbitrary. That
overlap is still computed (step 3) and reported with the tie-group sizes.

Label-swap calibration (step 7): LIBERO's labels are correct by construction, so
the unmodified corpus can only show specificity. Known wrong labels are injected
by permuting instructions across task boundaries, and ROC AUC, precision@50,
precision@200 and recall@200 are measured against that set over several seeds.

kNN is exact brute-force cosine over 1,693 episode vectors. The two expensive
steps are cached: mean pooling 273,465 frame embeddings (~2 GB of reads) and AV1
decode for the contact sheets. ``--analyze`` reads the cached views only.

Run:
    python scripts/libero_crossmodal.py --build-views     # once, ~1-2 min
    python scripts/libero_crossmodal.py --analyze         # numbers + suspects csv
    python scripts/libero_crossmodal.py --contact-sheets  # PNGs for human review
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_label_audit.crossmodal import (  # noqa: E402
    cca_alignment,
    effective_rank,
    gaussian_mi_from_cca,
    instruction_space_report,
    knn_indices,
    neighborhood_disagreement,
    neighborhood_overlap,
    normalize,
    rank_correlation_across_views,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
LIBERO = CACHE / "libero"
RESULTS = ROOT / "results"

DATASET_TAG = "0e874628fd84"
N_ROWS = 273_465
N_EPISODES = 1693

VISUAL_ENCODERS = ["dinov2", "clip"]
CAMERAS = {"image": "observation.images.image", "image2": "observation.images.image2"}
# Same two language encoders as the DROID analysis. MiniLM is the primary; mpnet
# is the second arm for checking whether the sentence encoder matters.
LANGUAGE_ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
PRIMARY_LANGUAGE = "minilm"
# The suspect ranking uses both cameras concatenated (wrist and scene).
PRIMARY_VISUAL_BLOCK = "concat"

PURITY_KS = [1, 5, 10, 25]
DISAGREEMENT_K = 10
N_BOOT = 1000
CCA_COMPONENTS = 10
CCA_SPLITS = 10
CCA_REG = 1e-4
SWAP_SIZES = [85, 169]
SWAP_SEEDS = 10
TOP_N_CSV = 200
N_CONTACT_SHEETS = 50
N_CONTROL_SHEETS = 10
CONTACT_GRID = 3  # 3x3 = 9 frames per sheet
FRAME_PX = 256

VIEWS_CACHE = CACHE / f"libero_views_{N_EPISODES}_{DATASET_TAG}.npz"
RESULTS_JSON = CACHE / "libero_crossmodal.json"
SUSPECTS_CSV = RESULTS / "libero_suspects.csv"
SHEET_DIR = RESULTS / "suspects"


# View construction


def load_index() -> dict:
    """Per-row and per-episode index arrays written by `libero_embed.py`."""
    z = np.load(CACHE / f"libero_index_{N_ROWS}_{DATASET_TAG}.npz", allow_pickle=True)
    idx = {k: z[k] for k in z.files}
    if idx["episode_length"].shape[0] != N_EPISODES:
        raise SystemExit(f"expected {N_EPISODES} episodes, manifest has {idx['episode_length'].shape[0]}")
    if int(idx["episode_length"].sum()) != N_ROWS:
        raise SystemExit("episode lengths do not sum to the row count")
    return idx


def pool_visual(encoder: str, camera_key: str, idx: dict) -> np.ndarray:
    """Pool frame embeddings into one unit vector per episode.

    Frames are normalised before mean pooling so that large-norm frames do not
    dominate the episode vector, then the mean is normalised again.

    Reads episode by episode off a memmap. The cached matrices are 400-560 MB
    each, and loading all four would use most of a 16 GB machine.
    """
    path = CACHE / f"libero_emb_{encoder}_{camera_key}_{N_ROWS}_{DATASET_TAG}.npy"
    frames = np.load(path, mmap_mode="r")
    if frames.shape[0] != N_ROWS:
        raise SystemExit(f"{path.name}: {frames.shape[0]} rows, expected {N_ROWS}")
    starts, lengths = idx["episode_row_start"], idx["episode_length"]
    out = np.empty((N_EPISODES, frames.shape[1]), dtype=np.float64)
    for i, (s, n) in enumerate(zip(starts, lengths)):
        block = np.asarray(frames[s : s + n], dtype=np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise SystemExit(f"{path.name}: zero-norm frame in episode {i}")
        out[i] = (block / norms).mean(axis=0)
    return normalize(out)


def build_action_view(idx: dict) -> tuple[np.ndarray, list[str]]:
    """A fixed-length per-episode summary of state and action (135 dims).

    Episodes vary in length (74-758 frames), so each is summarised by:

    * moments: mean, std, min, max of the 8 state dims and 7 action dims
      (60 numbers). These carry no information about order.
    * waypoints: state and action at fractions 0, 0.25, 0.5, 0.75, 1 of the
      episode (75 numbers), which add coarse temporal order.

    Columns are z-scored across episodes because the raw units differ (gripper
    width in metres, joint velocities) and the largest scale would otherwise
    dominate cosine distance.
    """
    lengths = idx["episode_length"]
    state = [None] * N_EPISODES
    action = [None] * N_EPISODES
    for path in sorted((LIBERO / "data" / "chunk-000").glob("*.parquet")):
        df = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "observation.state", "action"]
        )
        df = df.sort_values(["episode_index", "frame_index"], kind="stable")
        for ep, g in df.groupby("episode_index", sort=True):
            s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
            a = np.stack(g["action"].to_numpy()).astype(np.float64)
            if state[ep] is not None:
                raise SystemExit(f"episode {ep} appears in more than one data shard")
            if len(s) != lengths[ep]:
                raise SystemExit(f"episode {ep}: {len(s)} rows, manifest says {lengths[ep]}")
            state[ep], action[ep] = s, a

    missing = [i for i, s in enumerate(state) if s is None]
    if missing:
        raise SystemExit(f"{len(missing)} episodes had no rows in data/")

    fracs = np.linspace(0.0, 1.0, 5)
    rows, names = [], []
    for i in range(N_EPISODES):
        s, a = state[i], action[i]
        feats = []
        for tag, m in (("state", s), ("action", a)):
            for stat, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
                feats.append(fn(m, axis=0))
                if i == 0:
                    names += [f"{tag}_{stat}_{d}" for d in range(m.shape[1])]
        picks = np.round(fracs * (len(s) - 1)).astype(int)
        for j, p in enumerate(picks):
            feats += [s[p], a[p]]
            if i == 0:
                names += [f"state_wp{j}_{d}" for d in range(s.shape[1])]
                names += [f"action_wp{j}_{d}" for d in range(a.shape[1])]
        rows.append(np.concatenate(feats))
    raw = np.stack(rows)

    mu, sd = raw.mean(0), raw.std(0)
    # Constant columns stay at zero (instead of being dropped) so the column
    # names stay aligned.
    sd_safe = np.where(sd > 0, sd, 1.0)
    z = (raw - mu) / sd_safe
    return z, names


def embed_instructions(model_name: str, texts: list[str]) -> np.ndarray:
    """Embed the 40 instruction strings."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=64, convert_to_numpy=True).astype(np.float64)


def build_views() -> None:
    """Compute every per-episode view once and cache it."""
    idx = load_index()
    tasks = [str(t) for t in idx["tasks"]]
    ep_task = idx["episode_task_index"].astype(int)
    store: dict[str, np.ndarray] = {"episode_task_index": ep_task}

    for enc in VISUAL_ENCODERS:
        per_cam = []
        for cam in CAMERAS:
            t0 = time.perf_counter()
            v = pool_visual(enc, cam, idx)
            store[f"visual_{enc}_{cam}"] = v
            per_cam.append(v)
            print(f"  {enc}/{cam}: {v.shape} pooled in {time.perf_counter() - t0:.1f}s")
        # Concatenating two unit vectors and renormalising averages the two
        # cameras' cosine similarities.
        store[f"visual_{enc}_concat"] = normalize(np.hstack(per_cam))

    t0 = time.perf_counter()
    action, action_names = build_action_view(idx)
    store["action"] = action
    print(f"  action view: {action.shape} from parquet in {time.perf_counter() - t0:.1f}s")

    for key, model_name in LANGUAGE_ENCODERS.items():
        emb = embed_instructions(model_name, tasks)
        store[f"task_lang_{key}"] = emb  # 40 x d, indexed by task_index
        store[f"language_{key}"] = emb[ep_task]  # per episode
        print(f"  language/{key}: 40 x {emb.shape[1]} instruction vectors")

    np.savez_compressed(
        VIEWS_CACHE,
        tasks=np.array(tasks, dtype=object),
        action_feature_names=np.array(action_names, dtype=object),
        episode_length=idx["episode_length"],
        **store,
    )
    print(f"views cached to {VIEWS_CACHE}")


def load_views() -> dict:
    if not VIEWS_CACHE.exists():
        raise SystemExit(
            f"missing {VIEWS_CACHE}\nrun: python scripts/libero_crossmodal.py --build-views"
        )
    z = np.load(VIEWS_CACHE, allow_pickle=True)
    return {k: z[k] for k in z.files}


def visual_blocks(views: dict) -> dict[str, np.ndarray]:
    """Every visual view, keyed ``encoder/camera``, in a stable order."""
    out = {}
    for enc in VISUAL_ENCODERS:
        for block in [*CAMERAS, "concat"]:
            out[f"{enc}/{block}"] = views[f"visual_{enc}_{block}"]
    return out


# Measurements


def chance_purity(task_index: np.ndarray) -> float:
    """Probability that two distinct episodes drawn at random share a task.

    ``sum_t n_t(n_t-1) / (N(N-1))``, not ``sum_t (n_t/N)^2``, because neighbours
    are drawn without replacement. At n_t ~ 42 the difference matters.
    """
    _, counts = np.unique(task_index, return_counts=True)
    n = task_index.shape[0]
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def purity_curve(view: np.ndarray, task_index: np.ndarray, ks: list[int]) -> dict:
    """Mean task purity at several k, with an episode-clustered bootstrap CI.

    The kNN graph is computed once at max(ks) and prefixes are taken, which
    relies on ``knn_indices`` returning neighbours sorted by similarity.

    The bootstrap resamples episodes with the neighbour graph held fixed, so the
    CI is the sampling error of the mean over the observed episodes.
    """
    nn = knn_indices(view, max(ks))
    same = task_index[nn] == task_index[:, None]
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, view.shape[0], size=(N_BOOT, view.shape[0]))
    out = {}
    for k in ks:
        per_ep = same[:, :k].mean(axis=1)
        reps = per_ep[boot_idx].mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        out[str(k)] = {
            "mean_purity": float(per_ep.mean()),
            "ci95": [float(lo), float(hi)],
            "sd_across_episodes": float(per_ep.std(ddof=1)),
            "frac_episodes_purity_1": float((per_ep == 1.0).mean()),
            "frac_episodes_purity_0": float((per_ep == 0.0).mean()),
        }
    return out


def tie_group_sizes(task_index: np.ndarray) -> dict:
    """How many episodes share each episode's exact instruction vector.

    With 40 instructions over 1,693 episodes, an episode's 10 language nearest
    neighbours are 10 arbitrary members of a much larger tie group, which is why
    the language-side kNN overlap is degenerate.
    """
    _, counts = np.unique(task_index, return_counts=True)
    per_ep = counts[task_index]
    return {
        "n_unique_instruction_vectors": int(counts.shape[0]),
        "group_size_min": int(counts.min()),
        "group_size_median": float(np.median(counts)),
        "group_size_mean": float(counts.mean()),
        "group_size_max": int(counts.max()),
        "group_sizes": [int(c) for c in np.sort(counts)[::-1]],
        "mean_tie_group_size_per_episode": float(per_ep.mean()),
        # Upper bound on language-side overlap: k neighbour slots can cover at
        # most this fraction of the tie group.
        "mean_frac_of_tie_group_reachable_at_k10": float(
            np.mean(np.minimum(1.0, DISAGREEMENT_K / (per_ep - 1)))
        ),
    }


def cca_weights(a: np.ndarray, b: np.ndarray, k: int, reg: float):
    """Canonical directions, by the same construction ``cca_alignment`` uses.

    ``cca_alignment`` returns only correlations, and cross-validation needs the
    weights to apply to the held-out half. ``cross_validated_cca`` records the
    largest difference between these singular values and the module's.
    """
    n = a.shape[0]
    mu_a, mu_b = a.mean(0), b.mean(0)
    ac, bc = a - mu_a, b - mu_b
    ca = ac.T @ ac / (n - 1) + reg * np.eye(a.shape[1])
    cb = bc.T @ bc / (n - 1) + reg * np.eye(b.shape[1])
    cab = ac.T @ bc / (n - 1)

    def inv_sqrt(m):
        w, v = np.linalg.eigh(m)
        return v @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ v.T

    ia, ib = inv_sqrt(ca), inv_sqrt(cb)
    u, s, vt = np.linalg.svd(ia @ cab @ ib)
    return ia @ u[:, :k], ib @ vt[:k].T, s[:k], mu_a, mu_b


def cross_validated_cca(a: np.ndarray, b: np.ndarray, *, n_splits: int, k: int, reg: float) -> dict:
    """In-sample vs held-out canonical correlations over random half-splits.

    With 1,693 episodes against 384-1024 dimensions, in-sample canonical
    correlations are close to 1 even for random matrices. Only the held-out
    correlations are evidence; the gap measures the overfitting.
    """
    n = a.shape[0]
    in_s, in_v, out_s = [], [], []
    max_gap = 0.0
    for split in range(n_splits):
        rng = np.random.default_rng(split)
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2 :]
        wa, wb, s, mu_a, mu_b = cca_weights(a[tr], b[tr], k, reg)
        ref = cca_alignment(a[tr], b[tr], n_components=k, reg=reg).canonical_correlations
        max_gap = max(max_gap, float(np.abs(ref - np.clip(s, 0, 1)).max()))
        za, zb = (a[te] - mu_a) @ wa, (b[te] - mu_b) @ wb
        held = [float(sps.pearsonr(za[:, j], zb[:, j]).statistic) for j in range(k)]
        # The ridge shrinks the singular values below the realised in-sample
        # correlations of the variates, so both are reported.
        ta, tb = (a[tr] - mu_a) @ wa, (b[tr] - mu_b) @ wb
        in_v.append(np.array([sps.pearsonr(ta[:, j], tb[:, j]).statistic for j in range(k)]))
        in_s.append(np.clip(s, 0, 1))
        out_s.append(np.array(held))
    in_s, in_v, out_s = np.stack(in_s), np.stack(in_v), np.stack(out_s)
    # Held-out correlations can be negative (the sign of a canonical direction
    # is not identified out of sample). MI depends on rho^2, so use |rho|.
    mi = [gaussian_mi_from_cca(np.abs(r)) for r in out_s]
    return {
        "n_splits": n_splits,
        "n_components": k,
        "reg": reg,
        "in_sample_corr_mean": [float(v) for v in in_s.mean(0)],
        "in_sample_variate_corr_mean": [float(v) for v in in_v.mean(0)],
        "in_sample_variate_mean_top_k": float(in_v.mean()),
        "overfitting_gap_variate_mean_top_k": float(in_v.mean() - out_s.mean()),
        "held_out_corr_mean": [float(v) for v in out_s.mean(0)],
        "held_out_corr_sd": [float(v) for v in out_s.std(0, ddof=1)],
        "in_sample_mean_top_k": float(in_s.mean()),
        "held_out_mean_top_k": float(out_s.mean()),
        "overfitting_gap_mean_top_k": float(in_s.mean() - out_s.mean()),
        "held_out_gaussian_mi_nats_mean": float(np.mean(mi)),
        "held_out_gaussian_mi_nats_sd": float(np.std(mi, ddof=1)),
        "in_sample_gaussian_mi_nats_mean": float(
            np.mean([gaussian_mi_from_cca(r) for r in in_s])
        ),
        "weights_vs_module_max_abs_diff": max_gap,
    }


def rank_desc(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = most suspect, average ranks for ties."""
    return sps.rankdata(-scores, method="average")


def disagreement_from_nn(nn: np.ndarray, language: np.ndarray) -> np.ndarray:
    """``neighborhood_disagreement`` with the neighbour graph supplied.

    The graph depends only on the visual view, so it is computed once and reused
    across the label-swap replicates. ``analyze`` checks this against
    ``neighborhood_disagreement`` to 1e-12 on the unmodified corpus.

    Scores are rounded to 12 decimals. Most LIBERO episodes have ten neighbours
    with a bit-identical instruction vector, so the true score is zero but
    floating point leaves +/-4e-16. That noise orders the ~1,600 tied episodes
    in a way that is correlated across encoders, and it produced a top-200
    encoder agreement of 0.80 where the tie-broken value is 0.17.
    """
    lab = normalize(language)
    raw = 1.0 - np.einsum("nd,nkd->nk", lab, lab[nn]).mean(axis=1)
    return np.round(raw, 12) + 0.0  # +0.0 collapses -0.0 onto 0.0


def ensemble_scores(per_encoder: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Ensemble the encoders by averaging within-encoder ranks.

    DINOv2 and CLIP top-200 lists overlap only about half, so intersecting them
    would drop most of what each found, and their raw scores have different
    spreads, so those cannot be averaged directly.

    Returns ``(suspicion, ranks)``. ``suspicion`` is the negated mean rank, so
    higher means more suspect.
    """
    ranks = {k: rank_desc(v) for k, v in per_encoder.items()}
    mean_rank = np.mean(np.stack(list(ranks.values())), axis=0)
    return -mean_rank, ranks


def top_k_order(suspicion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Episode indices most-suspect first, ties broken at random.

    With 40 distinct instruction vectors the score has few distinct values, so
    ties are common. Breaking them by array order would make precision@50 depend
    on episode numbering.
    """
    return np.lexsort((rng.random(suspicion.shape[0]), -suspicion))


def swap_instructions(
    task_index: np.ndarray, n_swap: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Permute instructions among a random subset, never onto the same task.

    The permutation is within the selected subset, so the corpus-wide
    instruction frequencies are unchanged and only the pairing moves.

    Fixed points (an episode mapped back onto its own task) are repaired by
    swapping with a partner such that both end up on a different task. Raises
    if the subset spans only one task.
    """
    n = task_index.shape[0]
    sel = rng.choice(n, size=n_swap, replace=False)
    old = task_index[sel]
    new = old[rng.permutation(n_swap)]
    for _ in range(1000):
        bad = np.flatnonzero(new == old)
        if bad.size == 0:
            break
        for i in bad:
            partners = np.flatnonzero((new != old[i]) & (old != new[i]))
            partners = partners[partners != i]
            if partners.size == 0:
                raise RuntimeError("cannot repair fixed point; subset spans one task only")
            j = rng.choice(partners)
            new[i], new[j] = new[j], new[i]
    else:
        raise RuntimeError("swap repair did not converge")
    if np.any(new == old):
        raise RuntimeError("swap left an episode on its own task")
    out = task_index.copy()
    out[sel] = new
    return out, sel


def detection_metrics(
    suspicion: np.ndarray, truth: np.ndarray, n_swap: int, rng: np.random.Generator
) -> dict:
    """ROC AUC plus precision and recall at worklist sizes 50 and 200."""
    from sklearn.metrics import roc_auc_score

    order = top_k_order(suspicion, rng)
    is_swapped = np.zeros(suspicion.shape[0], dtype=bool)
    is_swapped[truth] = True
    hits50 = int(is_swapped[order[:50]].sum())
    hits200 = int(is_swapped[order[:200]].sum())
    return {
        "auc": float(roc_auc_score(is_swapped, suspicion)),
        "p_at_50": hits50 / 50.0,
        "p_at_200": hits200 / 200.0,
        "recall_at_200": hits200 / n_swap,
    }


def score_distribution(x: np.ndarray) -> dict:
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "percentiles": {str(q): float(v) for q, v in zip(qs, np.percentile(x, qs))},
        "n_distinct_values": int(np.unique(np.round(x, 12)).size),
    }


# Analysis driver


def analyze() -> None:
    views = load_views()
    tasks = [str(t) for t in views["tasks"]]
    ep_task = views["episode_task_index"].astype(int)
    blocks = visual_blocks(views)
    action = views["action"]
    lang = {k: views[f"language_{k}"] for k in LANGUAGE_ENCODERS}
    task_lang = {k: views[f"task_lang_{k}"] for k in LANGUAGE_ENCODERS}
    res: dict = {
        "config": {
            "n_episodes": N_EPISODES,
            "n_frames": N_ROWS,
            "dataset_tag": DATASET_TAG,
            "purity_ks": PURITY_KS,
            "disagreement_k": DISAGREEMENT_K,
            "n_bootstrap": N_BOOT,
            "primary_visual_block": PRIMARY_VISUAL_BLOCK,
            "primary_language_encoder": PRIMARY_LANGUAGE,
            "language_models": LANGUAGE_ENCODERS,
            "swap_sizes": SWAP_SIZES,
            "swap_seeds": SWAP_SEEDS,
        }
    }

    # 1. Spectral diagnostics
    spectral = {}
    for name, v in blocks.items():
        spectral[f"visual_{name}"] = {"dims": int(v.shape[1]), "effective_rank": effective_rank(v)}
    spectral["action"] = {"dims": int(action.shape[1]), "effective_rank": effective_rank(action)}
    for k in LANGUAGE_ENCODERS:
        spectral[f"language_{k}_per_episode"] = {
            "dims": int(lang[k].shape[1]),
            "n_rows": N_EPISODES,
            "effective_rank": effective_rank(lang[k]),
        }
        spectral[f"language_{k}_unique_40"] = {
            "dims": int(task_lang[k].shape[1]),
            "n_rows": 40,
            "effective_rank": effective_rank(task_lang[k]),
        }
    res["spectral"] = spectral
    res["instruction_space"] = {
        k: instruction_space_report(task_lang[k], tasks) for k in LANGUAGE_ENCODERS
    }

    # 2. Visual-neighbourhood task purity
    chance = chance_purity(ep_task)
    purity = {"chance_baseline": chance, "by_view": {}}
    for name, v in blocks.items():
        cur = purity_curve(v, ep_task, PURITY_KS)
        for k in cur:
            cur[k]["ratio_to_chance"] = cur[k]["mean_purity"] / chance
        purity["by_view"][name] = cur
    # Action view as a control.
    cur = purity_curve(action, ep_task, PURITY_KS)
    for k in cur:
        cur[k]["ratio_to_chance"] = cur[k]["mean_purity"] / chance
    purity["by_view"]["action (control)"] = cur
    res["task_purity"] = purity

    # 3. Language overlap (degenerate on LIBERO, reported anyway)
    overlap = {}
    for enc in VISUAL_ENCODERS:
        for lk in LANGUAGE_ENCODERS:
            ov = neighborhood_overlap(
                blocks[f"{enc}/{PRIMARY_VISUAL_BLOCK}"], lang[lk], k=DISAGREEMENT_K
            )
            overlap[f"{enc}/{PRIMARY_VISUAL_BLOCK} vs {lk}"] = {
                "mean_jaccard": float(ov.mean()),
                "median_jaccard": float(np.median(ov)),
                "sd": float(ov.std(ddof=1)),
                "max": float(ov.max()),
                "frac_zero": float((ov == 0).mean()),
            }
    res["neighborhood_overlap"] = overlap
    res["language_tie_groups"] = tie_group_sizes(ep_task)

    # 4. Rank correlation across views
    rc = {}
    for enc in VISUAL_ENCODERS:
        v = blocks[f"{enc}/{PRIMARY_VISUAL_BLOCK}"]
        for lk in LANGUAGE_ENCODERS:
            rc[f"{enc} vs language_{lk}"] = rank_correlation_across_views(
                v, lang[lk], sample=N_EPISODES, seed=0
            )
        rc[f"{enc} vs action"] = rank_correlation_across_views(v, action, sample=N_EPISODES, seed=0)
    rc["action vs language_minilm"] = rank_correlation_across_views(
        action, lang["minilm"], sample=N_EPISODES, seed=0
    )
    rc["dinov2 vs clip (both concat)"] = rank_correlation_across_views(
        blocks["dinov2/concat"], blocks["clip/concat"], sample=N_EPISODES, seed=0
    )
    # The language side has at most C(40,2)+1 distinct pairwise distances.
    n_tasks = len(tasks)
    rc_note = {
        "distinct_language_pair_distances_max": n_tasks * (n_tasks - 1) // 2 + 1,
        "n_pairs_used": N_EPISODES * (N_EPISODES - 1) // 2,
    }
    res["rank_correlation"] = {"values": rc, "language_tie_note": rc_note}

    # 5. Cross-validated CCA
    cca = {}
    for enc in VISUAL_ENCODERS:
        v = blocks[f"{enc}/{PRIMARY_VISUAL_BLOCK}"]
        for lk in LANGUAGE_ENCODERS:
            cca[f"{enc} vs language_{lk}"] = cross_validated_cca(
                v, lang[lk], n_splits=CCA_SPLITS, k=CCA_COMPONENTS, reg=CCA_REG
            )
        cca[f"{enc} vs action"] = cross_validated_cca(
            v, action, n_splits=CCA_SPLITS, k=CCA_COMPONENTS, reg=CCA_REG
        )
    res["cca"] = {
        "note": (
            "language side has only 40 distinct vectors, so its centred rank is at most 39; "
            f"any component beyond that is fitting noise. n_components={CCA_COMPONENTS}."
        ),
        "by_pair": cca,
    }

    # 6. Suspect list
    nn = {
        enc: knn_indices(blocks[f"{enc}/{PRIMARY_VISUAL_BLOCK}"], DISAGREEMENT_K)
        for enc in VISUAL_ENCODERS
    }
    scores = {}
    for enc in VISUAL_ENCODERS:
        fast = disagreement_from_nn(nn[enc], lang[PRIMARY_LANGUAGE])
        ref = neighborhood_disagreement(
            blocks[f"{enc}/{PRIMARY_VISUAL_BLOCK}"], lang[PRIMARY_LANGUAGE], k=DISAGREEMENT_K
        )
        if not np.allclose(fast, ref, atol=1e-12):
            raise SystemExit(f"{enc}: cached-graph disagreement does not match the module")
        scores[enc] = fast
    suspicion, ranks = ensemble_scores(scores)
    ens_rank = sps.rankdata(-suspicion, method="ordinal")

    # The score depends only on the multiset of neighbour tasks, so hundreds of
    # episodes tie. A seeded random tie-break keeps the worklist from depending
    # on episode numbering.
    csv_rng = np.random.default_rng(0)
    order = top_k_order(suspicion, csv_rng)[:TOP_N_CSV]
    RESULTS.mkdir(exist_ok=True)
    rows = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "episode_index": order,
            "task_index": ep_task[order],
            "instruction": [tasks[t] for t in ep_task[order]],
            "dinov2_score": scores["dinov2"][order],
            "dinov2_rank": ranks["dinov2"][order],
            "clip_score": scores["clip"][order],
            "clip_rank": ranks["clip"][order],
            "ensemble_rank": ens_rank[order],
        }
    )
    rows.to_csv(SUSPECTS_CSV, index=False)
    print(f"  top-{TOP_N_CSV} suspects written to {SUSPECTS_CSV}")

    # Each encoder gets its own tie-break seed. With a shared seed the 1,595
    # episodes tied at score 0 break identically in both encoders and the
    # top-200 overlap comes out at 0.8 from the shared draw alone.
    def top_set(x: np.ndarray, k: int, seed: int) -> set:
        return set(top_k_order(x, np.random.default_rng(seed))[:k].tolist())

    d, c = scores["dinov2"], scores["clip"]
    top_d = lambda k: top_set(d, k, 101)
    top_c = lambda k: top_set(c, k, 202)
    agree = {
        "spearman_all_episodes": float(sps.spearmanr(d, c).statistic),
        "top50_overlap": len(top_d(50) & top_c(50)) / 50,
        "top200_overlap": len(top_d(200) & top_c(200)) / 200,
        "dinov2_top200_in_clip_top500": len(top_d(200) & top_c(500)) / 200,
        "clip_top200_in_dinov2_top500": len(top_c(200) & top_d(500)) / 200,
        # Only ~60 episodes per encoder have a nonzero score, so fixed-K overlaps
        # depend on tie-breaking. The Jaccard of the nonzero sets does not.
        "nonzero_set_jaccard": float(
            len(set(np.flatnonzero(d > 1e-9).tolist()) & set(np.flatnonzero(c > 1e-9).tolist()))
            / max(1, len(set(np.flatnonzero(d > 1e-9).tolist()) | set(np.flatnonzero(c > 1e-9).tolist())))
        ),
        "n_nonzero_dinov2": int((d > 1e-9).sum()),
        "n_nonzero_clip": int((c > 1e-9).sum()),
        "n_nonzero_both": int(((d > 1e-9) & (c > 1e-9)).sum()),
        "tail_only_spearman_union_top200": float(
            sps.spearmanr(
                d[sorted(top_d(200) | top_c(200))],
                c[sorted(top_d(200) | top_c(200))],
            ).statistic
        ),
        "tail_only_spearman_union_nonzero": float(
            sps.spearmanr(
                d[sorted(set(np.flatnonzero(d > 1e-9)) | set(np.flatnonzero(c > 1e-9)))],
                c[sorted(set(np.flatnonzero(d > 1e-9)) | set(np.flatnonzero(c > 1e-9)))],
            ).statistic
        ),
    }
    res["suspects"] = {
        "csv": str(SUSPECTS_CSV),
        "query_view": f"visual {PRIMARY_VISUAL_BLOCK} (both cameras)",
        "label_view": PRIMARY_LANGUAGE,
        "k": DISAGREEMENT_K,
        "encoder_agreement": agree,
        "top50_task_index_counts": {
            str(t): int(n) for t, n in zip(*np.unique(ep_task[order[:50]], return_counts=True))
        },
        "tie_break": "seeded random (default_rng(0)); scores are heavily tied, see specificity",
    }

    # 8. Specificity on the unmodified corpus
    rng = np.random.default_rng(0)
    ord_un = top_k_order(suspicion, rng)
    mean_rank = -suspicion
    spec = {
        "per_encoder_score_distribution": {k: score_distribution(v) for k, v in scores.items()},
        "ensemble_mean_rank_distribution": score_distribution(mean_rank),
        "top50_vs_bulk": {},
    }
    # A score of zero means all ten visual neighbours share the episode's task.
    # Specificity is the count of episodes with a nonzero score.
    tol = 1e-9
    for k, v in scores.items():
        top = np.sort(v)[::-1][:50]
        rest = np.sort(v)[::-1][50:]
        boundary = float(top[-1])
        spec["top50_vs_bulk"][k] = {
            "top50_mean": float(top.mean()),
            "rest_mean": float(rest.mean()),
            "gap_in_sd_of_bulk": float((top.mean() - rest.mean()) / rest.std(ddof=1)),
            "score_at_rank_50": boundary,
            "score_at_rank_51": float(rest[0]),
            "drop_rank50_to_rank51": float(top[-1] - rest[0]),
            "median": float(np.median(v)),
            "max": float(v.max()),
            "n_episodes_score_gt_0": int((v > tol).sum()),
            "frac_episodes_score_eq_0": float((v <= tol).mean()),
            "n_tied_at_rank50_boundary": int((np.abs(v - boundary) <= tol).sum()),
            "n_strictly_above_rank50_boundary": int((v > boundary + tol).sum()),
        }
    both_zero = int(((scores["dinov2"] <= tol) & (scores["clip"] <= tol)).sum())
    spec["ensemble"] = {
        "n_episodes_zero_in_both_encoders": both_zero,
        "n_episodes_nonzero_in_either": N_EPISODES - both_zero,
        "top50_mean_rank_range": [
            float(mean_rank[ord_un[:50]].min()),
            float(mean_rank[ord_un[:50]].max()),
        ],
        "mean_rank_of_the_zero_block": float(mean_rank.max()),
        "n_distinct_ensemble_scores": int(np.unique(np.round(mean_rank, 9)).size),
    }
    res["specificity_unmodified"] = spec

    # 7. Label-swap calibration
    swap = {}
    task_vecs = task_lang[PRIMARY_LANGUAGE]
    for n_swap in SWAP_SIZES:
        per_arm: dict[str, list[dict]] = {"dinov2": [], "clip": [], "ensemble": []}
        for seed in range(SWAP_SEEDS):
            rng = np.random.default_rng([n_swap, seed])
            new_task, sel = swap_instructions(ep_task, n_swap, rng)
            lang_mod = task_vecs[new_task]
            s = {enc: disagreement_from_nn(nn[enc], lang_mod) for enc in VISUAL_ENCODERS}
            ens, _ = ensemble_scores(s)
            for enc in VISUAL_ENCODERS:
                per_arm[enc].append(detection_metrics(s[enc], sel, n_swap, rng))
            per_arm["ensemble"].append(detection_metrics(ens, sel, n_swap, rng))
        swap[str(n_swap)] = {
            arm: {
                m: {
                    "mean": float(np.mean([r[m] for r in reps])),
                    "sd": float(np.std([r[m] for r in reps], ddof=1)),
                }
                for m in ("auc", "p_at_50", "p_at_200", "recall_at_200")
            }
            for arm, reps in per_arm.items()
        }
        swap[str(n_swap)]["n_swapped"] = n_swap
        swap[str(n_swap)]["swapped_fraction"] = n_swap / N_EPISODES
        # precision@K cannot exceed S/K when S < K, so store the ceiling too.
        swap[str(n_swap)]["p_at_50_ceiling"] = min(1.0, n_swap / 50)
        swap[str(n_swap)]["p_at_200_ceiling"] = min(1.0, n_swap / 200)
        print(f"  label-swap S={n_swap}: ensemble AUC {swap[str(n_swap)]['ensemble']['auc']['mean']:.3f}")
    res["label_swap"] = {
        "procedure": (
            "select S episodes uniformly without replacement; permute their task_index "
            "among themselves, repairing fixed points so every selected episode receives an "
            "instruction from a different task_index; unselected episodes untouched. The "
            "corpus-wide instruction frequency is preserved. Visual views are unchanged; only "
            "the label view moves. rng = numpy default_rng([S, seed]), seed = 0..%d, and the "
            "same generator supplies the random tie-break in the ranking."
        )
        % (SWAP_SEEDS - 1),
        "by_size": swap,
    }

    RESULTS_JSON.write_text(json.dumps(res, indent=2))
    print(f"results written to {RESULTS_JSON}")
    print_summary(res)


def print_summary(res: dict) -> None:
    print("\n" + "=" * 78)
    print("LIBERO CROSS-MODAL ALIGNMENT")
    print("=" * 78)
    print("\n  effective rank")
    for k, v in res["spectral"].items():
        print(f"    {k:<34} dims {v['dims']:>4}   eff.rank {v['effective_rank']:.2f}")

    ch = res["task_purity"]["chance_baseline"]
    print(f"\n  visual-neighbourhood task purity (chance = {ch:.4f})")
    header = "    {:<20}".format("view") + "".join(f"{'k='+str(k):>22}" for k in PURITY_KS)
    print(header)
    for name, cur in res["task_purity"]["by_view"].items():
        line = f"    {name:<20}"
        for k in PURITY_KS:
            e = cur[str(k)]
            line += "{:>10.3f} [{:.3f},{:.3f}]".format(e["mean_purity"], *e["ci95"])
        print(line)

    print("\n  degenerate language overlap (Jaccard, k=10)")
    for k, v in res["neighborhood_overlap"].items():
        print(f"    {k:<34} mean {v['mean_jaccard']:.4f}  frac-zero {v['frac_zero']:.3f}")
    tg = res["language_tie_groups"]
    print(
        f"    tie groups: {tg['n_unique_instruction_vectors']} unique vectors, "
        f"sizes {tg['group_size_min']}-{tg['group_size_max']} (median {tg['group_size_median']:.0f})"
    )

    print("\n  rank correlation across views (Spearman over pairwise distances)")
    for k, v in res["rank_correlation"]["values"].items():
        print(f"    {k:<34} {v:+.4f}")

    print("\n  CCA (10 half-splits, top-10 components)")
    for k, v in res["cca"]["by_pair"].items():
        print(
            f"    {k:<34} in-sample {v['in_sample_mean_top_k']:.4f}  "
            f"held-out {v['held_out_mean_top_k']:+.4f}  "
            f"gap {v['overfitting_gap_mean_top_k']:+.4f}  "
            f"MI(held-out) {v['held_out_gaussian_mi_nats_mean']:.3f} nats"
        )

    sp = res["specificity_unmodified"]
    print("\n  specificity on the unmodified corpus (nothing to find by construction)")
    for k, v in sp["top50_vs_bulk"].items():
        print(
            f"    {k:<8} score>0 for {v['n_episodes_score_gt_0']:>4} / {N_EPISODES} episodes; "
            f"median {v['median']:.4f}  max {v['max']:.4f}  "
            f"top50 mean {v['top50_mean']:.4f} = {v['gap_in_sd_of_bulk']:.1f} sd of the bulk; "
            f"rank-50 boundary is a tie block of {v['n_tied_at_rank50_boundary']}"
        )
    e = sp["ensemble"]
    print(
        f"    ensemble: {e['n_episodes_nonzero_in_either']} episodes nonzero in either encoder, "
        f"{e['n_episodes_zero_in_both_encoders']} tied in the zero block "
        f"({e['n_distinct_ensemble_scores']} distinct ensemble scores)"
    )

    a = res["suspects"]["encoder_agreement"]
    print("\n  DINOv2 vs CLIP suspect agreement")
    for k, v in a.items():
        print(f"    {k:<34} {v:+.4f}")

    print("\n  label-swap calibration (mean +/- sd over seeds)")
    print(
        "    {:<6} {:<10} {:>14} {:>14} {:>14} {:>14}".format(
            "S", "arm", "AUC", "P@50", "P@200", "recall@200"
        )
    )
    for s, arms in res["label_swap"]["by_size"].items():
        for arm in ("dinov2", "clip", "ensemble"):
            m = arms[arm]
            print(
                "    {:<6} {:<10}".format(s, arm)
                + "".join(
                    "{:>9.3f}+/-{:.3f}".format(m[key]["mean"], m[key]["sd"])
                    for key in ("auc", "p_at_50", "p_at_200", "recall_at_200")
                )
            )


# Contact sheets


def episode_frame_plan(episodes: list[int], views: dict) -> dict[int, list[int]]:
    """Nine uniformly spaced frame indices per episode."""
    lengths = views["episode_length"]
    return {
        ep: np.round(np.linspace(0, lengths[ep] - 1, CONTACT_GRID**2)).astype(int).tolist()
        for ep in episodes
    }


def fetch_frames(plan: dict[int, list[int]]) -> dict[int, list[np.ndarray]]:
    """Decode the requested frames, one sequential pass per video file.

    Seeking to each frame was measured at ~200x the per-frame cost of decoding
    a whole file sequentially, so requests are grouped by video file and each
    file is decoded once.
    """
    import av

    ep_df = pd.read_parquet(LIBERO / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    cam = CAMERAS["image"]
    file_index = ep_df[f"videos/{cam}/file_index"].to_numpy()
    from_ts = ep_df[f"videos/{cam}/from_timestamp"].to_numpy()

    wanted: dict[int, dict[int, tuple[int, int]]] = {}
    for ep, frames in plan.items():
        f = int(file_index[ep])
        base = int(round(from_ts[ep] * 10.0))
        for slot, fr in enumerate(frames):
            wanted.setdefault(f, {})[base + fr] = (ep, slot)

    out: dict[int, list[np.ndarray]] = {ep: [None] * (CONTACT_GRID**2) for ep in plan}
    for f in sorted(wanted):
        path = LIBERO / "videos" / cam / "chunk-000" / f"file-{f:03d}.mp4"
        need = wanted[f]
        last = max(need)
        pos = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if pos in need:
                    ep, slot = need[pos]
                    out[ep][slot] = frame.to_ndarray(format="rgb24")
                pos += 1
                if pos > last:
                    break
        print(f"    decoded {path.name} ({len(need)} frames kept)")
    for ep, frames in out.items():
        if any(fr is None for fr in frames):
            raise SystemExit(f"episode {ep}: some frames were never decoded")
    return out


def render_sheet(path: Path, frames: list[np.ndarray], caption: list[str]) -> None:
    """3x3 grid of 256x256 frames under a multi-line caption."""
    from PIL import Image, ImageDraw, ImageFont

    g, px = CONTACT_GRID, FRAME_PX
    pad = 4
    grid_w = g * px + (g + 1) * pad
    font = None
    for cand in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 22)
            break
    if font is None:
        font = ImageFont.load_default()

    lines: list[str] = []
    for c in caption:
        lines += textwrap.wrap(c, width=70) or [""]
    line_h = 30
    cap_h = pad * 2 + line_h * len(lines)
    canvas = Image.new("RGB", (grid_w, cap_h + grid_w), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((pad * 2, pad + i * line_h), line, fill=(0, 0, 0), font=font)
    for i, arr in enumerate(frames):
        im = Image.fromarray(arr)
        if im.size != (px, px):
            im = im.resize((px, px), Image.BICUBIC)
        r, c = divmod(i, g)
        canvas.paste(im, (pad + c * (px + pad), cap_h + pad + r * (px + pad)))
    canvas.save(path)


def contact_sheets() -> None:
    """Render the top-50 suspects plus 10 controls for human inspection."""
    views = load_views()
    tasks = [str(t) for t in views["tasks"]]
    ep_task = views["episode_task_index"].astype(int)
    if not SUSPECTS_CSV.exists():
        raise SystemExit(f"missing {SUSPECTS_CSV}; run --analyze first")
    susp = pd.read_csv(SUSPECTS_CSV)
    top = susp["episode_index"].to_numpy()[:N_CONTACT_SHEETS]

    # Controls come from outside the whole top-200 list, so none is a near-miss
    # suspect.
    rng = np.random.default_rng(0)
    excluded = set(susp["episode_index"].tolist())
    pool = np.array([i for i in range(N_EPISODES) if i not in excluded])
    controls = rng.choice(pool, size=N_CONTROL_SHEETS, replace=False)

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    plan = episode_frame_plan([*top.tolist(), *controls.tolist()], views)
    t0 = time.perf_counter()
    frames = fetch_frames(plan)
    print(f"  decode took {time.perf_counter() - t0:.1f}s")

    for rank, ep in enumerate(top, start=1):
        row = susp.iloc[rank - 1]
        render_sheet(
            SHEET_DIR / f"rank{rank:02d}_ep{ep:04d}.png",
            frames[ep],
            [
                f"SUSPECT rank {rank}  |  episode {ep}  |  task_index {ep_task[ep]}"
                f"  |  ens.rank {int(row['ensemble_rank'])}",
                f"dinov2 {row['dinov2_score']:.4f} (rank {row['dinov2_rank']:.0f})   "
                f"clip {row['clip_score']:.4f} (rank {row['clip_rank']:.0f})",
                f'instruction: "{tasks[ep_task[ep]]}"',
            ],
        )
    for i, ep in enumerate(controls, start=1):
        render_sheet(
            SHEET_DIR / f"control{i:02d}_ep{ep:04d}.png",
            frames[ep],
            [
                f"CONTROL {i}  |  episode {ep}  |  task_index {ep_task[ep]}  |  not in top-200",
                f'instruction: "{tasks[ep_task[ep]]}"',
            ],
        )
    print(f"  {len(top)} suspect + {len(controls)} control sheets in {SHEET_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-views", action="store_true", help="pool frames into per-episode views")
    mode.add_argument("--analyze", action="store_true", help="all measurements from cached views")
    mode.add_argument("--contact-sheets", action="store_true", help="render PNGs for the top suspects")
    args = ap.parse_args()

    if args.build_views:
        build_views()
    elif args.analyze:
        analyze()
    else:
        contact_sheets()


if __name__ == "__main__":
    main()
