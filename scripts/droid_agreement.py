"""Inter-annotator agreement on DROID's language instructions.

DROID collected up to three crowdsourced instructions per episode, and the
dataset paper reports no agreement number for them. The annotations ship as a
separate 12 MB JSON, so none of the video is needed. Structure:

    "IRIS+7dfa2da3+2023-04-25-11h-42m-28s": {
        "language_instruction1": "Pour the contents of the bottle on the right into the sink",
        "language_instruction2": "Pour the contents of the larger container into the sink",
        "language_instruction3": "Pour the content of the bottle into the sink"
    }

Run:
    python scripts/droid_agreement.py                 # download + full analysis
    python scripts/droid_agreement.py --limit 2000    # quick smoke test first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import urllib.request
from pathlib import Path

import numpy as np

from vla_label_audit.crossmodal import effective_rank, instruction_space_report
from vla_label_audit.scalable import (
    alpha_nominal,
    alpha_semantic,
    bootstrap_alpha_semantic,
    per_unit_disagreement,
)

ANNOTATION_URL = (
    "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/"
    "aggregated-annotations-030724.json"
)
CACHE = Path("data")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, ~80 MB, fast on MPS


def fetch_annotations() -> dict:
    """Download the annotation JSON on first use and cache it under data/."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "droid_annotations.json"
    if not path.exists():
        print(f"downloading annotations (~12 MB) from\n  {ANNOTATION_URL}")
        try:
            # The python.org build on macOS has no CA bundle, so urlopen fails with
            # CERTIFICATE_VERIFY_FAILED. certifi is already a transitive dependency.
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(ANNOTATION_URL, context=ctx) as resp:
                path.write_bytes(resp.read())
        except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
            path.unlink(missing_ok=True)  # never leave a truncated file behind
            raise SystemExit(
                f"download failed: {exc}\n\n"
                "Fallback, if you have the Google Cloud SDK:\n"
                "  gsutil -m cp gs://gresearch/robotics/droid_raw/1.0.1/"
                "aggregated-annotations-030724.json data/droid_annotations.json"
            ) from exc
    return json.loads(path.read_text())


_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse every run of whitespace to a single space, then strip.

    The annotation file has 794 strings with a trailing newline and 267 with a
    double space. Nominal agreement compares strings exactly, so without this
    identical instructions that differ by a stray space count as disagreement.
    """
    return _WHITESPACE.sub(" ", text).strip()


def flatten(raw: dict, limit: int | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """One row per annotation. Returns (episode_ids, texts, annotator_slots).

    Empty instructions are dropped. A slot can also be missing: 12,500 of the
    50,092 episodes carry only `language_instruction1`, hence `.get`.
    """
    episodes, texts, slots = [], [], []
    dropped_empty = 0
    for i, (episode, fields) in enumerate(raw.items()):
        if limit and i >= limit:
            break
        for slot in (1, 2, 3):
            text = normalize(fields.get(f"language_instruction{slot}") or "")
            if not text:
                dropped_empty += 1
                continue
            episodes.append(episode)
            texts.append(text)
            slots.append(slot)
    if dropped_empty:
        print(f"  dropped {dropped_empty:,} empty or missing instruction slots")
    return np.array(episodes), texts, np.array(slots)


def corpus_tag(texts: list[str]) -> str:
    """Cache key built from the model name and a hash of every text.

    The row count alone is not enough: a parsing change such as whitespace
    normalisation keeps the count but changes the strings.
    """
    h = hashlib.sha256(MODEL.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return f"{len(texts)}_{h.hexdigest()[:12]}"


def embed(texts: list[str], tag: str) -> np.ndarray:
    """Sentence embeddings, cached to disk keyed by corpus content."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"embeddings_{tag}.npy"
    if path.exists():
        print(f"  using cached embeddings at {path}")
        return np.load(path)

    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  embedding {len(texts):,} sentences with {MODEL} on {device}")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    np.save(path, emb)
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only use the first N episodes")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap replicates")
    ap.add_argument("--top", type=int, default=15, help="worst episodes to print")
    args = ap.parse_args()

    raw = fetch_annotations()
    print(f"\nloaded {len(raw):,} episodes from the annotation file")

    episodes, texts, slots = flatten(raw, args.limit)
    n_ep = len(set(episodes.tolist()))
    print(f"  {len(texts):,} annotations across {n_ep:,} episodes")
    counts = np.bincount(np.unique(episodes, return_inverse=True)[1])
    for k in sorted(set(counts.tolist())):
        print(f"    {int((counts == k).sum()):,} episodes with {k} annotation(s)")

    emb = embed(texts, tag=corpus_tag(texts))

    print("\n" + "=" * 66)
    print("DO THE ANNOTATORS AGREE?")
    print("=" * 66)
    sem = alpha_semantic(episodes, emb)
    nom = alpha_nominal(episodes, texts)
    print(f"  semantic alpha (do they mean the same?)  {sem.alpha:+.4f}")
    print(f"  nominal alpha  (do they type the same?)  {nom.alpha:+.4f}")
    print(f"  paraphrase gap                           {sem.alpha - nom.alpha:+.4f}")
    print(f"\n  observed disagreement  {sem.observed:.4f}")
    print(f"  expected disagreement  {sem.expected:.4f}")
    print(f"  ({sem.n_units:,} pairable episodes, {sem.n_pairable:,} annotations)")

    print(f"\n  bootstrapping ({args.boot} replicates, resampling episodes)...")
    t0 = time.perf_counter()
    point, lo, hi = bootstrap_alpha_semantic(episodes, emb, n_boot=args.boot, seed=0)
    print(f"  semantic alpha = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   ({time.perf_counter()-t0:.1f}s)")
    print("\n  1.0 = perfect agreement | 0.0 = no better than random | <0 = worse than chance")

    print("\n" + "=" * 66)
    print("HOW MUCH VARIETY IS IN THE LANGUAGE?")
    print("=" * 66)
    uniq = len(set(texts))
    print(f"  unique instruction strings   {uniq:,} of {len(texts):,}  ({uniq/len(texts):.1%})")
    print(f"  effective rank of the space  {effective_rank(emb):.1f} of {emb.shape[1]} dims")
    # The similarity statistics need an N x N matrix, so they run on a sample.
    sample = np.random.default_rng(0).choice(len(texts), size=min(4_000, len(texts)), replace=False)
    rep = instruction_space_report(emb[sample])
    print(f"  mean nearest-neighbour sim   {rep['mean_nearest_neighbor_similarity']:.3f}  (4k sample)")
    print(f"  mean pairwise similarity     {rep['mean_pairwise_similarity']:.3f}  (4k sample)")

    print("\n" + "=" * 66)
    print("BY LAB")
    print("=" * 66)
    # Per-lab samples are much smaller than the corpus, so each lab gets its own
    # episode-clustered bootstrap and its intervals are wider.
    labs = np.array([e.split("+")[0] for e in episodes])
    rows = []
    for lab in np.unique(labs):
        m = labs == lab
        if len(set(episodes[m].tolist())) < 30:
            continue
        try:
            res = alpha_semantic(episodes[m], emb[m])
            _, lab_lo, lab_hi = bootstrap_alpha_semantic(
                episodes[m], emb[m], n_boot=args.boot, seed=0
            )
        except ValueError:
            continue
        rows.append((lab, res, lab_lo, lab_hi))
    rows.sort(key=lambda r: r[1].alpha)
    for lab, res, lab_lo, lab_hi in rows[:12]:
        print(
            f"  {lab:<10} alpha {res.alpha:+.4f}  95% CI [{lab_lo:+.4f}, {lab_hi:+.4f}]"
            f"   ({res.n_units:,} episodes)"
        )
    if len(rows) > 12:
        print(f"  ... {len(rows) - 12} more labs")

    if len(rows) >= 2:
        worst, best = rows[0], rows[-1]
        # The extreme pair is picked after seeing the data, so non-overlapping
        # intervals are only suggestive.
        verdict = (
            "do not overlap: the spread across labs is larger than sampling noise"
            if worst[3] < best[2]
            else "overlap: the ordering of labs is not resolved at this sample size"
        )
        print(f"\n  extremes ({worst[0]} vs {best[0]}) {verdict}")
        print(f"  note: extreme pair selected post hoc from {len(rows)} labs; not a corrected test")

    print("\n" + "=" * 66)
    print(f"THE {args.top} EPISODES WHERE ANNOTATORS DISAGREED MOST")
    print("=" * 66)
    scores = per_unit_disagreement(episodes, emb)
    by_ep: dict[str, list[str]] = {}
    for e, t in zip(episodes.tolist(), texts):
        by_ep.setdefault(e, []).append(t)
    for ep, score in sorted(scores.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"\n  [{score:.3f}] {ep}")
        for t in by_ep[ep]:
            print(f"       - {t}")

    out = CACHE / "droid_agreement_results.json"
    out.write_text(
        json.dumps(
            {
                "n_episodes": int(sem.n_units),
                "n_annotations": int(sem.n_pairable),
                "alpha_semantic": sem.alpha,
                "alpha_semantic_ci": [lo, hi],
                "alpha_nominal": nom.alpha,
                "observed_disagreement": sem.observed,
                "expected_disagreement": sem.expected,
                "unique_string_fraction": uniq / len(texts),
                "effective_rank": effective_rank(emb),
                "per_lab_alpha": {
                    lab: {"alpha": r.alpha, "ci": [lab_lo, lab_hi], "n_episodes": int(r.n_units)}
                    for lab, r, lab_lo, lab_hi in rows
                },
                "worst_episodes": dict(sorted(scores.items(), key=lambda kv: -kv[1])[:200]),
            },
            indent=2,
        )
    )
    print(f"\n\nresults written to {out}")


if __name__ == "__main__":
    main()
