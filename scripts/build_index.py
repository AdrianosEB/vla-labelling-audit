"""Turn the cached embedding matrices into searchable indexes.

Builds the FAISS indexes that `scripts/search.py` queries, from embeddings
already cached on disk. There are two, one per embedding space:

* droid-text: 125,276 MiniLM instruction vectors, 384-d, searched with a
  MiniLM-embedded query. Each row carries its episode, lab, and the episode's
  disagreement score.
* libero-frame: 273,465 CLIP ViT-B/32 image-tower vectors, 512-d. Searchable
  by text because CLIP's text tower projects into the same space. The cached
  DINOv2 vectors have no text side and are not indexed.

Each gets an exact `IndexFlatIP` and an `IndexIVFFlat`. The audit uses the
exact index, which takes milliseconds at this scale. `--benchmark` scores IVF
recall@10 against the exact index as ground truth.

The metric is inner product on L2-normalised vectors, i.e. cosine similarity.
The cached matrices are stored un-normalised, so they must be normalised here.

Run:
    python scripts/build_index.py --which all
    python scripts/build_index.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from droid_agreement import fetch_annotations, flatten  # noqa: E402

from vla_label_audit.scalable import per_unit_disagreement  # noqa: E402

DATA = Path("data")
INDEX_DIR = DATA / "index"

DROID_EMB = DATA / "embeddings_125276_18b80e0e2eb4.npy"
LIBERO_EMB = DATA / "libero_emb_clip_image_273465_0e874628fd84.npy"
LIBERO_MANIFEST = DATA / "libero_index_273465_0e874628fd84.npz"

# Roughly sqrt(n) cells, the usual IVF starting point. Hard-coded so a rebuilt
# index matches the published benchmark numbers.
DROID_NLIST = 350
LIBERO_NLIST = 520


def normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows in float32, the only dtype FAISS accepts.

    Unlike `crossmodal.normalize`, this does not raise on a zero-norm row. The
    norm is clamped and the row stays at the origin, where it never wins a
    search. The current caches contain no such row.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def build_pair(vecs: np.ndarray, nlist: int, name: str) -> tuple[dict, float, float]:
    """Build and save the exact and IVF indexes for one embedding matrix.

    Returns the on-disk sizes and both build times. The IVF time includes
    training.
    """
    flat_path = INDEX_DIR / f"{name}_flat.faiss"
    ivf_path = INDEX_DIR / f"{name}_ivf.faiss"
    dim = vecs.shape[1]

    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(vecs)
    flat_secs = time.perf_counter() - t0
    faiss.write_index(flat, str(flat_path))

    # The coarse quantiser has to use the same metric as the index. An L2
    # quantiser partitions slightly differently and costs recall.
    t0 = time.perf_counter()
    quantiser = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vecs)
    ivf.add(vecs)
    ivf_secs = time.perf_counter() - t0
    faiss.write_index(ivf, str(ivf_path))

    sizes = {
        "flat_bytes": flat_path.stat().st_size,
        "ivf_bytes": ivf_path.stat().st_size,
    }
    print(f"  {flat_path.name:<26} {sizes['flat_bytes'] / 1e6:8.1f} MB  {flat_secs:6.1f}s")
    print(
        f"  {ivf_path.name:<26} {sizes['ivf_bytes'] / 1e6:8.1f} MB  {ivf_secs:6.1f}s"
        f"  (nlist={nlist})"
    )
    return sizes, flat_secs, ivf_secs


def droid_rows() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Rebuild the row order the cached DROID embeddings were written in.

    The .npy has no row labels, so the metadata has to come from
    `droid_agreement.flatten`, the function that produced the order. A
    different ordering would attach the wrong episode to every hit without any
    error. The row-count check catches a stale cache.
    """
    episodes, texts, _ = flatten(fetch_annotations())
    emb = np.load(DROID_EMB)
    if emb.shape[0] != len(texts):
        raise SystemExit(
            f"row mismatch: {DROID_EMB.name} has {emb.shape[0]:,} rows but the "
            f"flattened corpus has {len(texts):,}. The embedding cache is stale."
        )
    return episodes, texts, emb


def build_droid() -> dict:
    print("[droid-text] all-MiniLM-L6-v2 instruction vectors")
    episodes, texts, emb = droid_rows()
    print(f"  {emb.shape[0]:,} x {emb.shape[1]} vectors, {len(set(episodes.tolist())):,} episodes")

    # Recomputed because droid_agreement_results.json stores only the worst 200
    # episodes and all ~37k multiply-annotated ones are needed here.
    scores = per_unit_disagreement(episodes, emb)
    # Singly-annotated episodes have no score, so they get NaN, not zero.
    agreement = np.array([scores.get(e, np.nan) for e in episodes], dtype=np.float64)
    n_scored = int(np.isfinite(agreement).sum())
    print(f"  agreement score on {n_scored:,}/{len(agreement):,} rows ({len(scores):,} episodes)")

    labs = np.array([e.split("+")[0] for e in episodes])
    np.savez_compressed(
        INDEX_DIR / "droid_text_meta.npz",
        episode_id=episodes,
        lab=labs,
        instruction=np.array(texts, dtype=object),
        agreement_score=agreement,
    )

    sizes, flat_secs, ivf_secs = build_pair(normalise(emb), DROID_NLIST, "droid_text")
    return {
        "n_vectors": int(emb.shape[0]),
        "dims": int(emb.shape[1]),
        "source": DROID_EMB.name,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "nlist": DROID_NLIST,
        "n_episodes_scored": len(scores),
        "n_rows_scored": n_scored,
        "build_seconds": {"flat": flat_secs, "ivf": ivf_secs},
        **sizes,
    }


def build_libero() -> dict:
    print("[libero-frame] CLIP ViT-B/32 image-tower vectors")
    man = np.load(LIBERO_MANIFEST, allow_pickle=True)
    emb = np.load(LIBERO_EMB)
    if emb.shape[0] != man["episode_index"].shape[0]:
        raise SystemExit("row mismatch between the CLIP matrix and its manifest")
    print(f"  {emb.shape[0]:,} x {emb.shape[1]} vectors, {len(man['episode_length']):,} episodes")

    # Store task_index plus the 40-entry tasks table instead of 273,465
    # instruction strings.
    np.savez_compressed(
        INDEX_DIR / "libero_frame_meta.npz",
        episode_index=man["episode_index"],
        frame_index=man["frame_index"],
        task_index=man["task_index"],
        tasks=man["tasks"],
        episode_task_index=man["episode_task_index"],
    )

    sizes, flat_secs, ivf_secs = build_pair(normalise(emb), LIBERO_NLIST, "libero_frame")
    return {
        "n_vectors": int(emb.shape[0]),
        "dims": int(emb.shape[1]),
        "source": LIBERO_EMB.name,
        "encoder": "openai/clip-vit-base-patch32 (image tower, image_embeds)",
        "camera": "observation.images.image",
        "nlist": LIBERO_NLIST,
        "n_tasks": int(len(man["tasks"])),
        "build_seconds": {"flat": flat_secs, "ivf": ivf_secs},
        **sizes,
    }


def bench_one(name: str, nprobes: list[int], n_queries: int, k: int, seed: int) -> dict:
    """Recall and latency for one index family, exact vs IVF at each nprobe.

    Queries are drawn from the indexed vectors, so rank 1 is always a
    self-match. That helps both indexes equally, and recall is measured as IVF
    against exact, so it does not bias the comparison.

    Latency is single-query and single-threaded, to match interactive use.
    """
    flat = faiss.read_index(str(INDEX_DIR / f"{name}_flat.faiss"))
    ivf = faiss.read_index(str(INDEX_DIR / f"{name}_ivf.faiss"))

    rng = np.random.default_rng(seed)
    rows = rng.choice(flat.ntotal, size=n_queries, replace=False)
    queries = np.ascontiguousarray(flat.reconstruct_n(0, flat.ntotal)[rows], dtype=np.float32)

    faiss.omp_set_num_threads(1)

    def latency(index) -> tuple[float, float]:
        times = []
        for q in queries:
            t0 = time.perf_counter()
            index.search(q.reshape(1, -1), k)
            times.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(times)), float(np.percentile(times, 95))

    _, truth = flat.search(queries, k)
    med, p95 = latency(flat)
    out = {"exact": {"recall_at_10": 1.0, "median_ms": med, "p95_ms": p95}}
    print(f"  {name} exact              recall 1.000  median {med:7.3f} ms  p95 {p95:7.3f} ms")

    for np_ in nprobes:
        ivf.nprobe = np_
        _, got = ivf.search(queries, k)
        # Recall is set overlap per query. Order within the top-k is ignored
        # because IVF can reorder ties.
        hits = sum(len(set(t.tolist()) & set(g.tolist())) for t, g in zip(truth, got))
        recall = hits / (n_queries * k)
        med, p95 = latency(ivf)
        out[f"ivf_nprobe{np_}"] = {
            "recall_at_10": recall,
            "median_ms": med,
            "p95_ms": p95,
            "nprobe": np_,
        }
        print(
            f"  {name} ivf nprobe={np_:<3}      recall {recall:.3f}"
            f"  median {med:7.3f} ms  p95 {p95:7.3f} ms"
        )

    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    return out


def run_benchmark(which: str, nprobes: list[int], n_queries: int, k: int, seed: int) -> None:
    names = []
    if which in ("droid", "all"):
        names.append(("droid_text", DROID_NLIST))
    if which in ("libero", "all"):
        names.append(("libero_frame", LIBERO_NLIST))

    missing = [n for n, _ in names if not (INDEX_DIR / f"{n}_flat.faiss").exists()]
    if missing:
        raise SystemExit(f"no index for {', '.join(missing)}; run --which all first")

    manifest_path = INDEX_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    print(f"\nbenchmark: {n_queries} queries, k={k}, single-threaded")
    results = {}
    for name, nlist in names:
        results[name] = {
            "nlist": nlist,
            "n_vectors": faiss.read_index(str(INDEX_DIR / f"{name}_flat.faiss")).ntotal,
            "flat_bytes": (INDEX_DIR / f"{name}_flat.faiss").stat().st_size,
            "ivf_bytes": (INDEX_DIR / f"{name}_ivf.faiss").stat().st_size,
            "build_seconds": manifest.get(name.split("_")[0], {}).get("build_seconds"),
            "methods": bench_one(name, nprobes, n_queries, k, seed),
        }

    out = INDEX_DIR / "benchmark.json"
    out.write_text(
        json.dumps(
            {
                "n_queries": n_queries,
                "k": k,
                "seed": seed,
                "nprobes": nprobes,
                "threads": 1,
                "ground_truth": "exact IndexFlatIP over the same vectors",
                "indexes": results,
            },
            indent=2,
        )
    )
    print(f"\nbenchmark written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", choices=("droid", "libero", "all"), default="all")
    ap.add_argument(
        "--benchmark", action="store_true", help="score existing indexes instead of rebuilding"
    )
    ap.add_argument("--nprobe", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--queries", type=int, default=200, help="held-out queries for the benchmark")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        run_benchmark(args.which, args.nprobe, args.queries, args.k, args.seed)
        return

    manifest_path = INDEX_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    t0 = time.perf_counter()
    if args.which in ("droid", "all"):
        manifest["droid"] = build_droid()
    if args.which in ("libero", "all"):
        manifest["libero"] = build_libero()

    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = sum(
        f.stat().st_size for f in INDEX_DIR.iterdir() if f.suffix in (".faiss", ".npz")
    )
    print(f"\nbuilt in {time.perf_counter() - t0:.1f}s; {total / 1e6:.1f} MB in {INDEX_DIR}/")
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
