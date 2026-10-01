"""Search the DROID and LIBERO indexes built by `scripts/build_index.py`.

Three modes:

* ``--text-to-episodes``: DROID annotations closest in meaning to a query.
  Each hit shows its lab and its episode's disagreement score.
* ``--text-to-frames``: LIBERO frames that look like a description. The query
  goes through CLIP's text tower, which shares a space with the cached image
  vectors.
* ``--episode-similar``: episodes described like a given one, using the mean
  of that episode's annotation vectors as the query.

Search is exact by default; `--ivf` uses the approximate index.

Run:
    python scripts/search.py --text-to-episodes "pick up the mug"
    python scripts/search.py --text-to-frames "a robot arm holding a bowl" -k 5
    python scripts/search.py --episode-similar IRIS+7dfa2da3+2023-04-25-11h-42m-28s
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DATA = Path("data")
INDEX_DIR = DATA / "index"

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
# Must match the checkpoint whose image tower produced the cached LIBERO
# vectors. Another CLIP checkpoint with 512 dims would search without error
# and return noise.
CLIP = "openai/clip-vit-base-patch32"


def load(name: str, use_ivf: bool, nprobe: int):
    """Open one index and its metadata sidecar.

    `faiss` is imported here, not at module scope, so the embedding worker
    (see `embed_query`) never loads it.
    """
    import faiss

    suffix = "ivf" if use_ivf else "flat"
    path = INDEX_DIR / f"{name}_{suffix}.faiss"
    if not path.exists():
        raise SystemExit(f"{path} missing; run: python scripts/build_index.py --which all")
    index = faiss.read_index(str(path))
    if use_ivf:
        index.nprobe = nprobe
    meta = np.load(INDEX_DIR / f"{name}_meta.npz", allow_pickle=True)
    return index, meta


def unit(vec: np.ndarray) -> np.ndarray:
    """Reshape and L2-normalise a query to match the indexed vectors."""
    vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def embed_query(kind: str, text: str) -> np.ndarray:
    """Embed a query string in a child process and return the vector.

    torch and faiss-cpu each ship their own OpenMP runtime on macOS, and loading
    both in one interpreter segfaults with no traceback, whichever is imported
    first. Pinning threads or setting KMP_DUPLICATE_LIB_OK does not help. So
    the child imports torch and never faiss, the parent imports faiss and never
    torch, and a temporary .npy carries the vector between them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "query.npy"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--_embed", kind, "--_text", text, "--_out", str(out)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out.exists():
            raise SystemExit(
                f"query embedding failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
            )
        return np.load(out)


def _embed_worker(kind: str, text: str, out: Path) -> None:
    """The child half of `embed_query`. Must never import faiss."""
    if kind == "minilm":
        from sentence_transformers import SentenceTransformer

        vec = SentenceTransformer(MINILM).encode([text], convert_to_numpy=True)[0]
    else:
        vec = embed_clip_text(text)
    np.save(out, unit(vec))


def embed_clip_text(text: str) -> np.ndarray:
    """Embed a query with CLIP's text tower, in the shared image/text space.

    Uses `CLIPTextModelWithProjection` because on transformers 5.x
    `CLIPModel.get_text_features` returns a `BaseModelOutputWithPooling` whose
    `pooler_output` is the pre-projection hidden state. For ViT-B/32 that is
    also 512-d, so it would search without error and return wrong results.
    """
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP).eval()
    with torch.no_grad():
        out = model(**tok([text], return_tensors="pt", padding=True))
    return out.text_embeds[0].numpy()


def text_to_episodes(query: str, k: int, use_ivf: bool, nprobe: int) -> None:
    vec = embed_query("minilm", query)
    index, meta = load("droid_text", use_ivf, nprobe)
    scores, ids = index.search(vec, k)
    agreement = meta["agreement_score"]

    print(f'\nDROID annotations most like: "{query}"')
    print(f"  {'#':>3} {'score':>7}  {'lab':<10} {'disagree':>8}  episode / instruction")
    for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
        if i < 0:  # IVF with a small nprobe can return fewer than k hits
            continue
        d = agreement[i]
        d_str = f"{d:8.4f}" if np.isfinite(d) else "     n/a"
        print(f"  {rank:>3} {s:7.4f}  {str(meta['lab'][i]):<10} {d_str}  {meta['episode_id'][i]}")
        print(f"                                    {meta['instruction'][i]}")


def text_to_frames(query: str, k: int, use_ivf: bool, nprobe: int) -> None:
    vec = embed_query("clip", query)
    index, meta = load("libero_frame", use_ivf, nprobe)
    scores, ids = index.search(vec, k)
    tasks = meta["tasks"]

    print(f'\nLIBERO frames most like: "{query}"')
    print(f"  {'#':>3} {'score':>7}  {'episode':>7} {'frame':>6}  ground-truth instruction")
    for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
        if i < 0:
            continue
        ep, fr = int(meta["episode_index"][i]), int(meta["frame_index"][i])
        print(f"  {rank:>3} {s:7.4f}  {ep:>7} {fr:>6}  {tasks[int(meta['task_index'][i])]}")


def episode_similar(episode_id: str, k: int, use_ivf: bool, nprobe: int) -> None:
    """Nearest other episodes to one episode's mean annotation vector.

    The index is per annotation, so this over-fetches and keeps the
    best-scoring annotation per episode.
    """
    index, meta = load("droid_text", use_ivf, nprobe)
    episodes = meta["episode_id"]
    rows = np.flatnonzero(episodes == episode_id)
    if rows.size == 0:
        raise SystemExit(f"episode {episode_id!r} is not in the index")

    import faiss

    # query = normalised mean of the episode's annotation vectors
    flat = faiss.read_index(str(INDEX_DIR / "droid_text_flat.faiss"))
    query = unit(np.mean([flat.reconstruct(int(r)) for r in rows], axis=0))

    agreement = meta["agreement_score"]
    d = agreement[rows[0]]
    own = f"{d:.4f}" if np.isfinite(d) else "n/a"
    print(f"\nEpisodes described most like {episode_id}")
    print(f"  query = mean of {rows.size} annotation vector(s); own disagreement {own}")
    for r in rows:
        print(f"    - {meta['instruction'][r]}")

    scores, ids = index.search(query, k * 8 + rows.size)
    print(f"\n  {'#':>3} {'score':>7}  {'lab':<10} {'disagree':>8}  episode / instruction")
    seen: set[str] = {episode_id}
    rank = 0
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        ep = str(episodes[i])
        if ep in seen:
            continue
        seen.add(ep)
        rank += 1
        dd = agreement[i]
        d_str = f"{dd:8.4f}" if np.isfinite(dd) else "     n/a"
        print(f"  {rank:>3} {s:7.4f}  {str(meta['lab'][i]):<10} {d_str}  {ep}")
        print(f"                                    {meta['instruction'][i]}")
        if rank >= k:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Internal flags used by `embed_query`; hidden from --help.
    ap.add_argument("--_embed", choices=("minilm", "clip"), help=argparse.SUPPRESS)
    ap.add_argument("--_text", help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    if "--_embed" in sys.argv:
        args = ap.parse_known_args()[0]
        _embed_worker(args._embed, args._text, Path(args._out))
        return

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text-to-episodes", metavar="QUERY", help="search DROID instructions")
    mode.add_argument("--text-to-frames", metavar="QUERY", help="search LIBERO frames")
    mode.add_argument("--episode-similar", metavar="EPISODE_ID", help="nearest other episodes")
    ap.add_argument("-k", type=int, default=10, help="how many hits to print")
    exact = ap.add_mutually_exclusive_group()
    exact.add_argument("--exact", action="store_true", default=True, help="exact search (default)")
    exact.add_argument("--ivf", dest="exact", action="store_false", help="approximate IVF search")
    ap.add_argument("--nprobe", type=int, default=32, help="IVF cells to probe (with --ivf)")
    args = ap.parse_args()

    if args.text_to_episodes:
        text_to_episodes(args.text_to_episodes, args.k, not args.exact, args.nprobe)
    elif args.text_to_frames:
        text_to_frames(args.text_to_frames, args.k, not args.exact, args.nprobe)
    else:
        episode_similar(args.episode_similar, args.k, not args.exact, args.nprobe)


if __name__ == "__main__":
    main()
