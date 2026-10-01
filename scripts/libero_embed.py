"""Turn LIBERO's 546,930 AV1-encoded frames into cached image embeddings, once.

Everything downstream reads embeddings, so the video is decoded once here and
the vectors are written to disk.

Constraints on the machine this was developed on (M2 Pro, 16 GB):

* No AV1 hardware decode, so decoding is software (libdav1d via PyAV). The run
  is resumable per video file and writes into a memmap, so a crash costs one
  file.
* Sequential decode is ~200x cheaper per frame than seeking: ~2,300-2,500 img/s
  for a full-file linear decode against low tens when seeking per frame. Each
  mp4 holds ~10k frames from ~35 episodes back to back, so each file is opened
  once, decoded in presentation order, and mapped to dataset rows using the
  ``from_timestamp`` ranges in the episodes metadata. Seeking is only used by
  ``--verify``, as an independent check on that mapping.
* Memory is shared with the GPU and ``is_amp_available("mps")`` is False, so
  this runs fp32 with a modest batch.

Two encoders with opposite training recipes:

* ``dinov2``: facebook/dinov2-small, 384-d. Self-supervised and vision-only,
  so it has no language prior. This is the primary encoder.
* ``clip``: openai/clip-vit-base-patch32 image tower, 512-d. Language-supervised.

MPS has had bugs that return garbage without an error (lerobot#496), so
``--check-mps`` re-embeds a sample on CPU and fails if cosine similarity drops
below ``MPS_COSINE_FLOOR``.

Run:
    python scripts/libero_embed.py --check-mps --encoder dinov2
    python scripts/libero_embed.py --encoder dinov2      # both cameras, resumable
    python scripts/libero_embed.py --encoder clip
    python scripts/libero_embed.py --verify --encoder dinov2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
LIBERO = CACHE / "libero"

# dinov2-small (ViT-S/14, 384-d) instead of ViT-B/14: half the embed time and
# half the index memory.
ENCODERS = {
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
}
CAMERAS = ["observation.images.image", "observation.images.image2"]

# MPS throughput is flat past this (dinov2 117-123 img/s at 64/128/256, clip
# 205 at 64 and 202 at 128), so a bigger batch only costs memory.
BATCH_SIZE = 128

# A whole decoded mp4 is ~2.0 GB of uint8 (~10k x 256 x 256 x 3), so frames are
# passed to the GPU in 512-frame chunks of ~100 MB through a 4-deep queue.
DECODE_CHUNK = 512
DECODE_QUEUE_DEPTH = 4

MPS_COSINE_FLOOR = 0.9999
MPS_CHECK_N = 200


def dataset_tag() -> str:
    """Hash identifying this dataset revision, used in every cache filename.

    Covers info.json, the episodes table, the task table, and the byte size of
    every video file. Hashing the 1.94 GB of video itself would be too slow to
    do on every run.
    """
    h = hashlib.sha256()
    for rel in ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]:
        h.update((LIBERO / rel).read_bytes())
    for cam in CAMERAS:
        for p in sorted((LIBERO / "videos" / cam / "chunk-000").glob("*.mp4")):
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class Layout:
    """Episode metadata needed to place a decoded frame in the output matrix.

    Rows are ordered by (episode_index, frame_index) ascending, the same order
    the per-frame parquet files use.
    """

    episodes: pd.DataFrame
    n_rows: int
    tasks: list[str]
    task_index: np.ndarray  # per episode


def load_layout() -> Layout:
    ep = pd.read_parquet(LIBERO / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep = ep.sort_values("episode_index").reset_index(drop=True)

    info = json.loads((LIBERO / "meta" / "info.json").read_text())
    n_rows = int(info["total_frames"])

    starts = ep["dataset_from_index"].to_numpy()
    lengths = ep["length"].to_numpy()
    # Row assignment assumes the global row index is the running total of
    # episode lengths. If that were false, embeddings would land in wrong rows.
    expected = np.concatenate([[0], np.cumsum(lengths)])[:-1]
    if not np.array_equal(starts, expected):
        raise SystemExit("dataset_from_index is not the cumulative sum of episode lengths")
    if int(lengths.sum()) != n_rows:
        raise SystemExit(f"episode lengths sum to {lengths.sum()}, info.json says {n_rows}")

    tasks_df = pd.read_parquet(LIBERO / "meta" / "tasks.parquet")
    # tasks.parquet is indexed by the instruction string, with task_index as
    # its only column, so invert it.
    instructions = [""] * len(tasks_df)
    for text, idx in zip(tasks_df.index.astype(str), tasks_df["task_index"].to_numpy()):
        instructions[int(idx)] = text

    return Layout(
        episodes=ep,
        n_rows=n_rows,
        tasks=instructions,
        task_index=episode_task_index(ep),
    )


def episode_task_index(ep: pd.DataFrame) -> np.ndarray:
    """One task_index per episode, read from the per-frame data parquets.

    The episodes table has no task column; the task is only stored per frame.
    Reads the two index columns from each of the 377 data shards and fails if
    any episode has more than one task.
    """
    out = np.full(len(ep), -1, dtype=np.int64)
    for path in sorted((LIBERO / "data" / "chunk-000").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index", "task_index"])
        grouped = df.groupby("episode_index")["task_index"].agg(["min", "max"])
        if (grouped["min"] != grouped["max"]).any():
            raise SystemExit(f"{path.name}: an episode carries more than one task_index")
        out[grouped.index.to_numpy()] = grouped["min"].to_numpy()
    if (out < 0).any():
        raise SystemExit(f"{int((out < 0).sum())} episodes had no rows in data/")
    return out


def file_plan(layout: Layout, camera: str) -> list[tuple[int, np.ndarray]]:
    """For each video file: the dataset rows its frames map to, in decode order.

    Returns an explicit row-index array per file, built from each episode's own
    ``dataset_from_index``. A (start, stop) slice would give the same result on
    this dataset but assumes the episodes sharing a file are contiguous in
    episode_index.
    """
    ep = layout.episodes
    fidx = ep[f"videos/{camera}/file_index"].to_numpy()
    fts = ep[f"videos/{camera}/from_timestamp"].to_numpy()
    starts = ep["dataset_from_index"].to_numpy()
    lengths = ep["length"].to_numpy()

    plan = []
    for f in np.unique(fidx):
        sel = np.flatnonzero(fidx == f)
        # Visit episodes in from_timestamp order, which is the decode order.
        # episode_index order happens to agree here but is not guaranteed to.
        sel = sel[np.argsort(fts[sel], kind="stable")]
        offsets = np.round(fts[sel] * 10.0).astype(np.int64)
        if not np.array_equal(offsets, np.concatenate([[0], np.cumsum(lengths[sel])])[:-1]):
            raise SystemExit(f"{camera} file {f}: from_timestamps are not back-to-back")
        rows = np.concatenate([np.arange(s, s + n) for s, n in zip(starts[sel], lengths[sel])])
        plan.append((int(f), rows))
    return plan


def decode_file(path: Path, expected: int):
    """Every frame of one mp4, in presentation order, as uint8 NHWC chunks.

    Yields ``(offset, images)`` where offset is the frame's position within the
    file. PyAV yields frames in decode order, so this fails if timestamps are
    not strictly increasing. It also fails if the frame count differs from
    ``expected``, since a short file would shift every later row.
    """
    seen = 0
    buf: list[np.ndarray] = []
    last_pts = None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if last_pts is not None and frame.pts <= last_pts:
                raise SystemExit(f"{path.name}: frames are not in presentation order at {seen}")
            last_pts = frame.pts
            buf.append(frame.to_ndarray(format="rgb24"))
            if len(buf) == DECODE_CHUNK:
                yield seen, np.stack(buf)
                seen += len(buf)
                buf = []
    if buf:
        yield seen, np.stack(buf)
        seen += len(buf)
    if seen != expected:
        raise SystemExit(f"{path.name}: decoded {seen} frames, metadata expects {expected}")


def decode_worker(work, out: queue.Queue) -> None:
    """Decode files ahead of the GPU on a background thread.

    PyAV releases the GIL inside libdav1d, so decode (~2,400 img/s) overlaps
    with the ViT forward pass (~120 img/s). An ``("eof", f)`` marker follows
    each file so the consumer only records a file as complete once all of its
    chunks have been written.
    """
    try:
        for file_index, path, rows in work:
            for offset, images in decode_file(path, len(rows)):
                out.put(("chunk", file_index, rows[offset : offset + len(images)], images))
            out.put(("eof", file_index, None, None))
    except BaseException as exc:  # re-raised by the consumer
        out.put(exc)
    else:
        out.put(None)


def seek_frame(path: Path, timestamp: float) -> np.ndarray:
    """One frame at a wall-clock offset, decoded by seeking (slow).

    Used only by ``--verify``. Shares no code with the sequential reader so the
    two can be checked against each other.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = int(round(timestamp / float(stream.time_base)))
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(video=0):
            if frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
    raise SystemExit(f"{path.name}: no frame at t={timestamp}")


class Embedder:
    """One encoder on one device, called with a uint8 image batch.

    Preprocessing runs on CPU even when the model is on MPS. MPS's bicubic
    resize differs from the CPU kernel by up to ~0.35 in normalised units,
    which would make the MPS-vs-CPU check measure the resampler and not the
    model.
    """

    def __init__(self, key: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        self.key = key
        self.model_id = ENCODERS[key]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, backend="torchvision")
        loader = CLIPVisionModelWithProjection if key == "clip" else AutoModel
        self.model = loader.from_pretrained(self.model_id, dtype=torch.float32).to(device).eval()
        self.dim = 512 if key == "clip" else int(self.model.config.hidden_size)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Embed an NHWC uint8 batch to (N, dim) float32.

        Uses CLIP's projected ``image_embeds`` and DINOv2's ``pooler_output``
        (the CLS token after the final layernorm). Vectors are not
        L2-normalised here; the index does that.
        """
        out = np.empty((len(images), self.dim), dtype=np.float32)
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        with torch.inference_mode():
            for i in range(0, len(images), BATCH_SIZE):
                chunk = tensor[i : i + BATCH_SIZE]
                px = self.processor(images=chunk, return_tensors="pt")["pixel_values"]
                res = self.model(pixel_values=px.to(self.device))
                vec = res.image_embeds if self.key == "clip" else res.pooler_output
                out[i : i + len(chunk)] = vec.float().cpu().numpy()
        return out


def cam_slug(camera: str) -> str:
    return camera.rsplit(".", 1)[-1]


def cache_path(key: str, camera: str, n_rows: int, tag: str) -> Path:
    return CACHE / f"libero_emb_{key}_{cam_slug(camera)}_{n_rows}_{tag}.npy"


def manifest_path(n_rows: int, tag: str) -> Path:
    return CACHE / f"libero_index_{n_rows}_{tag}.npz"


def write_manifest(layout: Layout, tag: str) -> Path:
    """Write the per-row episode/frame/task index shared by all embedding files.

    Row order matches every embedding matrix. The instruction strings are
    stored too, so the analysis does not need to reopen the LIBERO metadata.
    """
    path = manifest_path(layout.n_rows, tag)
    if path.exists():
        return path
    lengths = layout.episodes["length"].to_numpy()
    episode_index = np.repeat(layout.episodes["episode_index"].to_numpy(), lengths).astype(np.int32)
    frame_index = np.concatenate([np.arange(n) for n in lengths]).astype(np.int32)
    np.savez(
        path,
        episode_index=episode_index,
        frame_index=frame_index,
        task_index=layout.task_index[episode_index].astype(np.int32),
        episode_task_index=layout.task_index.astype(np.int32),
        episode_length=lengths.astype(np.int32),
        episode_row_start=layout.episodes["dataset_from_index"].to_numpy().astype(np.int64),
        tasks=np.array(layout.tasks, dtype=object),
        dataset_tag=np.array(tag),
    )
    return path


def embed_camera(key: str, camera: str, layout: Layout, tag: str, device: str) -> Path:
    """Embed every frame of one camera, resumable per video file.

    Writes into a ``.part`` memmap with a progress file listing the video files
    already done. An interrupted run loses at most one file (~90 s of GPU) and
    resumes when the command is rerun.
    """
    final = cache_path(key, camera, layout.n_rows, tag)
    if final.exists():
        print(f"  [{key}/{cam_slug(camera)}] cached at {final.name}")
        return final

    embedder = Embedder(key, device)
    part = final.with_suffix(".part.npy")
    progress = final.with_suffix(".progress.json")
    done: set[int] = set()
    if part.exists() and progress.exists():
        done = set(json.loads(progress.read_text())["files_done"])
        print(f"  [{key}/{cam_slug(camera)}] resuming, {len(done)} video files already embedded")
    mm = np.lib.format.open_memmap(
        part, mode="r+" if part.exists() else "w+", dtype=np.float32,
        shape=(layout.n_rows, embedder.dim),
    )

    plan = [(f, rows) for f, rows in file_plan(layout, camera) if f not in done]
    video_dir = LIBERO / "videos" / camera / "chunk-000"
    work = [(f, video_dir / f"file-{f:03d}.mp4", rows) for f, rows in plan]

    q: queue.Queue = queue.Queue(maxsize=DECODE_QUEUE_DEPTH)
    producer = threading.Thread(target=decode_worker, args=(work, q), daemon=True)
    producer.start()

    t0 = time.perf_counter()
    n_done = 0
    total = sum(len(rows) for _, rows in plan)
    while True:
        item = q.get()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            break
        kind, f, rows, images = item
        if kind == "chunk":
            mm[rows] = embedder(images)
            del images
            n_done += len(rows)
            continue
        mm.flush()
        done.add(f)
        progress.write_text(json.dumps({"files_done": sorted(done)}))
        rate = n_done / (time.perf_counter() - t0)
        eta = (total - n_done) / rate if rate else 0.0
        print(
            f"  [{key}/{cam_slug(camera)}] file {f:3d}  {n_done:>7,}/{total:,} frames"
            f"  {rate:6.1f} img/s  eta {eta / 60:5.1f} min",
            flush=True,
        )

    if n_done != total:
        raise SystemExit(f"embedded {n_done} frames, planned {total}")
    wall = time.perf_counter() - t0
    del mm
    part.rename(final)
    progress.unlink(missing_ok=True)

    sidecar = final.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "encoder": key,
                "model_id": embedder.model_id,
                "dims": embedder.dim,
                "camera": camera,
                "n_rows": layout.n_rows,
                "dtype": "float32",
                "batch_size": BATCH_SIZE,
                "device": device,
                # Timings cover this invocation only; a resumed run embeds
                # fewer frames than the whole camera.
                "wall_seconds_this_run": round(wall, 1),
                "images_per_second_this_run": round(n_done / wall, 1) if wall else None,
                "frames_embedded_this_run": n_done,
                "resumed": n_done != layout.n_rows,
                "row_order": "(episode_index, frame_index) ascending",
                "pooling": "image_embeds" if key == "clip" else "pooler_output (CLS post-LN)",
                "l2_normalised": False,
                "dataset_tag": tag,
                "manifest": manifest_path(layout.n_rows, tag).name,
            },
            indent=2,
        )
    )
    print(f"  [{key}/{cam_slug(camera)}] wrote {final.name} in {wall / 60:.1f} min")
    return final


def check_mps(key: str, layout: Layout) -> None:
    """Embed the same images on MPS and CPU and exit if they differ.

    See lerobot#496, where an MPS transfer returned garbage with no error.
    fp32 differences between backends are around 1e-4 absolute, which keeps
    cosine above MPS_COSINE_FLOOR.
    """
    cam = CAMERAS[0]
    path = LIBERO / "videos" / cam / "chunk-000" / "file-000.mp4"
    with av.open(str(path)) as container:
        images = []
        for frame in container.decode(video=0):
            images.append(frame.to_ndarray(format="rgb24"))
            if len(images) >= MPS_CHECK_N:
                break
    images = np.stack(images)

    print(f"[mps-check] {key}: {len(images)} images, mps vs cpu")
    a = Embedder(key, "mps")(images)
    b = Embedder(key, "cpu")(images)
    max_abs = float(np.abs(a - b).max())
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    min_cos = float(cos.min())
    print(f"  max |mps - cpu| = {max_abs:.3e}")
    print(f"  min cosine      = {min_cos:.8f}")
    if min_cos < MPS_COSINE_FLOOR:
        raise SystemExit(
            f"MPS and CPU disagree (min cosine {min_cos:.6f} < {MPS_COSINE_FLOOR}). "
            "Do not ship this cache."
        )
    print("  verdict: agree to fp32 noise")


def verify_alignment(key: str, layout: Layout, tag: str, n: int, device: str) -> None:
    """Recompute a few cached rows via the seek path and compare.

    Catches an off-by-one in the mapping from decode position to dataset row,
    which would otherwise go unnoticed because a shifted cache still has the
    right shape and norms. Picks are spread across video files, episodes, and
    both cameras.
    """
    rng = np.random.default_rng(0)
    ep = layout.episodes
    embedders = {cam: None for cam in CAMERAS}

    picks = []
    for i in range(n):
        cam = CAMERAS[i % len(CAMERAS)]
        # Spread picks over files so they do not all land in one mp4.
        fidx = ep[f"videos/{cam}/file_index"].to_numpy()
        target_file = sorted(np.unique(fidx))[int(i * (len(np.unique(fidx)) - 1) / max(n - 1, 1))]
        candidates = np.flatnonzero(fidx == target_file)
        e = int(rng.choice(candidates))
        f = int(rng.integers(0, ep["length"].to_numpy()[e]))
        picks.append((cam, e, f))

    print(f"[verify] {key}: {len(picks)} spot-checks against the seek path")
    for cam, e, f in picks:
        mat = np.load(cache_path(key, cam, layout.n_rows, tag), mmap_mode="r")
        row = int(ep["dataset_from_index"].to_numpy()[e]) + f
        cached = np.asarray(mat[row], dtype=np.float32)

        file_index = int(ep[f"videos/{cam}/file_index"].to_numpy()[e])
        t = float(ep[f"videos/{cam}/from_timestamp"].to_numpy()[e]) + f / 10.0
        img = seek_frame(LIBERO / "videos" / cam / "chunk-000" / f"file-{file_index:03d}.mp4", t)

        if embedders[cam] is None:
            embedders[cam] = Embedder(key, device)
        fresh = embedders[cam](img[None])[0]
        cos = float(
            fresh @ cached / (np.linalg.norm(fresh) * np.linalg.norm(cached) + 1e-12)
        )
        flag = "ok" if cos > 0.9999 else "MISMATCH"
        print(
            f"  {cam_slug(cam):<7} ep {e:>5} frame {f:>4} (mp4 {file_index:>3}, row {row:>7})"
            f"  cosine {cos:.6f}  {flag}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--camera", choices=CAMERAS, default=None, help="default: both")
    ap.add_argument("--check-mps", action="store_true", help="MPS vs CPU sanity check, then exit")
    ap.add_argument("--verify", action="store_true", help="spot-check cached rows, then exit")
    ap.add_argument("--spot-checks", type=int, default=5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    layout = load_layout()
    tag = dataset_tag()
    print(f"libero: {layout.n_rows:,} frames x {len(CAMERAS)} cameras, dataset tag {tag}")

    if args.check_mps:
        check_mps(args.encoder, layout)
        return

    if args.verify:
        verify_alignment(args.encoder, layout, tag, args.spot_checks, args.device)
        return

    print(f"manifest: {write_manifest(layout, tag).name}")
    cameras = [args.camera] if args.camera else CAMERAS
    for cam in cameras:
        embed_camera(args.encoder, cam, layout, tag, args.device)


if __name__ == "__main__":
    main()
