#!/usr/bin/env python
"""Verify a v2.1 dataset made by convert_rlinf_collected_to_v21.py against its RLinf collection source.

Compared in converter order (shards by (rank, id), files by number, episodes by episode_index):
  count     : number of episodes and total frames
  per episode: length, is_success, actions and state (exact up to float tolerance), episode_index, timestamps
  videos    : both cameras of every episode have as many frames as the episode (ffprobe)
  pixels    : a sample of episodes, first / middle / last frame of each camera decoded and compared with
              the source PNG (PSNR; the videos are lossy AV1, so a high PSNR, not equality, is expected)

Usage:
    python scripts/verify_converted_dataset.py --src collected_v2/train_r1/m0 --dst data/smolvla_recap/libero10_task0_train_v2_r1_m0
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def source_episodes(src: Path, with_images: set | None = None, min_frames: int = 0):
    """Yield dicts per source episode in converter order. with_images: set of global indices to also load PNGs for."""
    shards = sorted(
        (p.parent.parent for p in src.glob("rank_*/id_*/meta/info.json")),
        key=lambda p: (int(re.search(r"rank_(\d+)", str(p)).group(1)), int(re.search(r"id_(\d+)", str(p)).group(1))),
    )
    gi = 0
    for shard in shards:
        files = sorted(shard.glob("data/chunk-*/file-*.parquet"), key=lambda p: int(p.stem.split("-")[-1]))
        for f in files:
            want_img = with_images is not None
            cols = ["episode_index", "frame_index", "actions", "state", "is_success"]
            table = pq.read_table(f, columns=cols).to_pandas()
            img_df = None
            for ep, g in table.groupby("episode_index", sort=True):
                g = g.sort_values("frame_index")
                if len(g) < min_frames:  # dropped by the converter (--min-frames)
                    continue
                d = {
                    "gi": gi, "file": str(f), "n": len(g),
                    "success": bool(g["is_success"].iloc[-1]),
                    "actions": np.stack([np.asarray(x, dtype=np.float32) for x in g["actions"]]),
                    "state": np.stack([np.asarray(x, dtype=np.float32) for x in g["state"]]),
                }
                if want_img and gi in with_images:
                    if img_df is None:
                        img_df = pq.read_table(f, columns=["episode_index", "frame_index", "image", "wrist_image"]).to_pandas()
                    e = img_df[img_df.episode_index == ep].sort_values("frame_index")
                    d["image"] = e["image"].tolist()
                    d["wrist_image"] = e["wrist_image"].tolist()
                yield d
                gi += 1


def probe_frames(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def decode_frame(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 256, 3) if len(raw) == 256 * 256 * 3 else None


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--min-frames", type=int, default=100, help="must equal the converter's --min-frames")
    ap.add_argument("--max-episodes", type=int, default=None, help="only check the first N episodes (smoke test of this script)")
    ap.add_argument("--pixel-samples", type=int, default=12, help="episodes whose frames are compared pixel-wise")
    ap.add_argument("--psnr-min", type=float, default=28.0)
    args = ap.parse_args()

    eps_meta = [json.loads(l) for l in open(args.dst / "meta" / "episodes.jsonl")]
    n_dst = len(eps_meta)
    problems: list[str] = []
    n_check = n_dst if args.max_episodes is None else min(args.max_episodes, n_dst)
    sample = set(np.linspace(0, n_check - 1, min(args.pixel_samples, n_check)).astype(int).tolist())

    src_n = src_frames = 0
    video_jobs = []
    pixel_jobs = []
    for gi, e in enumerate(source_episodes(args.src, with_images=sample, min_frames=args.min_frames)):
        src_n += 1
        src_frames += e["n"]
        if gi >= n_check:
            if args.max_episodes is not None:
                break
            continue
        m = eps_meta[gi]
        if m["episode_index"] != gi or m["length"] != e["n"]:
            problems.append(f"ep{gi}: episodes.jsonl says index {m['episode_index']} length {m['length']}, source length {e['n']}")
        p = args.dst / "data" / f"chunk-{gi // 1000:03d}" / f"episode_{gi:06d}.parquet"
        try:
            d = pd.read_parquet(p)
        except Exception as ex:  # noqa: BLE001
            problems.append(f"ep{gi}: cannot read {p.name} ({type(ex).__name__})")
            continue
        if len(d) != e["n"]:
            problems.append(f"ep{gi}: converted {len(d)} rows, source {e['n']}")
            continue
        a = np.stack([np.asarray(x, dtype=np.float32) for x in d["actions"]])
        s = np.stack([np.asarray(x, dtype=np.float32) for x in d["state"]])
        if not np.allclose(a, e["actions"], atol=1e-6):
            problems.append(f"ep{gi}: actions differ (max {np.abs(a - e['actions']).max():.3g})")
        if not np.allclose(s, e["state"], atol=1e-6):
            problems.append(f"ep{gi}: state differs (max {np.abs(s - e['state']).max():.3g})")
        if bool(d["is_success"].iloc[-1]) != e["success"]:
            problems.append(f"ep{gi}: is_success differs")
        if not (d["episode_index"] == gi).all():
            problems.append(f"ep{gi}: episode_index column wrong")
        if e["n"] > 1 and not np.allclose(np.diff(d["timestamp"].to_numpy(dtype=np.float64)), 1.0 / args.fps, atol=1e-3):
            problems.append(f"ep{gi}: timestamps not 1/{args.fps:g}")
        for cam in ("image", "wrist_image"):
            video_jobs.append((gi, cam, str(args.dst / "videos" / f"chunk-{gi // 1000:03d}" / cam / f"episode_{gi:06d}.mp4"), e["n"]))
        if "image" in e:
            for cam in ("image", "wrist_image"):
                for k in (0, e["n"] // 2, e["n"] - 1):
                    pixel_jobs.append((gi, cam, k, str(args.dst / "videos" / f"chunk-{gi // 1000:03d}" / cam / f"episode_{gi:06d}.mp4"),
                                       np.asarray(Image.open(io.BytesIO(e[cam][k]["bytes"])).convert("RGB"))))

    if args.max_episodes is None and src_n != n_dst:
        problems.append(f"episode count: source {src_n}, converted {n_dst}")
    info = json.load(open(args.dst / "meta" / "info.json")) if (args.dst / "meta" / "info.json").exists() else {}
    if info and args.max_episodes is None and (info.get("total_episodes") != src_n or info.get("total_frames") != src_frames):
        problems.append(f"info.json totals {info.get('total_episodes')}/{info.get('total_frames')} vs source {src_n}/{src_frames}")

    with ThreadPoolExecutor(16) as pool:
        for (gi, cam, path, n), got in zip(video_jobs, pool.map(lambda j: probe_frames(j[2]), video_jobs)):
            if got != n:
                problems.append(f"ep{gi}/{cam}: video has {got} frames, episode {n}")

        def _psnr(j):
            gi, cam, k, path, ref = j
            fr = decode_frame(path, k)
            return None if fr is None else psnr(fr, ref)

        vals = list(pool.map(_psnr, pixel_jobs))
    bad_px = [(j[0], j[1], j[2], v) for j, v in zip(pixel_jobs, vals) if v is None or v < args.psnr_min]
    for gi, cam, k, v in bad_px[:10]:
        problems.append(f"ep{gi}/{cam} frame {k}: PSNR {v if v is None else round(v, 1)} < {args.psnr_min}")
    ok_vals = [v for v in vals if v is not None]

    print(f"source: {src_n} episodes, {src_frames} frames | converted: {n_dst} episodes (checked {n_check})")
    print(f"videos checked: {len(video_jobs)} | pixel samples: {len(vals)}"
          + (f", PSNR min/mean {min(ok_vals):.1f}/{np.mean(ok_vals):.1f} dB" if ok_vals else ""))
    for p in problems[:20]:
        print("  PROBLEM:", p)
    print("RESULT:", "ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
