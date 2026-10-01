#!/usr/bin/env python
"""Integrity check of RLinf CollectEpisode output (LeRobot v3.0 shards, PNG frames).

Meant for outputs of runs that may have been killed (e.g. Ray OOM). Checks, per shard and episode:

  shard   : meta/info.json, meta/tasks.parquet, meta/stats.json load
  file    : parquet opens and reads completely (all columns)
  episode : frame_index == 0..n-1, n <= max steps, `done` is True exactly once and on the last frame,
            is_success constant, an unsuccessful episode ran to the step limit (else it was cut short),
            a successful one ends on a chunk boundary (n % 10 == 0, informational),
            timestamps advance by 1/fps, actions (n,7) and state (n,8) finite,
            the episode is not split over several files
  images  : every PNG of both cameras decodes to the expected shape (--no-images to skip)

Files modified within --min-age-min minutes are reported as IN-PROGRESS (a collection may still be writing
them) and are not counted as failures. Nothing is modified or deleted.

Usage:
    python scripts/verify_collected_integrity.py collected_v2/eval/m0 collected_v2/eval/m1 ...
    python scripts/verify_collected_integrity.py collected_v2          # every shard below
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def check_images(path_str: str):
    """Decode every PNG of one file. Returns (n_frames, n_bad, first_error)."""
    df = pd.read_parquet(path_str, columns=["image", "wrist_image"])
    bad, first = 0, ""
    for col in ("image", "wrist_image"):
        for v in df[col]:
            try:
                im = Image.open(io.BytesIO(v["bytes"]))
                im.load()
                if im.size != (256, 256) or im.mode not in ("RGB", "RGBA"):
                    raise ValueError(f"unexpected image {im.size} {im.mode}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                first = first or f"{type(e).__name__}: {str(e)[:60]}"
    return len(df), bad, first


def check_file(path: Path, fps: float, max_steps: int, chunk: int):
    """Returns (episodes_found, problems, notes)."""
    problems, notes, eps = [], [], []
    try:
        df = pq.read_table(path).to_pandas()
    except Exception as e:  # noqa: BLE001
        return eps, [f"unreadable parquet ({type(e).__name__})"], notes
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        n = len(g)
        tag = f"ep{int(ep)}"
        eps.append(int(ep))
        if not np.array_equal(g["frame_index"].to_numpy(), np.arange(n)):
            problems.append(f"{tag}: frame_index is not 0..{n - 1}")
        if n > max_steps:
            problems.append(f"{tag}: {n} frames > max {max_steps}")
        done = g["done"].to_numpy(dtype=bool)
        if done.sum() != 1 or not done[-1]:
            problems.append(f"{tag}: done flag count={int(done.sum())}, last={bool(done[-1])} (episode cut short?)")
        succ = g["is_success"].to_numpy(dtype=bool)
        if succ.min() != succ.max():
            problems.append(f"{tag}: is_success changes inside the episode")
        if not succ[-1] and n != max_steps:
            problems.append(f"{tag}: failed episode has {n} < {max_steps} frames (cut short)")
        if succ[-1] and n % chunk != 0:
            notes.append(f"{tag}: success ends at {n} (not a multiple of {chunk})")
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        if n > 1 and not np.allclose(np.diff(ts), 1.0 / fps, atol=1e-3):
            problems.append(f"{tag}: timestamps do not advance by 1/{fps:g}")
        a = np.stack([np.asarray(x, dtype=np.float32) for x in g["actions"]])
        s = np.stack([np.asarray(x, dtype=np.float32) for x in g["state"]])
        if a.shape != (n, 7) or s.shape != (n, 8):
            problems.append(f"{tag}: shapes actions {a.shape} state {s.shape}")
        if not (np.isfinite(a).all() and np.isfinite(s).all()):
            problems.append(f"{tag}: NaN/inf in actions or state")
    return eps, problems, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--min-age-min", type=float, default=5.0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    shards = []
    for s in args.src:
        shards += sorted(Path(p).parent.parent for p in glob.glob(str(Path(s) / "**" / "meta" / "info.json"), recursive=True))
    shards = sorted(set(shards))
    now = time.time()
    summary = []
    total_bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard in shards:
            issues, notes, n_eps, n_files, n_young = [], [], 0, 0, 0
            for m in ("info.json", "stats.json", "tasks.parquet"):
                try:
                    p = shard / "meta" / m
                    json.load(open(p)) if m.endswith(".json") else pd.read_parquet(p)
                except Exception as e:  # noqa: BLE001
                    issues.append(f"meta/{m}: {type(e).__name__}")
            files = sorted(shard.glob("data/chunk-*/file-*.parquet"), key=lambda p: int(p.stem.split("-")[-1]))
            ep_files = defaultdict(list)
            young = set()
            for f in files:
                n_files += 1
                if (now - f.stat().st_mtime) / 60 < args.min_age_min:
                    young.add(f)
                    n_young += 1
                    continue
                eps, pr, nt = check_file(f, args.fps, args.max_steps, args.chunk)
                n_eps += len(eps)
                for e in eps:
                    ep_files[e].append(f.name)
                issues += [f"{f.name}: {x}" for x in pr]
                notes += [f"{f.name}: {x}" for x in nt]
            for e, fl in ep_files.items():
                if len(fl) > 1:
                    issues.append(f"ep{e} is split over files {fl}")
            if not args.no_images:
                futs = {f: pool.submit(check_images, str(f)) for f in files if f not in young}
                for f, fut in futs.items():
                    try:
                        n, bad, first = fut.result()
                        if bad:
                            issues.append(f"{f.name}: {bad} of {2 * n} PNG(s) do not decode ({first})")
                    except Exception as e:  # noqa: BLE001
                        issues.append(f"{f.name}: image check failed ({type(e).__name__})")
            status = "OK" if not issues else "PROBLEMS"
            if issues:
                total_bad += 1
            summary.append((str(shard), status, n_eps, n_files, n_young, issues, notes))

    for name, status, n_eps, n_files, n_young, issues, notes in summary:
        extra = f", {n_young} IN-PROGRESS file(s) skipped" if n_young else ""
        print(f"[{status:8s}] {name}: {n_eps} episodes in {n_files} file(s){extra}")
        for x in issues[:12]:
            print(f"           - {x}")
        if len(issues) > 12:
            print(f"           - ... {len(issues) - 12} more")
        if notes:
            print(f"           (info: {len(notes)} success episode(s) not ending on a chunk boundary)")
    print(f"\n{len(summary)} shard(s) checked, {total_bad} with problems")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
