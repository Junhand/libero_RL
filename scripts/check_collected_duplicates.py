#!/usr/bin/env python
"""Detect duplicated episodes across collected rollout directories.

Three environments collecting with the same env seed and the same policy-noise seed produce
byte-identical rollouts. This script hashes the action sequence of every episode and reports
episodes that appear more than once, within one source and across sources.

Each SRC is a directory holding LeRobot parquet files, either

  * a RLinf collection ``save_dir`` (``rank_*/id_*/data/chunk-*/file-*.parquet``, LeRobot v3.0), or
  * a converted dataset (``data/chunk-*/episode_*.parquet``, LeRobot v2.1).

Usage:
    python scripts/check_collected_duplicates.py m0=collected_v2/train_r1/m0 m1=collected_v2/train_r1/m1 m2=...
    python scripts/check_collected_duplicates.py data/smolvla_recap/libero10_task0_train collected_v2/train_r1/m0

A bare path uses its directory name as the label. Exit code: 0 = no duplicates, 1 = duplicates found.

Two hashes per episode:
  full : all actions, rounded to --decimals (catches the same episode even if tiny float noise differs)
  head : the first --head-steps actions. Episodes that merely start from the same state differ in the
         first flow-matching chunk as soon as the noise differs, so an equal head means equal noise.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def find_parquets(src: Path) -> list[Path]:
    files = sorted(glob.glob(str(src / "**" / "data" / "chunk-*" / "*.parquet"), recursive=True))
    return [Path(f) for f in files]


def episode_hashes(path: Path, decimals: int, head_steps: int):
    df = pd.read_parquet(path, columns=["episode_index", "frame_index", "actions"])
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        a = np.stack([np.asarray(x, dtype=np.float32) for x in g["actions"]])
        q = np.round(a, decimals)
        full = hashlib.sha1(q.tobytes()).hexdigest()[:16]
        head = hashlib.sha1(q[:head_steps].tobytes()).hexdigest()[:16]
        yield int(ep), len(g), full, head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help="[label=]path of a collection dir or converted dataset")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--head-steps", type=int, default=30)
    ap.add_argument("--show", type=int, default=10, help="max duplicate groups to print per kind")
    args = ap.parse_args()

    by_full: dict[str, list] = defaultdict(list)
    by_head: dict[str, list] = defaultdict(list)
    counts: dict[str, int] = {}
    for item in args.src:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(item).name, item
        src = Path(path)
        files = find_parquets(src)
        if not files:
            print(f"[WARN] {label}: no parquet files under {src}", file=sys.stderr)
        n = 0
        bad = 0
        for f in files:
            try:
                rows = list(episode_hashes(f, args.decimals, args.head_steps))
            except Exception as e:  # e.g. a file truncated by a crashed collection run
                bad += 1
                print(f"[WARN] {label}: unreadable parquet skipped: {f.relative_to(src)} ({type(e).__name__})", file=sys.stderr)
                continue
            for ep, length, full, head in rows:
                where = (label, str(f.relative_to(src)), ep, length)
                by_full[full].append(where)
                by_head[head].append(where)
                n += 1
        counts[label] = n
        print(f"{label}: {n} episodes from {len(files) - bad} parquet file(s)" + (f" ({bad} unreadable skipped)" if bad else ""))

    total = sum(counts.values())
    status = 0
    for kind, groups in (("full", by_full), ("head", by_head)):
        dups = {h: w for h, w in groups.items() if len(w) > 1}
        extra = sum(len(w) - 1 for w in dups.values())
        cross = sum(1 for w in dups.values() if len({x[0] for x in w}) > 1)
        print(f"\n[{kind}] duplicate groups: {len(dups)}  (surplus episodes: {extra} of {total};"
              f" groups spanning several sources: {cross})")
        for h, w in list(dups.items())[: args.show]:
            print(f"  {h}: " + "; ".join(f"{lab}:{rel}#ep{ep}(len {ln})" for lab, rel, ep, ln in w))
        if dups:
            status = 1
    print("\nRESULT:", "DUPLICATES FOUND" if status else "no duplicates")
    return status


if __name__ == "__main__":
    sys.exit(main())
