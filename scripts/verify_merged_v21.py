#!/usr/bin/env python
"""Verify a dataset made by merge_v21_datasets.py against its sources (same order as at merge time).

Checks, for every episode:
  parquet : all columns equal to the source except episode_index (= new number) and index (= global frame number)
  videos  : both mp4 files byte-identical to the source's (they were copied, not re-encoded)
  meta    : episodes.jsonl / episodes_stats.jsonl equal apart from the episode number; info.json totals
returns (after Step1 was run on the merged dataset, `--tag fail300`):
  sidecar equals the concatenation of the sources' sidecars (return, reward, per (episode, frame)),
  and every episode follows the rule: success -> first -(n-1), last 0; failure -> first -(n-1)-300, last -300.

Usage:
    python scripts/verify_merged_v21.py --dst data/smolvla_recap/libero10_task0_train_v2_merged SRC1 SRC2 ...
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMERAS = ("image", "wrist_image")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--tag", default="fail300")
    ap.add_argument("srcs", nargs="+", type=Path)
    args = ap.parse_args()
    dst = args.dst
    chunk = json.load(open(dst / "meta" / "info.json"))["chunks_size"]
    problems: list[str] = []

    def bad(msg: str):
        if len(problems) < 30:
            problems.append(msg)

    d_eps = read_jsonl(dst / "meta" / "episodes.jsonl")
    d_stats = read_jsonl(dst / "meta" / "episodes_stats.jsonl")
    g = 0
    frames = 0
    n_success = 0
    src_ret = []
    for s in args.srcs:
        s_eps = read_jsonl(s / "meta" / "episodes.jsonl")
        s_stats = {e["episode_index"]: e for e in read_jsonl(s / "meta" / "episodes_stats.jsonl")}
        for e in s_eps:
            i = e["episode_index"]
            if g >= len(d_eps):
                bad(f"{s.name} ep{i}: merged dataset has fewer episodes")
                continue
            if d_eps[g] != {**e, "episode_index": g}:
                bad(f"ep{g}: episodes.jsonl differs from {s.name} ep{i}")
            if d_stats[g] != {**s_stats[i], "episode_index": g}:
                bad(f"ep{g}: episodes_stats.jsonl differs from {s.name} ep{i}")
            ts = pq.read_table(s / "data" / f"chunk-{i // chunk:03d}" / f"episode_{i:06d}.parquet")
            td = pq.read_table(dst / "data" / f"chunk-{g // chunk:03d}" / f"episode_{g:06d}.parquet")
            if ts.schema != td.schema:
                bad(f"ep{g}: schema differs")
            n = ts.num_rows
            if td.num_rows != n:
                bad(f"ep{g}: {td.num_rows} rows, source {n}")
            else:
                for col in ts.column_names:
                    if col in ("episode_index", "index"):
                        continue
                    if not ts[col].equals(td[col]):
                        bad(f"ep{g}: column {col} differs from {s.name} ep{i}")
                if set(td["episode_index"].to_pylist()) != {g}:
                    bad(f"ep{g}: episode_index column wrong")
                if not (td["index"].to_numpy() == frames + np.arange(n)).all():
                    bad(f"ep{g}: index is not the global frame number")
                n_success += int(bool(td["is_success"].to_pylist()[-1]))
            for cam in CAMERAS:
                a = s / "videos" / f"chunk-{i // chunk:03d}" / cam / f"episode_{i:06d}.mp4"
                b = dst / "videos" / f"chunk-{g // chunk:03d}" / cam / f"episode_{g:06d}.mp4"
                if not filecmp.cmp(a, b, shallow=False):
                    bad(f"ep{g}/{cam}: mp4 differs from {s.name} ep{i}")
            frames += n
            g += 1
        r = pq.read_table(s / "meta" / f"returns_{args.tag}.parquet", columns=["episode_index", "frame_index", "return", "reward"]).to_pandas()
        r["episode_index"] = r["episode_index"] + (g - len(s_eps))
        src_ret.append(r)
    info = json.load(open(dst / "meta" / "info.json"))
    if (g, frames) != (len(d_eps), info["total_frames"]) or info["total_episodes"] != g:
        bad(f"totals: episodes {g}/{len(d_eps)}/{info['total_episodes']}, frames {frames}/{info['total_frames']}")
    n_files = len(list((dst / "data").glob("chunk-*/*.parquet")))
    n_mp4 = len(list((dst / "videos").glob("chunk-*/*/*.mp4")))
    if n_files != g or n_mp4 != 2 * g:
        bad(f"file counts: {n_files} parquet, {n_mp4} mp4 for {g} episodes")

    import pandas as pd

    src_r = pd.concat(src_ret).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    d_r = pq.read_table(dst / "meta" / f"returns_{args.tag}.parquet", columns=["episode_index", "frame_index", "return", "reward"]).to_pandas()
    d_r = d_r.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    ret_equal = len(src_r) == len(d_r) and (src_r[["episode_index", "frame_index"]].to_numpy() == d_r[["episode_index", "frame_index"]].to_numpy()).all() \
        and np.array_equal(src_r["return"].to_numpy(), d_r["return"].to_numpy()) and np.array_equal(src_r["reward"].to_numpy(), d_r["reward"].to_numpy())
    if not ret_equal:
        bad("returns sidecar differs from the concatenated sources' sidecars")
    rule_bad = 0
    succ = {}
    for ep in range(g):
        t = pq.read_table(dst / "data" / f"chunk-{ep // chunk:03d}" / f"episode_{ep:06d}.parquet", columns=["is_success"])
        succ[ep] = bool(t["is_success"].to_pylist()[-1])
    for ep, grp in d_r.groupby("episode_index"):
        n = len(grp)
        first, last = float(grp["return"].iloc[0]), float(grp["return"].iloc[-1])
        exp_first, exp_last = (-(n - 1), 0.0) if succ[ep] else (-(n - 1) - 300.0, -300.0)
        if abs(first - exp_first) > 1e-3 or abs(last - exp_last) > 1e-3:
            rule_bad += 1
    if rule_bad:
        bad(f"{rule_bad} episode(s) violate the return rule")

    print(f"episodes {g}, frames {frames}, success {n_success} ({n_success / g:.1%}), parquet {n_files}, mp4 {n_mp4}")
    print(f"returns: {len(d_r)} rows, min {d_r['return'].min():.0f}, max {d_r['return'].max():.0f}, "
          f"equal to sources: {ret_equal}, rule violations: {rule_bad}")
    for p in problems:
        print("  PROBLEM:", p)
    print("RESULT:", "ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
