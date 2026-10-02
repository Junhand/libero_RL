#!/usr/bin/env python
"""Merge several LeRobot v2.1 datasets (same features/task/fps) into one, at file level.

Nothing is re-encoded and nothing is decoded: per-episode parquet files are rewritten with new
`episode_index` / `index` (global frame number), mp4 files are copied under the new episode number,
and meta/episodes.jsonl, episodes_stats.jsonl, info.json are rebuilt. Sources are not modified.

meta/returns_*.parquet and meta/stats.json are NOT merged here: run RECAP Step1 (compute_returns.py) on the
merged dataset (it rebuilds both) and compare with `--check-returns` of verify_merged_v21.py.

Usage:
    python scripts/merge_v21_datasets.py --dst data/smolvla_recap/libero10_task0_train_v2_merged SRC1 SRC2 ...
Sources are concatenated in the given order. Needs pyarrow (e.g. .pixi/envs/lerobot-v21/bin/python).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ("image", "wrist_image")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("srcs", nargs="+", type=Path)
    args = ap.parse_args()
    if args.dst.exists():
        sys.exit(f"{args.dst} already exists")

    infos = [json.load(open(s / "meta" / "info.json")) for s in args.srcs]
    for s, i in zip(args.srcs, infos):
        if i["codebase_version"] != "v2.1":
            sys.exit(f"{s}: not v2.1")
        for k in ("fps", "robot_type", "chunks_size"):
            if i[k] != infos[0][k]:
                sys.exit(f"{s}: {k} differs ({i[k]} vs {infos[0][k]})")
        if set(i["features"]) != set(infos[0]["features"]):
            sys.exit(f"{s}: feature set differs")
    tasks0 = (args.srcs[0] / "meta" / "tasks.jsonl").read_text()
    for s in args.srcs:
        if (s / "meta" / "tasks.jsonl").read_text() != tasks0:
            sys.exit(f"{s}: tasks.jsonl differs")
    chunk = infos[0]["chunks_size"]

    dst = args.dst
    (dst / "meta").mkdir(parents=True)
    (dst / "images").mkdir()
    g = 0  # global episode number
    frames = 0  # global frame offset
    ep_out, stats_out, report = [], [], []
    for s, info in zip(args.srcs, infos):
        eps = read_jsonl(s / "meta" / "episodes.jsonl")
        stats = {e["episode_index"]: e for e in read_jsonl(s / "meta" / "episodes_stats.jsonl")}
        if [e["episode_index"] for e in eps] != list(range(len(eps))) or len(eps) != info["total_episodes"]:
            sys.exit(f"{s}: episode indices are not 0..N-1 / total mismatch")
        n_frames_src = 0
        for e in eps:
            i = e["episode_index"]
            src_p = s / "data" / f"chunk-{i // chunk:03d}" / f"episode_{i:06d}.parquet"
            t = pq.read_table(src_p)
            n = t.num_rows
            if n != e["length"] or set(t["episode_index"].to_pylist()) != {i}:
                sys.exit(f"{src_p}: length/episode_index mismatch")
            idx = t["index"].to_numpy()
            if idx[0] != n_frames_src or (idx != idx[0] + range(n)).any():
                sys.exit(f"{src_p}: index is not contiguous from the running frame count")
            new_idx = pa.array(idx + (frames - n_frames_src), type=t.schema.field("index").type)
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                             pa.array([g] * n, type=t.schema.field("episode_index").type))
            t = t.set_column(t.schema.get_field_index("index"), "index", new_idx)
            out_p = dst / "data" / f"chunk-{g // chunk:03d}" / f"episode_{g:06d}.parquet"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, out_p)
            for cam in CAMERAS:
                v_src = s / "videos" / f"chunk-{i // chunk:03d}" / cam / f"episode_{i:06d}.mp4"
                v_dst = dst / "videos" / f"chunk-{g // chunk:03d}" / cam / f"episode_{g:06d}.mp4"
                v_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v_src, v_dst)
            ep_out.append({**e, "episode_index": g})
            stats_out.append({**stats[i], "episode_index": g})
            n_frames_src += n
            frames += n
            g += 1
        if n_frames_src != info["total_frames"]:
            sys.exit(f"{s}: frames {n_frames_src} != info total_frames {info['total_frames']}")
        report.append((str(s.name), len(eps), n_frames_src))

    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in ep_out)
    with open(dst / "meta" / "episodes_stats.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in stats_out)
    (dst / "meta" / "tasks.jsonl").write_text(tasks0)
    info = dict(infos[0])
    info.update(total_episodes=g, total_frames=frames, total_videos=g * len(CAMERAS),
                total_chunks=math.ceil(g / chunk), splits={"train": f"0:{g}"})
    json.dump(info, open(dst / "meta" / "info.json", "w"), indent=4)
    for name, n, fr in report:
        print(f"{name}: {n} episodes, {fr} frames")
    print(f"merged: {g} episodes, {frames} frames -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
