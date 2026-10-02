"""Merge episodes collected by RLinf's CollectEpisode wrapper into one LeRobot v2.1 dataset.

RLinf's `env.<train|eval>.data_collection` (export_format: lerobot) writes one LeRobot dataset per
env worker and shard (`<save_dir>/rank_<r>/id_<n>/`). Run from the SmolVLA environment
(LeRobot 0.6), these are LeRobot v3.0 datasets with PNG images. RECAP Steps 1-3 read the same
format as the RLinf RECAP datasets (LeRobot v2.x, videos), so this script re-writes all shards as a
single v2.1 dataset with LeRobot 0.3.3's `LeRobotDataset.create` (videos are encoded by LeRobot).

The columns match RLinf/RECAP-Libero10-Task0-48succ-Data: image, wrist_image, state, actions,
is_success, done (+ task). Convert the result to v3.0 for SmolVLA training with
scripts/convert_rlinf_to_lerobot_v30.sh.

Per-frame PNG decoding (two cameras x up to hundreds of frames per episode) is the main CPU cost
and is otherwise single-threaded; `add_frame`/`save_episode` on the single output `LeRobotDataset`
must stay sequential (one writer), so decoding runs in a process pool a few episodes ahead while
the main process writes episodes in order as they become ready.

Usage:
    pixi run -e lerobot-v21 python scripts/convert_rlinf_collected_to_v21.py \
        --src <log_dir>/collected_data --dst $DATA/smolvla/libero10_task0_train --max-episodes 4096
"""

import argparse
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

CAMERAS = ("image", "wrist_image")


def shard_dirs(src: Path) -> list[Path]:
    def key(p: Path) -> tuple[int, int]:
        rank = int(re.search(r"rank_(\d+)", str(p)).group(1))
        shard = int(re.search(r"id_(\d+)", str(p)).group(1))
        return rank, shard

    return sorted((p.parent.parent for p in src.glob("rank_*/id_*/meta/info.json")), key=key)


def decode_image(value) -> np.ndarray:
    if isinstance(value, dict):
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value, dtype=np.uint8)


def iter_episodes(shard: Path, min_frames: int = 0):
    """Yield (task, frames DataFrame) per episode of one v3.0 shard, in episode order.

    A data file holds one episode when episodes run to the step limit, but SEVERAL episodes
    when they end early (success termination): the LeRobot writer packs consecutive short
    episodes into one file up to its size limit. Files are therefore split by ``episode_index``.
    Files are read and yielded one at a time instead of
    concatenating the whole shard into one DataFrame first -- a 300-episode/20GB shard
    held entirely in memory (as decoded PNG-bytes columns, ~2x the on-disk size) was
    the main driver of the OOM kill (cgroup memory.oom_control) seen converting the
    full train split.
    """
    tasks = pd.read_parquet(shard / "meta/tasks.parquet")
    task_by_index = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}

    files = sorted(shard.glob("data/chunk-*/file-*.parquet"), key=lambda p: int(p.stem.split("-")[-1]))
    for f in files:
        table = pq.read_table(f).to_pandas()
        for epi, ep in table.groupby("episode_index", sort=True):
            ep = ep.sort_values("frame_index")
            if len(ep) < min_frames:
                # Artifact of the init-state pool wrapping around: an env that got reset id -1 is not reset and
                # runs one action chunk on a finished episode, which is recorded as a ~10-frame "failed" episode.
                print(f"[drop] {f} episode_index={int(epi)}: {len(ep)} frames < {min_frames} (collection artifact)", flush=True)
                continue
            yield task_by_index[int(ep["task_index"].iloc[0])], ep


def _decode_episode(args: tuple[str, list, list, list, list, list]) -> dict:
    """Worker: decode both cameras for every frame of one episode. Runs in a subprocess."""
    task, image_col, wrist_col, state_col, actions_col, is_success_col, done_col = args
    n = len(image_col)
    images = np.empty((n, *decode_image(image_col[0]).shape), dtype=np.uint8)
    wrists = np.empty((n, *decode_image(wrist_col[0]).shape), dtype=np.uint8)
    for i in range(n):
        images[i] = decode_image(image_col[i])
        wrists[i] = decode_image(wrist_col[i])
    return {
        "task": task,
        "images": images,
        "wrists": wrists,
        "state": np.asarray(state_col, dtype=np.float32),
        "actions": np.asarray(actions_col, dtype=np.float32),
        "is_success": np.asarray(is_success_col, dtype=bool),
        "done": np.asarray(done_col, dtype=bool),
    }


def _episode_jobs(shards: list[Path], max_episodes: int | None, min_frames: int = 0):
    """Yield picklable per-episode decode jobs, in the same order main() used to consume them."""
    n = 0
    for shard in shards:
        for task, ep in iter_episodes(shard, min_frames):
            if max_episodes is not None and n >= max_episodes:
                return
            yield (
                task,
                ep["image"].tolist(),
                ep["wrist_image"].tolist(),
                ep["state"].tolist(),
                ep["actions"].tolist(),
                ep["is_success"].tolist(),
                ep["done"].tolist(),
            )
            n += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="CollectEpisode save_dir")
    parser.add_argument("--dst", type=Path, required=True, help="output v2.1 dataset (must not exist)")
    parser.add_argument("--max-episodes", type=int, default=None, help="keep the first N episodes")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=100,
                        help="drop episodes shorter than this (collection artifacts; real episodes are >= ~250 frames)")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="process-pool size for decoding frames ahead of the (sequential) writer",
    )
    args = parser.parse_args()

    shards = shard_dirs(args.src)
    if not shards:
        raise SystemExit(f"no LeRobot shards under {args.src}")
    print(f"{len(shards)} shards, decoding with {args.workers} worker processes")

    out = None
    n_episodes = n_success = 0

    def consume(decoded: dict) -> None:
        nonlocal out, n_episodes, n_success
        if out is None:
            h, w, c = decoded["images"].shape[1:]
            features = {
                cam: {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channel"]}
                for cam in CAMERAS
            }
            features["state"] = {
                "dtype": "float32",
                "shape": (decoded["state"].shape[1],),
                "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]},
            }
            features["actions"] = {
                "dtype": "float32",
                "shape": (decoded["actions"].shape[1],),
                "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
            }
            features["is_success"] = {"dtype": "bool", "shape": (1,), "names": None}
            features["done"] = {"dtype": "bool", "shape": (1,), "names": None}
            out = LeRobotDataset.create(
                repo_id=f"local/{args.dst.name}",
                fps=args.fps,
                features=features,
                root=args.dst,
                robot_type="panda",
                use_videos=True,
                image_writer_threads=4,
            )

        n = decoded["images"].shape[0]
        for i in range(n):
            out.add_frame(
                {
                    "image": decoded["images"][i],
                    "wrist_image": decoded["wrists"][i],
                    "state": decoded["state"][i],
                    "actions": decoded["actions"][i],
                    "is_success": decoded["is_success"][i].reshape(1),
                    "done": decoded["done"][i].reshape(1),
                },
                task=decoded["task"],
            )
        out.save_episode()
        n_episodes += 1
        n_success += bool(decoded["is_success"][-1])
        print(f"{n_episodes} episodes", end="\r", flush=True)

    # Executor.map() submits every job up front with no backpressure: for a
    # 300-episode shard that means ~300 episodes' worth of raw PNG bytes pickled into
    # the task queue at once, regardless of --workers, which is what drove the OOM
    # kill converting the full train split. Keep only a small, bounded number of jobs
    # in flight instead (a few episodes ahead per worker).
    in_flight_target = args.workers * 2
    pending: list = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in _episode_jobs(shards, args.max_episodes, args.min_frames):
            pending.append(pool.submit(_decode_episode, job))
            if len(pending) >= in_flight_target:
                consume(pending.pop(0).result())
        for future in pending:
            consume(future.result())

    print()
    print(f"done: {n_episodes} episodes, success {n_success / max(n_episodes, 1):.1%} -> {args.dst}")


if __name__ == "__main__":
    main()
