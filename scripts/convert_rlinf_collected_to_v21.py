"""Merge episodes collected by RLinf's CollectEpisode wrapper into one LeRobot v2.1 dataset.

RLinf's `env.<train|eval>.data_collection` (export_format: lerobot) writes one LeRobot dataset per
env worker and shard (`<save_dir>/rank_<r>/id_<n>/`). Run from the SmolVLA environment
(LeRobot 0.6), these are LeRobot v3.0 datasets with PNG images. RECAP Steps 1-3 read the same
format as the RLinf RECAP datasets (LeRobot v2.x, videos), so this script re-writes all shards as a
single v2.1 dataset with LeRobot 0.3.3's `LeRobotDataset.create` (videos are encoded by LeRobot).

The columns match RLinf/RECAP-Libero10-Task0-48succ-Data: image, wrist_image, state, actions,
is_success, done (+ task). Convert the result to v3.0 for SmolVLA training with
scripts/convert_rlinf_to_lerobot_v30.sh.

Usage:
    pixi run -e lerobot-v21 python scripts/convert_rlinf_collected_to_v21.py \
        --src <log_dir>/collected_data --dst $DATA/smolvla/libero10_task0_train --max-episodes 4096
"""

import argparse
import io
import re
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


def iter_episodes(shard: Path):
    """Yield (task, frames DataFrame) per episode of one v3.0 shard, in episode order."""
    tasks = pd.read_parquet(shard / "meta/tasks.parquet")
    task_by_index = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
    files = sorted(shard.glob("data/chunk-*/file-*.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])
    for _, ep in df.groupby("episode_index", sort=True):
        yield task_by_index[int(ep["task_index"].iloc[0])], ep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="CollectEpisode save_dir")
    parser.add_argument("--dst", type=Path, required=True, help="output v2.1 dataset (must not exist)")
    parser.add_argument("--max-episodes", type=int, default=None, help="keep the first N episodes")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    shards = shard_dirs(args.src)
    if not shards:
        raise SystemExit(f"no LeRobot shards under {args.src}")
    print(f"{len(shards)} shards")

    out = None
    n_episodes = n_success = 0
    for shard in shards:
        for task, ep in iter_episodes(shard):
            if args.max_episodes is not None and n_episodes >= args.max_episodes:
                break
            first = ep.iloc[0]
            if out is None:
                h, w, c = decode_image(first["image"]).shape
                features = {
                    cam: {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channel"]}
                    for cam in CAMERAS
                }
                features["state"] = {
                    "dtype": "float32",
                    "shape": (len(first["state"]),),
                    "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]},
                }
                features["actions"] = {
                    "dtype": "float32",
                    "shape": (len(first["actions"]),),
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
            for row in ep.itertuples(index=False):
                out.add_frame(
                    {
                        "image": decode_image(row.image),
                        "wrist_image": decode_image(row.wrist_image),
                        "state": np.asarray(row.state, dtype=np.float32),
                        "actions": np.asarray(row.actions, dtype=np.float32),
                        "is_success": np.asarray(row.is_success, dtype=bool).reshape(1),
                        "done": np.asarray(row.done, dtype=bool).reshape(1),
                    },
                    task=task,
                )
            out.save_episode()
            n_episodes += 1
            n_success += bool(np.asarray(ep["is_success"].iloc[-1]).reshape(-1)[0])
            print(f"{n_episodes} episodes", end="\r", flush=True)

    print()
    print(f"done: {n_episodes} episodes, success {n_success / max(n_episodes, 1):.1%} -> {args.dst}")


if __name__ == "__main__":
    main()
