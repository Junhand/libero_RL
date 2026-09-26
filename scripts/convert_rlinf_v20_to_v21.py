"""Convert a local LeRobot v2.0 dataset (RLinf RECAP rollouts) to v2.1 with LeRobot 0.3.3.

Same steps as lerobot 0.3.3 `lerobot.datasets.v21.convert_dataset_v20_to_v21`, but for a local
directory and without pushing to the Hub:
  - compute per-episode stats (meta/episodes_stats.jsonl), skipping string features such as RLinf's
    `prompt` column, which the upstream converter cannot aggregate
  - check them against meta/stats.json
  - set codebase_version to v2.1
meta/stats.json is kept (RLinf RECAP reads the `return` stats from it).

Usage:
    pixi run -e lerobot-v21 python scripts/convert_rlinf_v20_to_v21.py --root /path/to/libero10_task0_train
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, load_stats, write_episode_stats, write_info
from lerobot.datasets.v21.convert_stats import sample_episode_video_frames
from tqdm import tqdm

V21 = "v2.1"


def episode_stats(dataset: LeRobotDataset, ep_idx: int) -> dict:
    """lerobot 0.3.3 `convert_episode_stats` without string features."""
    start = dataset.episode_data_index["from"][ep_idx]
    end = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(start, end))

    stats = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "string":
            continue
        if ft["dtype"] == "video":
            data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            data = np.array(ep_data[key])
        is_image = ft["dtype"] in ["image", "video"]
        axes = (0, 2, 3) if is_image else 0
        keepdims = True if is_image else data.ndim == 1
        stats[key] = get_feature_stats(data, axis=axes, keepdims=keepdims)
        if is_image:
            stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in stats[key].items()}
    return stats


def check_stats(dataset: LeRobotDataset, all_stats: dict, reference: dict) -> None:
    """lerobot 0.3.3 `check_aggregate_stats` limited to features that have stats."""
    agg = aggregate_stats(list(all_stats.values()))
    for key, ft in dataset.features.items():
        if key not in agg or key not in reference:
            continue
        rtol, atol = (1e-2, 1e-2) if ft["dtype"] == "video" else (5e-6, 6e-5)
        for stat, val in agg[key].items():
            if stat in reference[key]:
                try:
                    np.testing.assert_allclose(val, reference[key][stat], rtol=rtol, atol=atol)
                except AssertionError:
                    logging.warning(f"episode stats differ from meta/stats.json: {key}/{stat}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="local v2.0 dataset directory")
    args = parser.parse_args()
    root = args.root.resolve()

    dataset = LeRobotDataset(f"local/{root.name}", root=root)
    if dataset.meta.info["codebase_version"] != "v2.0":
        raise SystemExit(f"{root} is {dataset.meta.info['codebase_version']}, expected v2.0")

    if (root / EPISODES_STATS_PATH).is_file():
        (root / EPISODES_STATS_PATH).unlink()
    all_stats = {}
    for ep_idx in tqdm(range(dataset.meta.total_episodes), desc="episode stats"):
        all_stats[ep_idx] = episode_stats(dataset, ep_idx)
        write_episode_stats(ep_idx, all_stats[ep_idx], root)

    check_stats(dataset, all_stats, load_stats(root))

    dataset.meta.info["codebase_version"] = V21
    write_info(dataset.meta.info, root)
    print(f"converted to {V21}: {root}")


if __name__ == "__main__":
    main()
