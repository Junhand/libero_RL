#!/usr/bin/env python
"""Log collected LIBERO rollouts (RLinf CollectEpisode output / converted dataset) to Weights & Biases.

Logs, for the episodes found under the given sources:
  * a summary table (length, success, gripper open/close events, end-effector path length, ...)
  * summary scalars (success rate, mean length of successes vs failures, ...)
  * line plots of the 7-dim action, the end-effector position and the gripper command over time
    for a few successful and a few failed episodes
  * videos (agent view | wrist view) of the same episodes

Default mode is OFFLINE (no credentials needed, nothing leaves this machine). Upload later with
    wandb login            # in your own terminal
    wandb sync <run dir printed at the end>
or run with --mode online after logging in.

Usage:
    python scripts/log_rollouts_to_wandb.py A=tmp/seed_test/A1 B=tmp/seed_test/B --project libero_RL --name rollout-shape
"""

from __future__ import annotations

import argparse
import glob
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# LIBERO action = (dx, dy, dz, droll, dpitch, dyaw, gripper); gripper < 0 open, > 0 close.
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_episodes(label: str, src: Path, with_images: bool):
    cols = ["episode_index", "frame_index", "actions", "state", "is_success"]
    if with_images:
        cols += ["image", "wrist_image"]
    for f in sorted(glob.glob(str(src / "**" / "data" / "chunk-*" / "*.parquet"), recursive=True)):
        df = pd.read_parquet(f, columns=cols)
        for ep, g in df.groupby("episode_index", sort=True):
            g = g.sort_values("frame_index")
            yield {
                "label": label,
                "ep": int(ep),
                "file": f,
                "actions": np.stack([np.asarray(x, dtype=np.float32) for x in g["actions"]]),
                "state": np.stack([np.asarray(x, dtype=np.float32) for x in g["state"]]),
                "success": bool(g["is_success"].iloc[-1]),
                "frames": g if with_images else None,
            }


def gripper_events(grip: np.ndarray):
    closed = grip > 0
    d = np.diff(closed.astype(int))
    closes = np.where(d == 1)[0] + 1
    opens = np.where(d == -1)[0] + 1
    if closed[0]:
        closes = np.r_[0, closes]
    return closes, opens


def summarize(e):
    a, s = e["actions"], e["state"]
    closes, opens = gripper_events(a[:, -1])
    path = float(np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1).sum()) if len(s) > 1 else 0.0
    return {
        "source": e["label"], "episode": e["ep"], "success": e["success"], "length": len(a),
        "grip_close_events": len(closes), "grip_open_events": len(opens),
        "first_close_step": int(closes[0]) if len(closes) else -1,
        "last_event_step": int(max([*closes, *opens])) if len(closes) or len(opens) else -1,
        "eef_path_len": round(path, 3), "mean_speed_xyz": round(float(np.abs(a[:, :3]).mean()), 3),
    }


def decode(png):
    return np.asarray(Image.open(io.BytesIO(png["bytes"])).convert("RGB"))


def write_video(frames_df, path: Path, fps: int):
    imgs = [np.concatenate([decode(r["image"]), decode(r["wrist_image"])], axis=1) for _, r in frames_df.iterrows()]
    h, w, _ = imgs[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.run(cmd, input=b"".join(i.tobytes() for i in imgs), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:300])
    return imgs


def contact_sheet(imgs, n=8):
    idx = np.linspace(0, len(imgs) - 1, n).astype(int)
    row = [np.asarray(Image.fromarray(imgs[i]).resize((256, 128))) for i in idx]
    return np.concatenate([np.concatenate(row[: n // 2], axis=1), np.concatenate(row[n // 2:], axis=1)], axis=0), idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help="[label=]path")
    ap.add_argument("--project", default="libero_RL")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--name", default="rollout-shape")
    ap.add_argument("--mode", default="offline", choices=["offline", "online"])
    ap.add_argument("--dir", default="tmp/wandb_rollouts", help="wandb run dir (and temp videos)")
    ap.add_argument("--n-success", type=int, default=3)
    ap.add_argument("--n-fail", type=int, default=2)
    ap.add_argument("--video-fps", type=int, default=20, help="LIBERO controls at 20 Hz (dataset fps label is 10)")
    args = ap.parse_args()

    import wandb

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = []
    for item in args.src:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(item).name, item
        eps.extend(load_episodes(label, Path(path), with_images=False))
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 1
    rows = [summarize(e) for e in eps]
    df = pd.DataFrame(rows)
    succ, fail = df[df.success], df[~df.success]
    print(df.to_string(index=False))
    print(f"\nsuccess {len(succ)}/{len(df)}; mean length success {succ.length.mean():.0f} / fail {fail.length.mean():.0f}")
    print("grip close events  success: %s | fail: %s" % (succ.grip_close_events.tolist(), fail.grip_close_events.tolist()))

    run = wandb.init(project=args.project, entity=args.entity, name=args.name, mode=args.mode, dir=str(out),
                     config={"sources": args.src, "video_fps": args.video_fps,
                             "note": "success terminates at the action-chunk boundary (10 steps); LIBERO control 20 Hz"})
    run.log({"episodes": wandb.Table(dataframe=df)})
    run.summary.update({
        "n_episodes": len(df), "success_rate": float(df.success.mean()),
        "mean_len_success": float(succ.length.mean()) if len(succ) else None,
        "mean_len_fail": float(fail.length.mean()) if len(fail) else None,
        "mean_grip_close_success": float(succ.grip_close_events.mean()) if len(succ) else None,
        "mean_grip_close_fail": float(fail.grip_close_events.mean()) if len(fail) else None,
    })

    pick = pd.concat([succ.head(args.n_success), fail.head(args.n_fail)])
    keyed = {(r["source"], r["episode"]): r for _, r in pick.iterrows()}
    sheet_logged = 0
    for e in load_all(args.src):
        k = (e["label"], e["ep"])
        if k not in keyed:
            continue
        tag = f"{'success' if e['success'] else 'fail'}/{e['label']}_ep{e['ep']}"
        t = np.arange(len(e["actions"]))
        run.log({
            f"{tag}/actions": wandb.plot.line_series(
                xs=t.tolist(), ys=[e["actions"][:, i].tolist() for i in range(7)], keys=ACTION_NAMES,
                title=f"{tag}: action", xname="step"),
            f"{tag}/eef_position": wandb.plot.line_series(
                xs=t.tolist(), ys=[e["state"][:, i].tolist() for i in range(3)], keys=["x", "y", "z"],
                title=f"{tag}: end-effector position", xname="step"),
        })
        vid = out / f"{e['label']}_ep{e['ep']}.mp4"
        imgs = write_video(e["frames"], vid, args.video_fps)
        run.log({f"{tag}/video": wandb.Video(str(vid), format="mp4")})
        sheet, idx = contact_sheet(imgs)
        run.log({f"{tag}/frames": wandb.Image(sheet, caption=f"steps {idx.tolist()} (agent view | wrist view)")})
        if sheet_logged < 1 and e["success"]:
            Image.fromarray(sheet).save(out / "success_contact_sheet.png")
            sheet_logged += 1
    run.finish()
    print("\nrun dir:", run.dir)
    return 0


def load_all(srcs):
    for item in srcs:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(item).name, item
        yield from load_episodes(label, Path(path), with_images=True)


if __name__ == "__main__":
    sys.exit(main())
