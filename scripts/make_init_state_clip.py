#!/usr/bin/env python3
"""Cut the clip of ONE initial state (trial id) out of the video of a LIBERO-10 Task 0 evaluation run.

The evaluation (2 envs x 25 passes, max_steps_per_rollout_epoch=13000, task_id_filter=[0]) records one video
(<log dir>/video/eval/seed_0/0.mp4) with the two envs side by side. Trial ids 0..49 are the 50 fixed LIBERO initial
states of Task 0; env 0 plays the even ids, env 1 the odd ones, pass i plays ids 2i and 2i+1. So trial t is pass
t // 2, left half of the frame if t is even, right half if t is odd. Every episode lasts 520 frames.

  python scripts/make_init_state_clip.py <0.mp4> <trial id 0..49> <out.mp4> [--scale 512]
Use the same trial id for every model to compare them on the same initial state. Run the evaluations with the same
settings (2 envs, same noise_seed) so that the layout above holds.
"""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("trial", type=int); ap.add_argument("out"); ap.add_argument("--scale", type=int, default=512)
ap.add_argument("--passes", type=int, default=25)
a = ap.parse_args()
if not 0 <= a.trial < 2 * a.passes:
    sys.exit(f"trial id must be in 0..{2 * a.passes - 1}")
probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                 "stream=nb_read_packets,width,height", "-of", "csv=p=0", a.video]).decode().strip().split(",")
w, h, n = int(probe[0]), int(probe[1]), int(probe[2])
if w != 2 * h:
    sys.exit(f"expected two square env tiles side by side (width = 2 x height), got {w}x{h}: run the evaluation with 2 envs")
per = n / a.passes; i, env = divmod(a.trial, 2)
start, end = round(i * per), round((i + 1) * per) - 1
vf = f"select='between(n,{start},{end})',setpts=N/(10*TB),crop={h}:{h}:{env * h}:0,scale={a.scale}:{a.scale}:flags=lanczos"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-vf", vf, "-r", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], check=True)
print(f"trial {a.trial}: pass {i}, env {env} ({'left' if env == 0 else 'right'} half), frames {start}..{end} -> {a.out}")
