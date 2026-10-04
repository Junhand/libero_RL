"""Upload one evaluation (metrics + per-episode video clips) to wandb. Usage:
   python upload_eval_to_wandb.py <log file> <run name> <description> [--dry-run]
The eval log holds the metrics line and the run's log_path; the video is video/eval/seed_0/0.mp4 under it
(env 0 only: 25 consecutive episodes of 520 steps each)."""
import re, subprocess, sys, os, glob, json, math
log, name, desc = sys.argv[1], sys.argv[2], sys.argv[3]; dry = "--dry-run" in sys.argv
t = open(log, errors="ignore").read()
m = re.search(r"runner\.logger\.log_path=(\S+)", t); logdir = m.group(1)
metrics = {}
for k in ("success_once", "success_at_end", "episode_len", "return", "reward", "num_trajectories"):
    mm = re.findall(r"eval/%s': (?:array\()?([0-9.eE+\-]+)" % k, t)
    if mm: metrics[k] = float(mm[-1])
print("log dir:", logdir); print("metrics:", metrics)
vids = sorted(glob.glob(f"{logdir}/video/eval/seed_*/*.mp4")); print("videos:", vids)
clips = []
for v in vids:
    n = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", v]).decode().strip().rstrip(","))
    steps = re.search(r"max_steps_per_rollout_epoch=(\d+)", t); k = max(1, int(steps.group(1)) // 520) if steps else max(1, round(n / 521))
    per = n / k; print(f"{os.path.basename(v)}: {n} frames -> {k} clips of ~{per:.1f} frames")
    os.makedirs(f"{logdir}/clips", exist_ok=True)
    for i in range(k):
        out = f"{logdir}/clips/env{os.path.basename(v)[:-4]}_ep{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", v, "-vf", f"select='between(n,{round(i*per)},{round((i+1)*per)-1})',setpts=N/(10*TB)", "-r", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
        clips.append(out)
print("clips:", len(clips), "total MB: %.1f" % (sum(os.path.getsize(c) for c in clips) / 1e6))
if dry: sys.exit(0)
import wandb
run = wandb.init(project="libero_RL", name=name, notes=desc, job_type="eval", config={"log": log, "protocol": "LIBERO-10 Task0, 50 fixed init states once each, max 520 steps, 10 of 50 chunk steps executed"})
run.summary.update({f"eval/{k}": v for k, v in metrics.items()})
for i, c in enumerate(clips):
    run.log({"video/episode": wandb.Video(c, fps=10, format="mp4", caption=f"{name} clip {i}")}, step=i)
run.finish(); print("uploaded:", run.url)
