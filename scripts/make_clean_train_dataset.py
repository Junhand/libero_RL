"""Create libero10_task0_train_clean: libero10_task0_train minus BAD episodes, with
episode_index / index renumbered contiguously (LeRobot v2.1 requires 0..N-1)."""
import json, os, shutil
from pathlib import Path
import pandas as pd

SRC = Path("/workspace/libero_RL/data/smolvla_recap/libero10_task0_train")
DST = Path("/workspace/libero_RL/data/smolvla_recap/libero10_task0_train_clean")
BAD = {2599}
CHUNK = 1000

assert not DST.exists()
(DST / "meta").mkdir(parents=True)

eps = [json.loads(l) for l in open(SRC / "meta/episodes.jsonl")]
stats = [json.loads(l) for l in open(SRC / "meta/episodes_stats.jsonl")]
assert [e["episode_index"] for e in eps] == list(range(len(eps)))
keep = [e["episode_index"] for e in eps if e["episode_index"] not in BAD]
new_ep = {old: new for new, old in enumerate(keep)}
length = {e["episode_index"]: e["length"] for e in eps}
start = {}  # new global frame offset of each kept episode
off = 0
for old in keep:
    start[old] = off
    off += length[old]
total_frames = off

def rel(kind, old, key=None):
    new = new_ep[old]
    name = f"episode_{new:06d}" + (".parquet" if kind == "data" else ".mp4")
    return Path(kind) / f"chunk-{new // CHUNK:03d}" / (key or "") / name

vkeys = [p.name for p in (SRC / "videos/chunk-000").iterdir()]
for old in keep:
    src_d = SRC / "data" / f"chunk-{old // CHUNK:03d}" / f"episode_{old:06d}.parquet"
    out_d = DST / rel("data", old); out_d.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src_d)
    assert len(df) == length[old]
    df["episode_index"] = new_ep[old]
    df["index"] = start[old] + df["frame_index"]
    df.to_parquet(out_d, index=False)
    for k in vkeys:
        sv = SRC / "videos" / f"chunk-{old // CHUNK:03d}" / k / f"episode_{old:06d}.mp4"
        ov = DST / rel("videos", old, k); ov.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(sv, ov)

with open(DST / "meta/episodes.jsonl", "w") as f:
    for old in keep:
        e = dict(eps[old]); e["episode_index"] = new_ep[old]; f.write(json.dumps(e) + "\n")
with open(DST / "meta/episodes_stats.jsonl", "w") as f:
    for old in keep:
        s = stats[old]; assert s["episode_index"] == old
        n, st = new_ep[old], s["stats"]
        st["episode_index"].update(min=[n], max=[n], mean=[float(n)])
        a = start[old]; L = length[old]
        st["index"].update(min=[a], max=[a + L - 1], mean=[a + (L - 1) / 2])
        f.write(json.dumps({"episode_index": n, "stats": st}) + "\n")
shutil.copy(SRC / "meta/tasks.jsonl", DST / "meta/tasks.jsonl")
shutil.copy(SRC / "meta/stats.json", DST / "meta/stats.json")
if (SRC / "images").exists():
    os.symlink(SRC / "images", DST / "images")

info = json.load(open(SRC / "meta/info.json"))
info["total_episodes"] = len(keep)
info["total_frames"] = total_frames
info["total_videos"] = len(keep) * len(vkeys)
info["total_chunks"] = (len(keep) - 1) // CHUNK + 1
info["splits"] = {"train": f"0:{len(keep)}"}
json.dump(info, open(DST / "meta/info.json", "w"), indent=4)

r = pd.read_parquet(SRC / "meta/returns_fail300.parquet")
r = r[~r["episode_index"].isin(BAD)].copy()
r["episode_index"] = r["episode_index"].map(new_ep)
r.sort_values(["episode_index", "frame_index"]).to_parquet(DST / "meta/returns_fail300.parquet", index=False)
json.dump({"bad_removed": sorted(BAD), "old_to_new_episode": new_ep}, open(DST / "meta/episode_remap.json", "w"))
print("episodes", len(keep), "frames", total_frames, "returns rows", len(r))
