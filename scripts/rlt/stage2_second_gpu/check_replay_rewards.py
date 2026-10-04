"""Reward values stored in a replay buffer (one .pt per transition): the per-step rewards of every chunk.
   python check_replay_rewards.py <checkpoint dir, e.g. logs/smoke_*/<name>/checkpoints/global_step_N>"""
import sys, glob, collections, torch
R = sys.argv[1].rstrip("/") + "/actor/sac_components/replay_buffer/rank_0"
files = glob.glob(f"{R}/trajectory_*.pt"); print("rows:", len(files))
vals = collections.Counter(); n_pos = n_neg = n_both = n_term = 0; sums = []
for p in files:
    d = torch.load(p, map_location="cpu", weights_only=False); r = d["rewards"].reshape(-1)
    for v in r.tolist(): vals[round(v, 4)] += 1
    pos = bool((r > 0).any()); neg = bool((r < 0).any()); n_pos += pos; n_neg += neg; n_both += pos and neg; n_term += bool(d["dones"].reshape(-1).any()); sums.append(float(r.sum()))
print("distinct per-step reward values:", dict(sorted(vals.items())))
print(f"rows with a positive step reward {n_pos}, with a negative one {n_neg}, with both {n_both}, terminal rows {n_term}")
print(f"row reward sum: min {min(sums):.3f}, max {max(sums):.3f}, mean {sum(sums)/len(sums):.4f}")
