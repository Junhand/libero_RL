"""Offline check of the 2-phase gate on recorded episodes (state / actions only)."""
import sys, glob, types, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev"); sys.path.insert(0, "/workspace/libero_RL/tmp/rlt_stage2")
from rlinf.envs.sim.libero.libero_env import LiberoEnv
from diag_analyze import runs
def gate(num_phases):
    f = types.SimpleNamespace(_rlt_hold_gap_min=0.04, _rlt_hold_gap_max=0.071, _rlt_hold_min_steps=8, _rlt_num_phases=num_phases, _rlt_release_min_steps=3, num_envs=1)
    for n, v in (("_rlt_hold_count", np.int32), ("_rlt_holds_done", np.int32), ("_rlt_open_count", np.int32)): setattr(f, n, np.zeros(1, v))
    f._rlt_armed = np.ones(1, bool); f._rlt_last_confirmed = np.zeros(1, bool)
    return f
def run_episode(S, A, num_phases=2):
    f = gate(num_phases); ev = {"hold": [], "rearm": [], "flag": []}
    prev_armed = True
    for t in range(len(S)):
        new = LiberoEnv._update_rlt_hold(f, [{"robot0_gripper_qpos": S[t, 6:8]}], A[t:t + 1])
        if new[0]: ev["hold"].append(t)
        if f._rlt_armed[0] and not prev_armed and not new[0]: ev["rearm"].append(t)
        prev_armed = bool(f._rlt_armed[0]); ev["flag"].append(bool(f._rlt_armed[0] and f._rlt_holds_done[0] < num_phases))
    return ev
def episodes_collected(root):
    for fl in sorted(glob.glob(f"{root}/rank_*/id_*/data/chunk-*/file-*.parquet")):
        t = pq.read_table(fl, columns=["state", "actions", "is_success", "episode_index"]).to_pandas()
        for e, g in t.groupby("episode_index", sort=True):
            if len(g) >= 100: yield np.stack(g.state.values), np.stack(g.actions.values), bool(g.is_success.values[-1])
def episodes_lerobot(root, k=None):
    for i, fl in enumerate(sorted(glob.glob(f"{root}/data/chunk-000/episode_*.parquet"))):
        if k and i >= k: break
        d = pd.read_parquet(fl, columns=["state", "actions", "is_success"]); yield np.stack(d.state.values), np.stack(d.actions.values), bool(d.is_success.values.any())
def analyse(name, it):
    rows = []
    for S, A, succ in it:
        ev = run_episode(S, A); gap = S[:, 6] - S[:, 7]; close = A[:, 6] > 0
        H = runs(close & (gap > 0.04) & (gap < 0.071), 8)
        on = np.array(ev["flag"]); segs = runs(on, 1)
        rows.append(dict(succ=succ, nconf=len(ev["hold"]), t1=ev["hold"][0] if ev["hold"] else -1, t2=ev["hold"][1] if len(ev["hold"]) > 1 else -1,
                         rearm=ev["rearm"][0] if ev["rearm"] else -1, h1_end=H[0][1] if H else -1, nH=len(H), nseg=len(segs),
                         seg1_len=segs[0][1] - segs[0][0] if segs else 0, seg2_len=segs[1][1] - segs[1][0] if len(segs) > 1 else 0, duty=on.mean()))
    R = pd.DataFrame(rows); n = len(R); s = R[R.succ]; f = R[~R.succ]
    print(f"\n=== {name}: {n} episodes ({len(s)} success, {len(f)} failure)")
    print(f"  successes with 2 confirmed grasps: {int((s.nconf==2).sum())}/{len(s)} | failures: 0 grasps {int((f.nconf==0).sum())}, 1 grasp {int((f.nconf==1).sum())}, 2 grasps {int((f.nconf==2).sum())}")
    ok = R[(R.nconf >= 1) & (R.h1_end > 0)]
    print(f"  1st confirm t: median {R[R.t1>=0].t1.median():.0f}, p90 {R[R.t1>=0].t1.quantile(.9):.0f} | 2nd confirm t (successes): median {s[s.t2>=0].t2.median():.0f}, p90 {s[s.t2>=0].t2.quantile(.9):.0f}")
    rr = R[R.rearm >= 0]; d = (rr.rearm - rr.h1_end)
    print(f"  re-arm (release) found in {len(rr)}/{int((R.nconf>=1).sum())} episodes with a 1st grasp | re-arm minus end of the hold segment: median {d.median():.0f}, p10 {d.quantile(.1):.0f}, p90 {d.quantile(.9):.0f} | early re-arm (>2 steps before the hold ended): {int((d < -2).sum())}")
    print(f"  gate-on windows: phase1 median {R.seg1_len.median():.0f} steps; phase2 (episodes with 2 windows: {int((R.nseg>=2).sum())}) median {R[R.nseg>=2].seg2_len.median():.0f} steps | duty cycle (fraction of the episode with the actor): {R.duty.mean():.2f}")
    print(f"  segments per episode: {R.nseg.value_counts().sort_index().to_dict()}")
if __name__ == "__main__":
    R = "/workspace/libero_RL/tmp/rlt_stage2/diag"
    analyse("diag B (VLA only, eval protocol)", episodes_collected(f"{R}/B"))
    analyse("diag D (RLT actor, eval protocol)", episodes_collected(f"{R}/D"))
    analyse("train_v2_r4_m2 (VLA only, 397 episodes)", episodes_lerobot("/workspace/libero_RL/data/smolvla_recap/libero10_task0_train_v2_r4_m2"))
