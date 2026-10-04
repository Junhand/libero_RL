"""Failure diagnosis of recorded evaluation episodes (state / actions / is_success only; images are not read).
   python diag_analyze.py <collection dir> [<collection dir> ...]
Grasp state from the finger opening gap = state[6]-state[7] and the gripper command actions[6] (>0 = close):
  hold = closing and 0.04 < gap < 0.071 (an object between the fingers); miss = closing and gap <= 0.03."""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
def runs(mask, minlen):
    m = np.concatenate([[0], mask.astype(int), [0]]); d = np.diff(m)
    return [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= minlen]
def load(root):
    eps = []
    for f in sorted(glob.glob(f"{root}/rank_*/id_*/data/chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["state", "actions", "is_success", "episode_index"]).to_pandas()
        for e, g in t.groupby("episode_index", sort=True):
            if len(g) < 100: continue          # init-state-pool wrap-around artifact
            S = np.stack(g.state.values); A = np.stack(g.actions.values)
            gap = S[:, 6] - S[:, 7]; close = A[:, 6] > 0
            H = runs(close & (gap > 0.04) & (gap < 0.071), 8); Mi = runs(close & (gap <= 0.03), 3)
            succ = bool(g.is_success.values[-1])
            # success time: the episode is cut at the end of the epoch, success flag is per episode
            eps.append(dict(T=len(g), succ=succ, nH=len(H), nM=len(Mi), t1=H[0][0] if H else -1, t2=H[1][0] if len(H) > 1 else -1,
                            hold_frac=float((close & (gap > 0.04) & (gap < 0.071)).mean())))
    return pd.DataFrame(eps)
def se(p, n): return (max(p * (1 - p), 1e-9) / max(n, 1)) ** 0.5
def report(name, R):
    n = len(R); s = R.succ.mean()
    print(f"\n=== {name}: {n} episodes | task success {s:.3f} ± {se(s, n):.3f}")
    h1 = R[R.nH >= 1]; h2 = R[R.nH >= 2]; h0 = R[R.nH == 0]
    f = lambda d: f"{len(d)} ({len(d)/n:.0%}), success {d.succ.mean():.2f}" if len(d) else "0"
    print(f"  never held an object (no grasp):   {f(h0)}")
    print(f"  held >=1 time:                     {f(h1)}")
    print(f"  held >=2 times (2nd object):       {f(h2)}")
    print(f"  held exactly once, then failed:    {len(R[(R.nH==1)&(~R.succ)])} ({len(R[(R.nH==1)&(~R.succ)])/n:.0%})")
    print(f"  held >=2 but still failed:         {len(R[(R.nH>=2)&(~R.succ)])} ({len(R[(R.nH>=2)&(~R.succ)])/n:.0%})")
    if len(h1): print(f"  first hold at t: median {h1.t1.median():.0f}, p90 {h1.t1.quantile(.9):.0f} | 2nd hold at t: median {h2.t2.median() if len(h2) else float('nan'):.0f}")
    print(f"  missed grasps (closed on nothing) per episode: mean {R.nM.mean():.1f}, median {R.nM.median():.0f}")
    fails = R[~R.succ]; print(f"  FAILURES ({len(fails)}): no grasp {len(fails[fails.nH==0])} | one grasp {len(fails[fails.nH==1])} | 2+ grasps {len(fails[fails.nH>=2])}")
if __name__ == "__main__":
    for r in sys.argv[1:]:
        report(r.rstrip("/").split("/")[-1], load(r))
