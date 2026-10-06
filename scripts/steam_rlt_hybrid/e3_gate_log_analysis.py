"""E3: analyse the gate logs (tmp/rlt_stage2/gate_<name>.jsonl): kNN distance of the live RL token per actor-phase run, vs offline distances at the same step."""
import json, numpy as np, pandas as pd
off = pd.read_parquet("/workspace/libero_RL/tmp/hybrid/knn_test_dist.parquet")   # from build_knn_ref.py
def runs(name):
    rows = [json.loads(l) for l in open(f"/workspace/libero_RL/tmp/rlt_stage2/gate_{name}.jsonl")]; out = []
    for e in range(2):                       # 2 parallel eval envs = 2 columns
        cur = []
        for r in rows:
            if r["active"][e]: cur.append(r["dist"][e])
            elif cur: out.append(cur); cur = []
        if cur: out.append(cur)
    return out
for name in ("gall", "gnone"):
    E = runs(name); L = np.array([len(x) for x in E]); print(name, "runs", len(E), "chunks per run median", np.median(L))
    for k in (0, 1, 2, 4, 6, 9, 14, 19, 29):
        v = [x[k] for x in E if len(x) > k]; t = 10 * (k + 1); o = off[(off.t >= t - 3) & (off.t <= t + 3)]
        if len(v) >= 5: print(f" k={k:2d} t~{t:3d} n={len(v):3d} live {np.median(v):5.1f} [{np.percentile(v, 90):5.1f}] success {o[o.succ].dist.median():5.1f} failure {o[~o.succ].dist.median():5.1f}")
    for thr in (9.938, 14.367, 24.291): print(f" runs whose first chunk exceeds {thr}: {np.mean([r[0] > thr for r in E]):.2f}")
