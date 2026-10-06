"""Recover the initial-state id of every episode: the RL token at t=0 depends only on the scene layout, so episodes of the same initial state have (almost) identical t=0 tokens."""
import glob, numpy as np, pandas as pd, torch, os, json
from scipy.cluster.hierarchy import linkage, fcluster
R = "/workspace/libero_RL/tmp/hybrid"; Z = f"{R}/zrl"
tags = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{Z}/*.npy"))
rows, zs = [], []
for t in tags:
    z = np.load(f"{Z}/{t}.npy"); m = pd.read_parquet(f"{Z}/{t}_meta.parquet")
    for e, g in m.groupby("ep"):
        if (g.t == 0).any(): i = g.index[g.t == 0][0]; zs.append(z[i].astype(np.float32)); rows.append((t, int(e), bool(g.succ.iloc[0])))
df = pd.DataFrame(rows, columns=["src", "ep", "succ"]); Z0 = np.stack(zs); print("episodes with a t=0 frame", len(df), "of sources", len(tags))
Z0 = (Z0 - Z0.mean(0)) / (Z0.std(0) + 1e-6)
L = linkage(Z0, method="average", metric="euclidean"); h = L[:, 2]
print("merge heights near the end (last 70 merges):", np.round(h[-70:], 1).tolist())
gaps = np.diff(h); k = np.argsort(gaps)[-8:][::-1]
print("largest gaps in merge heights (n_clusters after the gap, height_before -> height_after):")
for i in sorted(k, reverse=True): print(f"   {len(h) - i - 1 + 1:4d} clusters  {h[i]:.1f} -> {h[i+1]:.1f}")
for nc in (40, 50, 60, 64, 80):
    c = fcluster(L, nc, criterion="maxclust"); sz = np.bincount(c)[1:]; print(f"maxclust={nc}: sizes min {sz.min()} median {int(np.median(sz))} max {sz.max()}")
np.save(f"{R}/initstate_Z0.npy", Z0); df.to_parquet(f"{R}/initstate_eps.parquet"); np.save(f"{R}/initstate_linkage.npy", L)
