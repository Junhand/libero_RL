"""Evaluate with the recovered initial-state ids: (a) per-state base success vs kNN distance, (b) kNN OOD detection with unseen initial states, (c) the offset MLP with unseen initial states."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.cluster.hierarchy import fcluster
from scipy.stats import spearmanr, rankdata
R = "/workspace/libero_RL/tmp/hybrid"; Z = f"{R}/zrl"; D = "/workspace/libero_RL/data/smolvla_recap"
de = pd.read_parquet(f"{R}/initstate_eps.parquet"); L = np.load(f"{R}/initstate_linkage.npy")
de["cid"] = fcluster(L, 50, criterion="maxclust") - 1; de.to_parquet(f"{R}/initstate_ids.parquet")
sz = de.groupby("cid").size(); print("50 initial states: episodes per state min", sz.min(), "max", sz.max())
# base success rate per initial state (the rollouts of the base VLA: train_v2 + eval)
sr = de.groupby("cid").succ.mean(); print("success rate per initial state: min %.2f median %.2f max %.2f;  states with rate<0.1: %d, >0.9: %d" % (sr.min(), sr.median(), sr.max(), (sr < .1).sum(), (sr > .9).sum()))
print("success rate sorted:", np.round(np.sort(sr.values), 2).tolist())
def auroc(y, s):
    y = np.asarray(y, bool); r = rankdata(s); n1 = y.sum(); n0 = (~y).sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
# ---- load all frames
tags = sorted(de.src.unique()); zs, ms = [], []
for t in tags:
    zz = np.load(f"{Z}/{t}.npy"); m = pd.read_parquet(f"{Z}/{t}_meta.parquet"); m["src"] = t; zs.append(zz); ms.append(m)
M = pd.concat(ms).reset_index(drop=True); ZA = np.concatenate(zs).astype(np.float32); del zs
M = M.merge(de[["src", "ep", "cid"]], on=["src", "ep"], how="left"); assert len(M) == len(ZA); M = M.dropna(subset=["cid"]).copy(); keep = M.index.values; M["cid"] = M.cid.astype(int)
ZA = ZA[keep]; M = M.reset_index(drop=True)
print("frames", len(M), "success frames", int(M.succ.sum()))
# ---- (a) kNN distance (source-split reference, as in C1) vs initial-state difficulty
off = pd.read_parquet(f"{R}/knn_test_dist.parquet").merge(de[["src", "ep", "cid"]], on=["src", "ep"])
early = off[(off.t >= 6) & (off.t <= 30)].groupby("cid").dist.mean()
j = pd.concat([sr.rename("sr"), early.rename("d_early")], axis=1).dropna()
print(f"(a) across {len(j)} initial states: Spearman(base success rate, mean kNN distance at t=6..30) = {spearmanr(j.sr, j.d_early)[0]:.3f}")
# ---- (b) kNN failure-vs-success AUROC: unseen initial states vs unseen episodes of seen states
rng = np.random.default_rng(0); perm = rng.permutation(50); folds = np.array_split(perm, 5)
Zg = torch.from_numpy(ZA).cuda()
def knn_auroc(ref_idx, test_idx):
    ref = Zg[ref_idx]; mu, sd = ref.mean(0), ref.std(0) + 1e-6; ref = (ref - mu) / sd; out = []
    for i in range(0, len(test_idx), 4096):
        q = (Zg[test_idx[i:i+4096]] - mu) / sd; out.append(torch.cdist(q, ref).topk(20, dim=1, largest=False).values[:, -1].cpu())
    return torch.cat(out).numpy()
res_b = []
for f in range(5):
    H = np.isin(M.cid.values, folds[f]); tr_s = (~H) & M.succ.values
    d = knn_auroc(np.where(tr_s)[0], np.where(H)[0]); y = ~M.succ.values[H]; t = M.t.values[H]
    res_b.append((auroc(y, d), *[auroc(y[(t >= lo) & (t < hi)], d[(t >= lo) & (t < hi)]) for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 10**6))]))
rb = np.array(res_b); print("(b) kNN k=20, failure-vs-success AUROC with UNSEEN initial states (5 folds, 10 states held out each): all %.3f (folds %s); by t-bin [0-60,60-120,120-180,180+]: %s" % (rb[:, 0].mean(), np.round(rb[:, 0], 3).tolist(), np.round(rb[:, 1:].mean(0), 3).tolist()))
# control: same fold sizes, but hold out random EPISODES (initial states stay seen)
ep_key = M.src + "_" + M.ep.astype(str); ue = ep_key.unique(); res_c = []
for f in range(5):
    Ht = set(rng.choice(ue, size=int(len(ue) * 0.2), replace=False)); H = ep_key.isin(Ht).values; tr_s = (~H) & M.succ.values
    d = knn_auroc(np.where(tr_s)[0], np.where(H)[0]); y = ~M.succ.values[H]; t = M.t.values[H]
    res_c.append((auroc(y, d), *[auroc(y[(t >= lo) & (t < hi)], d[(t >= lo) & (t < hi)]) for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 10**6))]))
rc = np.array(res_c); print("    control (random 20%% of episodes held out, initial states seen): all %.3f; by t-bin: %s" % (rc[:, 0].mean(), np.round(rc[:, 1:].mean(0), 3).tolist()))
# difficulty-matched view: for the held-out states, AUROC within the early phase t<=30 using only the distance
