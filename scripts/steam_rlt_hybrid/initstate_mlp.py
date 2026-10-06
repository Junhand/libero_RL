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
# ---- (c) offset MLP (E0 recipe) with unseen initial states
rng = np.random.default_rng(0); perm = rng.permutation(50); folds = np.array_split(perm, 5)
Tlen = {}
for t in tags:
    Tlen[t] = [json.loads(l)["length"] for l in open(f"{D}/" + (f"libero10_task0_train_v2_{t}" if t.startswith("r") else (f"libero10_task0_eval_v2_{t[5:]}" if t.startswith("eval") else "libero10_task0_sft_target3")) + "/meta/episodes.jsonl")]
M["T"] = [Tlen[s_][e_] for s_, e_ in zip(M.src, M.ep)]
M = M.sort_values(["src", "ep", "t"]).reset_index(drop=True) if False else M
M["row"] = np.arange(len(M)); G = M.groupby(["src", "ep"])
epi = []   # (rows array, succ, cid, T)
for (s_, e_), g in G:
    g = g.sort_values("t"); tt = g.t.values
    if len(g) >= 4 and (np.diff(tt) == 6).all(): epi.append((g.row.values, bool(g.succ.iloc[0]), int(g.cid.iloc[0]), int(g["T"].iloc[0]), s_, e_))
print("episodes with a complete grid", len(epi))
Lmax = float(np.percentile([x[3] for x in epi if x[1]], 95)); DS = [d for d in range(-5, 6) if d != 0]
gap = M.gap.values; cmd = M.cmd.values
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(960 * 3, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 1))
    def forward(s, a, b): return s.f(torch.cat([a, b, b - a], -1)).squeeze(-1)
def run_fold(train_eps, test_eps, name):
    trs = [x for x in train_eps if x[1]]; allz = ZA[np.concatenate([x[0] for x in trs])]; mu, sd = allz.mean(0), allz.std(0) + 1e-6
    Zn = torch.from_numpy((ZA - mu) / sd).cuda()
    I1, I2, Y = [], [], []
    for rows, _, _, T, _, _ in trs:
        n = len(rows)
        for i in range(n):
            for d in DS:
                if 0 <= i + d < n: I1.append(rows[i]); I2.append(rows[i + d]); Y.append(d * 6 * Lmax / T)
    I1 = torch.tensor(I1, device="cuda"); I2 = torch.tensor(I2, device="cuda"); Yt = torch.tensor(Y, dtype=torch.float32, device="cuda"); N = len(Yt)
    torch.manual_seed(0); net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); EP = 15
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (N // 1024))
    for ep in range(EP):
        pm = torch.randperm(N, device="cuda")
        for i in range(N // 1024):
            ix = pm[i * 1024:(i + 1) * 1024]; loss = nn.functional.smooth_l1_loss(net(Zn[I1[ix]], Zn[I2[ix]]), Yt[ix]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); sg = {True: [], False: []}; fr = []
    with torch.no_grad():
        for rows, succ, cid, T, s_, e_ in test_eps:
            n = len(rows)
            for d in (-5, -3, -1, 1, 3, 5):
                ii = np.arange(max(0, -d), min(n, n - d))
                if len(ii): p = net(Zn[rows[ii]], Zn[rows[ii + d]]).cpu().numpy(); sg[succ].append(np.mean(np.sign(p) == np.sign(d)))
            ii = np.arange(n); jj = np.minimum(ii + 5, n - 1); sc = net(Zn[rows[ii]], Zn[rows[jj]]).cpu().numpy()
            for k in range(n): fr.append((succ, sc[k], gap[rows[k]], cmd[rows[k]]))
    fr = pd.DataFrame(fr, columns=["succ", "score", "gap", "cmd"]); close = fr.cmd > 0; miss = close & (fr.gap <= 0.03); hold = close & (fr.gap >= 0.04) & (fr.gap <= 0.071); sm = miss | hold
    out = dict(sign_succ=np.mean(sg[True]), sign_fail=np.mean(sg[False]), au_hold_miss=auroc(hold[sm], fr.score[sm]), au_succ_fail=auroc(fr.succ, fr.score))
    print(f"   {name}: sign acc success {out['sign_succ']:.3f} fail {out['sign_fail']:.3f} | AUROC hold-vs-miss {out['au_hold_miss']:.3f} | AUROC success-vs-failure frames {out['au_succ_fail']:.3f}", flush=True); return out
cids = np.array([x[2] for x in epi]); R1, R2 = [], []
for f in range(5):
    Hs = set(folds[f]); te = [x for x in epi if x[2] in Hs]; tr = [x for x in epi if x[2] not in Hs]
    R1.append(run_fold(tr, te, f"UNSEEN initial states, fold {f}"))
for f in range(5):
    ids = rng.permutation(len(epi)); n_te = int(0.2 * len(epi)); te = [epi[i] for i in ids[:n_te]]; tr = [epi[i] for i in ids[n_te:]]
    R2.append(run_fold(tr, te, f"control (random episodes), fold {f}"))
for nm, RR in (("UNSEEN initial states", R1), ("control: random episodes", R2)):
    print(f"(c) MEAN {nm}: " + "  ".join(f"{k} {np.mean([r[k] for r in RR]):.3f}" for k in RR[0]))
