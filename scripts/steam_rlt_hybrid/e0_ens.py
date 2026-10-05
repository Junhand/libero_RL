"""E0: can a small MLP on cached RL tokens (z_rl, 960-d) predict the normalized temporal offset between two frames?
Grid: frames t % 6 == 0. Pair offsets d*6 frames, d in +-1..5 (max 30 frames, close to STEAM's k=32).
Train on successful episodes (as STEAM). Evaluate on held-out episodes: sign accuracy, Spearman, and agreement with the trained STEAM critic's A_STEAM."""
import json, glob, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from scipy.stats import rankdata
def roc_auc_score(y, s):
    y=np.asarray(y).astype(bool); r=rankdata(s); n1=y.sum(); n0=(~y).sum(); return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"
torch.manual_seed(0); rng = np.random.default_rng(0)
ORDER = ["r1_m0", "r1_m1", "r1_m2", "r2_m1", "r2_m2", "r3_m0", "r3_m1", "r3_m2", "r4_m0", "r4_m1", "r4_m2"]
def lens(path): return [json.loads(l)["length"] for l in open(f"{path}/meta/episodes.jsonl")]
MSU = [bool(json.loads(l)["is_success"]) for l in open(f"{D}/libero10_task0_train_v2_merged/meta/episodes.jsonl")]
eps = []   # dict(src, ep, T, succ, z[G,960], merged_ep)
g0 = 0
srcs = [(t, f"{D}/libero10_task0_train_v2_{t}", "train") for t in ORDER] + \
       [(f"eval_m{i}", f"{D}/libero10_task0_eval_v2_m{i}", "eval") for i in range(3)] + [("sft_target3", f"{D}/libero10_task0_sft_target3", "demo")]
for tag, path, kind in srcs:
    if not os.path.exists(f"{Z}/{tag}.npy"): print("missing", tag); continue
    z = np.load(f"{Z}/{tag}.npy"); m = pd.read_parquet(f"{Z}/{tag}_meta.parquet"); L = lens(path)
    for e in range(len(L)):
        mm = m[m.ep == e]; T = L[e]; G = (T + 5) // 6
        if len(mm) != G or not (mm.t.values == np.arange(G) * 6).all() and False:
            pass
        order = np.argsort(mm.t.values); tt = mm.t.values[order]
        if len(mm) < 4: continue
        eps.append(dict(src=tag, kind=kind, ep=e, T=T, succ=bool(mm.succ.iloc[0]), t=tt, z=z[mm.index.values[order]].astype(np.float32),
                        merged=(g0 + e) if kind == "train" else None))
    if kind == "train": g0 += len(L)
print("episodes", len(eps), {k: sum(e["kind"] == k for e in eps) for k in ("train", "eval", "demo")})
bad = [e for e in eps if len(e["t"]) != (e["T"] + 5) // 6]
print("incomplete grid episodes:", len(bad), "of", len(eps))
eps = [e for e in eps if len(e["t"]) == (e["T"] + 5) // 6]
Lmax = np.percentile([e["T"] for e in eps if e["succ"]], 95); print("Lmax", Lmax)
# split: train-kind + demo episodes -> 80/20 by episode; eval-kind is held-out as is
pool = [i for i, e in enumerate(eps) if e["kind"] in ("train", "demo")]
rng.shuffle(pool); cut = int(0.8 * len(pool)); tr_idx, te_idx = set(pool[:cut]), set(pool[cut:])
tr_eps = [eps[i] for i in tr_idx if eps[i]["succ"]]
allz = np.concatenate([e["z"] for e in tr_eps]); mu, sd = allz.mean(0), allz.std(0) + 1e-6
def norm(z): return (z - mu) / sd
DS = [d for d in range(-5, 6) if d != 0]
def make_pairs(elist, n=None):
    X1, X2, Y = [], [], []
    for e in elist:
        G = len(e["t"]); z = norm(e["z"])
        for i in range(G):
            for d in DS:
                j = i + d
                if 0 <= j < G:
                    X1.append(z[i]); X2.append(z[j]); Y.append(d * 6 * Lmax / e["T"])
    return np.array(X1, np.float32), np.array(X2, np.float32), np.array(Y, np.float32)
X1, X2, Y = make_pairs(tr_eps); print("train pairs", len(Y))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(960 * 3, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 1))
    def forward(s, a, b): return s.f(torch.cat([a, b, b - a], -1)).squeeze(-1)
tX1, tX2, tY = [torch.from_numpy(x).cuda() for x in (X1, X2, Y)]
N = len(tY); EP = 15; K = 5; nets = []
for k in range(K):
    torch.manual_seed(100 + k); net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    steps = EP * (N // 1024); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    for ep in range(EP):
        perm = torch.randperm(N, device="cuda")
        for i in range(N // 1024):
            ix = perm[i * 1024:(i + 1) * 1024]; loss = nn.functional.smooth_l1_loss(net(tX1[ix], tX2[ix]), tY[ix])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); nets.append(net); print("trained member", k, flush=True)
def predict(a1, a2):
    outs = []
    with torch.no_grad():
        for net in nets:
            outs.append(torch.cat([net(torch.from_numpy(a1[i:i+8192]).cuda(), torch.from_numpy(a2[i:i+8192]).cuda()).cpu() for i in range(0, len(a1), 8192)]).numpy())
    return np.stack(outs)   # [K, n]
def report(elist, name):
    a1, a2, y = make_pairs(elist); P = predict(a1, a2)
    ens = {"single (member 0)": P[0], "mean of 5": P.mean(0), "min of 5": P.min(0)}
    singles = np.mean([(np.sign(p) == np.sign(y)).mean() for p in P])
    print(f"[{name}] pairs {len(y)}  single-member sign acc (avg of 5) {singles:.3f}")
    for n, p in ens.items(): print(f"     {n:18s} sign acc {(np.sign(p) == np.sign(y)).mean():.3f}  Spearman {spearmanr(p, y)[0]:.3f}")
    sd = P.std(0); err = np.abs(P.mean(0) - y)
    print(f"     member disagreement (std): mean {sd.mean():.2f};  Spearman(disagreement, |error|) {spearmanr(sd, err)[0]:.3f}")
    return P, y
report([e for e in eps if e["kind"] == "eval" and e["succ"]], "eval success")
report([e for e in eps if e["kind"] == "eval" and not e["succ"]], "eval FAIL")
report([eps[i] for i in te_idx if eps[i]["succ"]], "held-out success")
report([eps[i] for i in te_idx if not eps[i]["succ"]], "held-out FAIL")
# E0b-style: hold vs miss with single / mean / min scoring (forward 30 frames)
rows = []
for i, e in enumerate(eps):
    if not (i in te_idx or e["kind"] == "eval"): continue
    z = norm(e["z"]); G = len(z); ii = np.arange(G); jj = np.minimum(ii + 5, G - 1)
    P = predict(z[ii], z[jj]); m = pd.read_parquet(f"{Z}/{e['src']}_meta.parquet"); mm = m[m.ep == e["ep"]].sort_values("t"); mm = mm[mm.t.isin(e["t"])]
    for k in range(G): rows.append((e["succ"], P[0, k], P[:, k].mean(), P[:, k].min(), float(mm.gap.values[k]), float(mm.cmd.values[k])))
df = pd.DataFrame(rows, columns=["succ", "single", "mean", "min", "gap", "cmd"]); close = df.cmd > 0
miss = close & (df.gap <= 0.03); hold = close & (df.gap >= 0.04) & (df.gap <= 0.071); sub = df[miss | hold]; lab = hold[miss | hold].values
print("hold-vs-miss AUROC (all / fail-episode frames only):")
for c in ("single", "mean", "min"):
    f = sub[~sub.succ]; lf = hold[f.index].values
    print(f"     {c:7s} all {roc_auc_score(lab, sub[c].values):.3f}   fail-only {roc_auc_score(lf, f[c].values):.3f}")
