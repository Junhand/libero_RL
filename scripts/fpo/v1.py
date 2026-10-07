"""F1(2): offline value-head probe on RL tokens (z_rl 960 + state 8) for FPO's GAE. Read-only inputs; outputs in this dir."""
import json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr, rankdata
ROOT = "/workspace/libero_RL"; G = 0.995
SRCS = [f"r{a}_m{b}" for a, b in [(1,0),(1,1),(1,2),(2,1),(2,2),(3,0),(3,1),(3,2),(4,0),(4,1),(4,2)]] + ["eval_m0","eval_m1","eval_m2"]
dev = "cuda"
ids = pd.read_parquet(f"{ROOT}/results/steam_rlt_hybrid/initstate_ids.parquet")
Z, S, T_, EP, V, SUC, CID, TT, SID = [], [], [], [], [], [], [], [], []
epid = 0
for si, src in enumerate(SRCS):
    z = np.load(f"{ROOT}/tmp/hybrid/zrl/{src}.npy"); m = pd.read_parquet(f"{ROOT}/tmp/hybrid/zrl/{src}_meta.parquet")
    d = f"{ROOT}/data/smolvla_recap/libero10_task0_{'eval' if src.startswith('eval') else 'train'}_v2{'' if src.startswith('eval') else '_'+src}"
    if src.startswith("eval"): d = f"{ROOT}/data/smolvla_recap/libero10_task0_eval_v2_{src[5:]}"
    L = {}
    for l in open(d + "/meta/episodes.jsonl"):
        j = json.loads(l); L[j["episode_index"]] = j["length"]
    sub = ids[ids.src == src].set_index("ep").cid
    ok = m.ep.isin(sub.index) & m.ep.isin(list(L))
    m = m[ok]; z = z[ok.values]
    cid = m.ep.map(sub).values; T = m.ep.map(L).values.astype(np.float64); t = m.t.values.astype(np.float64)
    v = np.where(m.succ.values, G ** np.maximum(T - 1 - t, 0), 0.0)
    Z.append(z); S.append(m[[f"s{i}" for i in range(8)]].values.astype(np.float32)); V.append(v); SUC.append(m.succ.values)
    CID.append(cid); TT.append(t); T_.append(T); EP.append(m.ep.values + 100000 * si); SID.append(np.full(len(m), si))
Z = np.concatenate(Z); S = np.concatenate(S); V = np.concatenate(V); SUC = np.concatenate(SUC); CID = np.concatenate(CID)
TT = np.concatenate(TT); T_ = np.concatenate(T_); EP = np.concatenate(EP)
print("frames", len(V), "episodes", len(np.unique(EP)), "succ-frame frac", SUC.mean(), flush=True)
zt = torch.tensor(Z, dtype=torch.float16, device=dev); st = torch.tensor(S, device=dev)
vt = torch.tensor(V, dtype=torch.float32, device=dev)

def fit(tr, use_state, seed=0):
    torch.manual_seed(seed)
    zi = zt[tr].float(); mu, sd = zi.mean(0), zi.std(0) + 1e-6
    sm, ss = st[tr].mean(0), st[tr].std(0) + 1e-6
    din = 960 + (8 if use_state else 0)
    net = nn.Sequential(nn.Linear(din, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    def feat(ix):
        x = (zt[ix].float() - mu) / sd
        return torch.cat([x, (st[ix] - sm) / ss], 1) if use_state else x
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4); tri = torch.tensor(tr, device=dev)
    for ep in range(30):
        p = tri[torch.randperm(len(tri), device=dev)]
        for i in range(0, len(p), 1024):
            b = p[i:i + 1024]; loss = ((net(feat(b)).squeeze(-1) - vt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    @torch.no_grad()
    def pred(ix):
        return torch.cat([net(feat(torch.tensor(ix[i:i+8192], device=dev))).squeeze(-1) for i in range(0, len(ix), 8192)]).cpu().numpy()
    return pred

def auroc(s, y):
    if y.all() or (~y).all(): return np.nan
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def metrics(te, pred):
    v, p = V[te], pred
    out = {"EV": 1 - np.var(v - p) / np.var(v), "spearman": spearmanr(v, p)[0]}
    for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 1e9)]:
        k = (TT[te] >= lo) & (TT[te] < hi); out[f"AUROC_t{lo}-{hi if hi < 1e9 else ''}"] = auroc(p[k], SUC[te][k]) if k.sum() > 20 else np.nan
    return out

def td(te, pred):
    """delta = G^12 V(t+12) - V(t) within an episode on the 6-step grid (2 grid points ahead); last interval adds the terminal reward."""
    idx = np.array(te); d = {}
    df = pd.DataFrame({"i": np.arange(len(idx)), "ep": EP[idx], "t": TT[idx]}).sort_values(["ep", "t"])
    dl, dv = [], []
    for ep, g in df.groupby("ep"):
        i = g.i.values; t = g.t.values; T = T_[idx[i[0]]]; ok = SUC[idx[i[0]]]
        tm = {int(a): b for a, b in zip(t, i)}
        for a, b in zip(t, i):
            c = tm.get(int(a) + 12)
            if c is not None:
                dl.append(G ** 12 * pred[c] - pred[b]); dv.append(G ** 12 * V[idx[c]] - V[idx[b]])
            elif a + 12 >= T - 1 + 1e-9 and a + 6 < T:  # episode ends inside the interval
                r = G ** max(T - 1 - a, 0) if ok else 0.0
                dl.append(r - pred[b]); dv.append(r - V[idx[b]])
    dl, dv = np.array(dl), np.array(dv)
    return {"n": len(dl), "std_delta_pred": dl.std(), "std_delta_true": dv.std(), "std_Vdiff_true": np.std([0])}

def calib(te, pred):
    q = np.quantile(pred, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, 9)
    return [(pred[b == k].mean(), V[te][b == k].mean()) for k in range(10)]

rng = np.random.RandomState(0); u = np.arange(50); rng.shuffle(u); folds = np.array_split(u, 5)
all_idx = np.arange(len(V)); eps = np.unique(EP); rng.shuffle(eps); held = set(eps[: int(0.2 * len(eps))])
res = {}; lines = []
for use_state in (True, False):
    nm = "state" if use_state else "nostate"
    # (a) unseen initial states, 5 folds
    pooled_p = np.zeros(len(V)); ms = []
    for f in folds:
        te = all_idx[np.isin(CID, f)]; tr = all_idx[~np.isin(CID, f)]
        pr = fit(tr, use_state)(te); pooled_p[te] = pr; ms.append(metrics(te, pr))
    m = {k: np.nanmean([x[k] for x in ms]) for k in ms[0]}; m["EV_min_fold"] = min(x["EV"] for x in ms)
    res[f"unseen_{nm}"] = (m, td(all_idx, pooled_p), calib(all_idx, pooled_p))
    # (b) control: random episodes
    ish = np.isin(EP, list(held)); te = all_idx[ish]; tr = all_idx[~ish]
    pr = fit(tr, use_state)(te); res[f"control_{nm}"] = (metrics(te, pr), td(te, pr), calib(te, pr))
    print(nm, "done", flush=True)
with open("v1_result.txt", "w") as f:
    f.write(f"frames {len(V)}  episodes {len(np.unique(EP))}  V* std {V.std():.3f}  V* mean {V.mean():.3f}\n")
    for k, (m, t, c) in res.items():
        f.write(f"\n== {k} ==\n" + "\n".join(f"{a}: {b:.3f}" for a, b in m.items()) + "\n")
        f.write(f"TD n={t['n']} std(delta_pred)={t['std_delta_pred']:.4f} std(delta with true V)={t['std_delta_true']:.4f}\n")
        f.write("calib (mean V_pred, mean V*) by decile: " + " ".join(f"({a:.2f},{b:.2f})" for a, b in c) + "\n")
print(open("v1_result.txt").read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
try: fm.fontManager.addfont("/root/.fonts/NotoSansCJKjp-Regular.otf"); plt.rcParams["font.family"] = "Noto Sans CJK JP"; jp = True
except Exception: jp = False
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, k in zip(ax, res):
    c = np.array(res[k][2]); a.plot(c[:, 0], c[:, 1], "o-"); a.plot([0, 1], [0, 1], "k--"); a.set_title(k)
    a.set_xlabel("予測 V の平均" if jp else "mean predicted V"); a.set_ylabel("V* の平均" if jp else "mean V*")
plt.tight_layout(); plt.savefig("v1_calibration.png", dpi=110)
