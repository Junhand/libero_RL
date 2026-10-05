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
net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
tX1, tX2, tY = [torch.from_numpy(x).cuda() for x in (X1, X2, Y)]
N = len(tY); EP = 15; steps = EP * (N // 1024); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
for ep in range(EP):
    perm = torch.randperm(N, device="cuda"); tot = 0
    for k in range(N // 1024):
        ix = perm[k * 1024:(k + 1) * 1024]; loss = nn.functional.smooth_l1_loss(net(tX1[ix], tX2[ix]), tY[ix])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    print("epoch", ep, "loss %.4f" % (tot / (N // 1024)), flush=True)
net.eval()
def evaluate(elist, name):
    a1, a2, y = make_pairs(elist)
    with torch.no_grad(): p = torch.cat([net(torch.from_numpy(a1[i:i+8192]).cuda(), torch.from_numpy(a2[i:i+8192]).cuda()).cpu() for i in range(0, len(y), 8192)]).numpy()
    print(f"[{name}] pairs {len(y)}  sign acc {(np.sign(p) == np.sign(y)).mean():.3f}  Spearman(pred, true offset) {spearmanr(p, y)[0]:.3f}")
evaluate([eps[i] for i in te_idx if eps[i]["succ"]], "held-out success episodes (train/demo split)")
evaluate([e for e in eps if e["kind"] == "eval" and e["succ"]], "eval rollouts, success")
evaluate([e for e in eps if e["kind"] == "eval" and not e["succ"]], "eval rollouts, FAIL")
# per-frame score A_hat(t) = predicted offset for (t, t+5 grid = 30 frames), compare with STEAM's A_STEAM
def score(e):
    z = norm(e["z"]); G = len(z); i = np.arange(G); j = np.minimum(i + 5, G - 1)
    with torch.no_grad(): return net(torch.from_numpy(z[i]).cuda(), torch.from_numpy(z[j]).cuda()).cpu().numpy()
print("STEAM table = merged episodes 0..1457 (lengths T-8), verified")
st = pd.read_parquet(f"{R}/tmp/steam/step4/adv_train_merged_n32e2000_endfixD_q30.parquet")
st_by = {k: g.sort_values("frame_index") for k, g in st.groupby("episode_index")}
rhos = {True: [], False: []}; allr = []
for i in te_idx:
    e = eps[i]
    if e["kind"] != "train" or e["merged"] not in st_by: continue
    g = st_by[e["merged"]]; assert len(g) == e["T"] - 8
    adv = g.advantage_continuous.values
    keep = e["t"] < len(adv); s_ = score(e)[keep]; a_ = adv[e["t"][keep]]
    if np.std(a_) > 0 and np.std(s_) > 0: rhos[e["succ"]].append(spearmanr(s_, a_)[0])
    allr.append((s_, a_))
for k, name in ((True, "success"), (False, "FAIL")):
    print(f"within-episode Spearman(A_hat, A_STEAM), held-out {name} episodes: mean {np.mean(rhos[k]):.3f} median {np.median(rhos[k]):.3f} n={len(rhos[k])}")
S = np.concatenate([a for a, _ in allr]); A = np.concatenate([b for _, b in allr]); print("pooled Spearman(A_hat, A_STEAM), all held-out frames with STEAM labels", round(spearmanr(S, A)[0], 3), "n_frames", len(S))
# success vs failure discrimination at episode level
def ep_stats(elist): return np.array([[score(e).mean(), np.percentile(score(e), 10)] for e in elist])
for name, el in (("eval rollouts", [e for e in eps if e["kind"] == "eval"]), ("held-out train rollouts", [eps[i] for i in te_idx if eps[i]["kind"] == "train"])):
    ys = np.array([e["succ"] for e in el])
    if len(set(ys)) < 2: print(name, "single class"); continue
    s = ep_stats(el); print(f"[{name}] episodes {len(el)} (success {ys.sum()})  AUROC mean-score {roc_auc_score(ys, s[:,0]):.3f}  AUROC p10-score {roc_auc_score(ys, s[:,1]):.3f}")
# agreement of the positive labels (top 30% of each score over the same held-out frames)
pa = S >= np.percentile(S, 70); pb = A >= np.percentile(A, 70)
po = (pa == pb).mean(); pe = pa.mean() * pb.mean() + (1 - pa.mean()) * (1 - pb.mean())
print(f"top-30% label agreement with STEAM: accuracy {po:.3f}  Cohen kappa {(po - pe) / (1 - pe):.3f}  IoU {(pa & pb).sum() / (pa | pb).sum():.3f}")
torch.save(dict(net=net.state_dict(), mu=mu, sd=sd, Lmax=Lmax), f"{R}/tmp/hybrid/offset_mlp.pt")
# ---- E0b: does the score drop on missed grasps? (held-out episodes + eval episodes), frames on the t%6==0 grid
metas = {}
for tag in {e["src"] for e in eps}:
    metas[tag] = pd.read_parquet(f"{Z}/{tag}_meta.parquet")
rows = []
for i, e in enumerate(eps):
    if not (i in te_idx or e["kind"] == "eval"): continue
    m = metas[e["src"]]; mm = m[m.ep == e["ep"]].sort_values("t"); mm = mm[mm.t.isin(e["t"])]
    sc = score(e); G = len(sc)
    for k in range(G):
        rows.append((e["src"], e["ep"], e["succ"], e["t"][k], sc[k], float(mm.gap.values[k]), float(mm.cmd.values[k]), e["T"]))
df = pd.DataFrame(rows, columns=["src", "ep", "succ", "t", "score", "gap", "cmd", "T"])
df["close"] = df.cmd > 0
df["miss"] = df.close & (df.gap <= 0.03)
df["hold"] = df.close & (df.gap >= 0.04) & (df.gap <= 0.071)
df.to_parquet(f"{R}/tmp/hybrid/e0b_frames.parquet")
print("frames", len(df), "closing", int(df.close.sum()), "miss", int(df.miss.sum()), "hold", int(df.hold.sum()))
print("mean score: all %.2f | closing-miss %.2f | closing-hold %.2f | open %.2f" % (df.score.mean(), df[df.miss].score.mean(), df[df.hold].score.mean(), df[~df.close].score.mean()))
sub = df[df.miss | df.hold]
if sub.miss.sum() > 5 and sub.hold.sum() > 5:
    print("AUROC(hold vs miss) by score, all frames:", round(roc_auc_score(sub.hold.values, sub.score.values), 3))
    # control: only failing episodes' frames vs only successful
    for nm, q in (("success eps", sub[sub.succ]), ("fail eps", sub[~sub.succ])):
        if q.miss.sum() > 5 and q.hold.sum() > 5: print(f"  {nm}: miss n={int(q.miss.sum())} hold n={int(q.hold.sum())}  AUROC {roc_auc_score(q.hold.values, q.score.values):.3f}  mean miss {q[q.miss].score.mean():.2f} hold {q[q.hold].score.mean():.2f}")
