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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("/root/.fonts/NotoSansCJKjp-Regular.otf", "/root/.fonts/NotoSansCJKjp-Bold.otf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
# ---- memory-light training: all training frames on the GPU once; pairs as index arrays
off = 0; I1, I2, Y = [], [], []; zs = []
for e in tr_eps:
    z = norm(e["z"]).astype(np.float32); G = len(z); zs.append(z)
    for i in range(G):
        for d in DS:
            if 0 <= i + d < G: I1.append(off + i); I2.append(off + i + d); Y.append(d * 6 * Lmax / e["T"])
    off += G
Zg = torch.from_numpy(np.concatenate(zs)).cuda(); del zs
I1 = torch.tensor(I1, device="cuda"); I2 = torch.tensor(I2, device="cuda"); Yt = torch.tensor(Y, dtype=torch.float32, device="cuda"); N = len(Yt); print("train pairs", N, flush=True)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(960 * 3, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 1))
    def forward(s, a, b): return s.f(torch.cat([a, b, b - a], -1)).squeeze(-1)
K, EP = 5, 15; nets = []
for k in range(K):
    torch.manual_seed(100 + k); net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (N // 1024))
    for ep in range(EP):
        perm = torch.randperm(N, device="cuda")
        for i in range(N // 1024):
            ix = perm[i * 1024:(i + 1) * 1024]; loss = nn.functional.smooth_l1_loss(net(Zg[I1[ix]], Zg[I2[ix]]), Yt[ix])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); nets.append(net); print("member", k, flush=True)
del Zg
def fwd(z_a, z_b):
    with torch.no_grad():
        a = torch.from_numpy(z_a).cuda(); b = torch.from_numpy(z_b).cuda()
        return np.stack([n(a, b).cpu().numpy() for n in nets])
# ---- per-frame outputs (forward 30 frames) for held-out train episodes + eval episodes
rows = []; pair_err = []
for i, e in enumerate(eps):
    if not (i in te_idx and e["kind"] == "train") and e["kind"] != "eval": continue
    z = norm(e["z"]).astype(np.float32); G = len(z); ii = np.arange(G); P = fwd(z[ii], z[np.minimum(ii + 5, G - 1)])
    m = pd.read_parquet(f"{Z}/{e['src']}_meta.parquet"); mm = m[m.ep == e["ep"]].sort_values("t"); mm = mm[mm.t.isin(e["t"])]
    for k in range(G): rows.append((e["src"], e["ep"], e["kind"], e["succ"], int(e["t"][k]), *P[:, k], float(mm.gap.values[k]), float(mm.cmd.values[k]), e["T"]))
    if e["kind"] == "train" and not e["succ"] and len(pair_err) < 60000:      # pair errors on failure episodes for the calibration plot
        for d in (-5, -3, 3, 5):
            jj = ii + d; ok = (jj >= 0) & (jj < G); Pp = fwd(z[ii[ok]], z[jj[ok]]); y = d * 6 * Lmax / e["T"]
            pair_err += [(Pp[:, q].std(), abs(Pp[:, q].mean() - y)) for q in range(Pp.shape[1])]
df = pd.DataFrame(rows, columns=["src", "ep", "kind", "succ", "t", *[f"m{k}" for k in range(K)], "gap", "cmd", "T"])
df.to_parquet(f"{R}/tmp/hybrid/e0_ens_frames.parquet"); pe = np.array(pair_err)
Mx = df[[f"m{k}" for k in range(K)]].values; mean_, std_, min_ = Mx.mean(1), Mx.std(1), Mx.min(1)
ho = (df.kind == "train").values; fail = ~df.succ.values; close = df.cmd.values > 0
miss = close & (df.gap.values <= 0.03); hold = close & (df.gap.values >= 0.04) & (df.gap.values <= 0.071)
def auroc(y, v):
    r = rankdata(v); n1 = y.sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum())
cands = {"単体": Mx[:, 0], "平均": mean_, "最小": min_, "平均−1×std": mean_ - std_, "平均−2×std": mean_ - 2 * std_, "平均−4×std": mean_ - 4 * std_}
fp_f, fp_m, au = {}, {}, {}
for n, v in cands.items():
    thr = np.percentile(v[ho & ~fail], 30); pos = v >= thr
    fp_f[n] = pos[ho & fail].mean(); fp_m[n] = pos[ho & miss].mean(); au[n] = auroc(hold[ho & (miss | hold)], v[ho & (miss | hold)])
# ---- figure
fig = plt.figure(figsize=(16, 10.5)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, :2]); ev = df[df.kind == "eval"]
ex = [("成功", ev[ev.succ].groupby(["src", "ep"]).size().index[1], "#1f6feb"), ("失敗", ev[~ev.succ].groupby(["src", "ep"]).size().index[0], "#d1242f")]
for lab, (src, ep), c in ex:
    q = ev[(ev.src == src) & (ev.ep == ep)].sort_values("t"); M5 = q[[f"m{k}" for k in range(K)]].values
    ax.fill_between(q.t, M5.min(1), M5.max(1), color=c, alpha=.18, lw=0); ax.plot(q.t, M5.mean(1), "-", c=c, lw=2, label=f"{lab}: 5 つの平均（帯 = 最小〜最大）")
    ax.plot(q.t, M5.mean(1) - 4 * M5.std(1), ":", c=c, lw=1.5, label=f"{lab}: 平均 − 4×std")
    mq = q[q.cmd > 0]; mq = mq[mq.gap <= 0.03]; ax.scatter(mq.t, M5.mean(1)[q.index.isin(mq.index)], marker="x", c="k", s=36, zorder=5)
ax.scatter([], [], marker="x", c="k", label="空振り"); ax.axhline(0, c="gray", lw=.5); ax.set_xlabel("ステップ"); ax.set_ylabel("スコア"); ax.legend(fontsize=8, ncol=2)
ax.set_title("(a) 5 つのモデルのスコアの時間変化（eval の例）: 失敗のほうが、帯（メンバー間のばらつき）が広い")
ax = fig.add_subplot(gs[0, 2]); bins = np.logspace(-2, 1.5, 40)
for lab, msk, c in (("成功エピソードのフレーム", ho & ~fail, "#1f6feb"), ("失敗エピソードのフレーム", ho & fail, "#d1242f"), ("空振りのフレーム", ho & miss, "k")):
    ax.hist(np.clip(std_[msk], 1e-2, None), bins=bins, density=True, histtype="step" if c == "k" else "stepfilled", alpha=.55 if c != "k" else 1, color=c, label=f"{lab}（n={msk.sum():,}）", lw=2 if c == "k" else 1)
ax.set_xscale("log"); ax.set_xlabel("メンバー間のばらつき（標準偏差）"); ax.set_ylabel("密度"); ax.legend(fontsize=8); ax.set_title("(b) ばらつきの分布（未使用エピソード）\n成功では小さく、失敗・空振りで大きくなる")
ax = fig.add_subplot(gs[1, 0]); q = pd.DataFrame(pe, columns=["sd", "err"]); q["bin"] = pd.qcut(q.sd, 10, duplicates="drop"); g = q.groupby("bin", observed=True).agg(sd=("sd", "mean"), err=("err", "mean"))
ax.plot(g.sd, g.err, "o-", c="#d1242f"); ax.set_xlabel("ばらつき（10 分位の平均）"); ax.set_ylabel("平均の絶対誤差（正解との差）")
ax.set_title(f"(c) ばらつきが大きいほど誤差も大きい（失敗エピソードの対 {len(q):,} 件）\n相関 {spearmanr(q.sd, q.err)[0]:.2f}（Spearman）。ただし、全体の 0.22 程度")
ax = fig.add_subplot(gs[1, 1]); names = list(cands); x = np.arange(len(names)); w = .38
ax.bar(x - w / 2, [fp_f[n] for n in names], w, color="#d1242f", alpha=.8, label="失敗エピソードのフレーム"); ax.bar(x + w / 2, [fp_m[n] for n in names], w, color="k", alpha=.7, label="空振りのフレーム")
for k, n in enumerate(names): ax.text(k - w / 2, fp_f[n], f"{fp_f[n]:.3f}", ha="center", va="bottom", fontsize=8); ax.text(k + w / 2, fp_m[n], f"{fp_m[n]:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("positive になった割合（低いほど良い）"); ax.legend(fontsize=8)
ax.set_title("(d) 偽陽性の割合（成功エピソードの 70% が positive になるしきい値）")
ax = fig.add_subplot(gs[1, 2]); ax.bar(x, [au[n] for n in names], color="#2da44e", alpha=.8)
for k, n in enumerate(names): ax.text(k, au[n], f"{au[n]:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0.75, 0.90); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("AUROC（保持 vs 空振り）"); ax.set_title("(e) 保持と空振りの区別（高いほど良い）")
fig.suptitle("E0e: アンサンブル（5 メンバー）のばらつきを使った判定", fontsize=15); fig.tight_layout(); fig.savefig(f"{R}/tmp/hybrid/e0e_ensemble.png", dpi=120)
print({n: (round(fp_f[n], 3), round(fp_m[n], 3), round(au[n], 3)) for n in names})
