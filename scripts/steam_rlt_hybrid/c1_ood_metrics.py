"""C1: which score best detects "outside the success-episode distribution" from cached RL tokens (z_rl, 960-d)?
Data prep (episode arrays, pair index arrays, MLP) is derived from scripts/steam_rlt_hybrid/e0_ens_viz.py.
Everything is fit on SUCCESSFUL episodes of FIT_SRC only; evaluation is on TEST_SRC (split by data source, not by episode).
Scores (higher = more out-of-distribution):
  (a) ensemble std of a temporal-offset MLP (M=5, M=10)      (b) k-NN distance to the fit frames (k=1,5,20; Euclid, cosine)
  (c) Mahalanobis distance (Ledoit-Wolf shrinkage; PCA 64 / 128)    (d) mean rank of (a) and (b)
Outputs (all in tmp/hybrid_env1/): c1_result.txt, c1_ood.png, c1_scores.parquet. Input tmp/hybrid/zrl is read only."""
import json, os, sys, time, resource
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import rankdata
T0 = time.time()
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"; OUT = f"{R}/tmp/hybrid_env1"
FIT_SRC = ["r1_m0", "r1_m1", "r1_m2", "r2_m1", "r2_m2", "r3_m0", "r3_m1", "r3_m2"]
TEST_SRC = ["r4_m0", "r4_m1", "r4_m2", "eval_m0", "eval_m1", "eval_m2"]
BINS = [(0, 60), (60, 120), (120, 180), (180, 10 ** 9)]; BIN_NAMES = ["t0-60", "t60-120", "t120-180", "t180+"]
DS = [d for d in range(-5, 6) if d != 0]          # grid offsets of the MLP: +-6..30 frames
M_MAX, EPOCHS, BS = 10, 15, 1024
def log(*a): print(f"[{time.time() - T0:6.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB]", *a, flush=True)
def auroc(y, s):
    y = np.asarray(y).astype(bool); n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 < 30 or n0 < 30: return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
torch.manual_seed(0)
# ------------------------------------------------------------------ data
def src_path(tag): return f"{D}/libero10_task0_eval_v2_m{tag[-1]}" if tag.startswith("eval") else f"{D}/libero10_task0_train_v2_{tag}"
def lens(path): return [json.loads(l)["length"] for l in open(f"{path}/meta/episodes.jsonl")]
def load(tags, success_only):
    eps, dropped = [], 0
    for tag in tags:
        z = np.load(f"{Z}/{tag}.npy"); m = pd.read_parquet(f"{Z}/{tag}_meta.parquet"); L = lens(src_path(tag)); assert len(m) == len(z)
        for e, g in m.groupby("ep"):
            T = L[int(e)]; G = (T + 5) // 6; g = g.sort_values("t")
            if len(g) != G or not (g.t.values == np.arange(G) * 6).all(): dropped += 1; continue
            succ = bool(g.succ.any())                      # succ from the meta (an episode succeeds if any frame does); NOT episodes.jsonl is_success
            if success_only and not succ: continue
            eps.append(dict(src=tag, ep=int(e), T=T, G=G, succ=succ, t=g.t.values, z=z[g.index.values].astype(np.float32),
                            gap=g.gap.values.astype(np.float32), cmd=g.cmd.values.astype(np.float32)))
    return eps, dropped
fit_eps, d1 = load(FIT_SRC, True); test_eps, d2 = load(TEST_SRC, False)
log("fit success episodes", len(fit_eps), "frames", sum(e["G"] for e in fit_eps), "| test episodes", len(test_eps),
    "(success", sum(e["succ"] for e in test_eps), "fail", sum(not e["succ"] for e in test_eps), ") | dropped incomplete", d1, d2)
Lmax = float(np.percentile([e["T"] for e in fit_eps], 95)); log("Lmax (95th pct of fit success episode length)", Lmax)
allz = np.concatenate([e["z"] for e in fit_eps]); mu, sd = allz.mean(0), allz.std(0) + 1e-6; del allz
Zf = torch.from_numpy(np.concatenate([(e["z"] - mu) / sd for e in fit_eps]).astype(np.float32)).cuda()      # fit frames, standardized
Zt = torch.from_numpy(np.concatenate([(e["z"] - mu) / sd for e in test_eps]).astype(np.float32)).cuda()     # test frames, standardized
Nf, Nt = len(Zf), len(Zt)
meta = pd.DataFrame({"src": np.concatenate([[e["src"]] * e["G"] for e in test_eps]), "ep": np.concatenate([[e["ep"]] * e["G"] for e in test_eps]),
                     "succ": np.concatenate([[e["succ"]] * e["G"] for e in test_eps]), "t": np.concatenate([e["t"] for e in test_eps]),
                     "gap": np.concatenate([e["gap"] for e in test_eps]), "cmd": np.concatenate([e["cmd"] for e in test_eps])})
off = 0; U = []                                            # partner index (30 frames ahead = +5 grid steps, clamped at the episode end)
for e in test_eps:
    i = np.arange(e["G"]); U.append(off + np.minimum(i + 5, e["G"] - 1)); off += e["G"]
U = torch.from_numpy(np.concatenate(U)).cuda()
# ------------------------------------------------------------------ (a) ensemble of offset MLPs (pairs as index arrays)
off = 0; I1, I2, Y = [], [], []
for e in fit_eps:
    G = e["G"]; i = np.arange(G)
    for d in DS:
        j = i + d; ok = (j >= 0) & (j < G); I1.append(off + i[ok]); I2.append(off + j[ok]); Y.append(np.full(int(ok.sum()), d * 6 * Lmax / e["T"], np.float32))
    off += G
I1 = torch.from_numpy(np.concatenate(I1)).cuda(); I2 = torch.from_numpy(np.concatenate(I2)).cuda(); Yt = torch.from_numpy(np.concatenate(Y)).cuda(); N = len(Yt)
log("train pairs", N, "(index arrays; the pair tensors are never materialised)")
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(960 * 3, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 1))
    def forward(s, a, b): return s.f(torch.cat([a, b, b - a], -1)).squeeze(-1)
nets = []
for k in range(M_MAX):
    torch.manual_seed(100 + k); net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * (N // BS))
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device="cuda")
        for i in range(N // BS):
            ix = perm[i * BS:(i + 1) * BS]; loss = nn.functional.smooth_l1_loss(net(Zf[I1[ix]], Zf[I2[ix]]), Yt[ix])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); nets.append(net); log("member", k, "final train loss", round(float(loss), 4))
del I1, I2, Yt
with torch.no_grad():
    P = np.stack([np.concatenate([n(Zt[a:a + 8192], Zt[U[a:a + 8192]]).cpu().numpy() for a in range(0, Nt, 8192)]) for n in nets])    # [M, Nt]
scores = {"ens_std_M5": P[:5].std(0), "ens_std_M10": P.std(0)}
# ------------------------------------------------------------------ (b) k-NN distance to the fit frames
@torch.no_grad()
def knn_kth(metric, ks=(1, 5, 20), chunk=2048):
    A = Zf if metric == "euc" else nn.functional.normalize(Zf, dim=1); a2 = (Zf ** 2).sum(1); out = {k: [] for k in ks}
    for s in range(0, Nt, chunk):
        q = Zt[s:s + chunk]
        if metric == "euc": d = ((q ** 2).sum(1, keepdim=True) + a2[None] - 2 * q @ A.T).clamp_min(0).sqrt(); v = d.topk(max(ks), dim=1, largest=False).values
        else: v = 1 - nn.functional.normalize(q, dim=1) @ A.T; v = v.topk(max(ks), dim=1, largest=False).values
        for k in ks: out[k].append(v[:, k - 1].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
for metric in ("euc", "cos"):
    for k, v in knn_kth(metric).items(): scores[f"knn{k}_{metric}"] = v
log("kNN done")
# ------------------------------------------------------------------ (c) Mahalanobis (Ledoit-Wolf shrinkage in PCA space)
@torch.no_grad()
def maha(p):
    Xd = Zf.double(); m0 = Xd.mean(0); Xc = Xd - m0; C = Xc.T @ Xc / Nf
    ev, V = torch.linalg.eigh(C); V = V[:, -p:]; X = Xc @ V; n = len(X)
    S = X.T @ X / n; mu_ = torch.trace(S) / p; I = torch.eye(p, device=X.device, dtype=X.dtype)
    delta2 = ((S - mu_ * I) ** 2).sum() / p
    b2 = ((X ** 2).sum(1) ** 2 - 2 * ((X @ S) * X).sum(1) + (S ** 2).sum()).sum() / (n * n) / p
    alpha = float(min(b2, delta2) / delta2); Sig = alpha * mu_ * I + (1 - alpha) * S
    Y = (Zt.double() - m0) @ V; d2 = (Y @ torch.linalg.inv(Sig) * Y).sum(1)
    return d2.clamp_min(0).sqrt().cpu().numpy(), alpha
for p in (64, 128):
    s, al = maha(p); scores[f"maha_pca{p}"] = s; log(f"mahalanobis pca{p}: shrinkage alpha", round(al, 4))
# ------------------------------------------------------------------ (d) rank averages and a time reference
def rk(v): return (rankdata(v) - 1) / (len(v) - 1)
KNN = [f"knn{k}_{m}" for m in ("euc", "cos") for k in (1, 5, 20)]
for a in ("ens_std_M5", "ens_std_M10"):
    for b in KNN: scores[f"comb[{a}+{b}]"] = (rk(scores[a]) + rk(scores[b])) / 2
scores["ref_t (time only)"] = meta.t.values.astype(float)
# ------------------------------------------------------------------ evaluation
succ = meta.succ.values; t = meta.t.values; close = meta.cmd.values > 0
miss = close & (meta.gap.values <= 0.03); hold = close & (meta.gap.values >= 0.04) & (meta.gap.values <= 0.071)
tasks = {   # name: (positive mask = out-of-distribution side, negative mask = in-distribution side)
    "OOD_vs_ID": (~succ, succ),
    "miss_vs_hold": (miss & ~succ, hold & succ),            # miss: failure-episode frame (OOD side); hold: success-episode frame (ID side)
    "miss_vs_hold_alleps": (miss, hold)}                    # same, ignoring episode success (as in E0b)
def evaluate(s):
    out = {}
    for tn, (pos, neg) in tasks.items():
        m = pos | neg; y = pos[m]; v = s[m]; tt = t[m]
        row = [auroc(y, v)] + [auroc(y[(tt >= lo) & (tt < hi)], v[(tt >= lo) & (tt < hi)]) for lo, hi in BINS]
        out[tn] = row + [np.nanmin(row[1:]) if not np.all(np.isnan(row[1:])) else np.nan]
    return out
res = {n: evaluate(s) for n, s in scores.items()}
cnt = {tn: ([int((pos & (t >= lo) & (t < hi)).sum()) for lo, hi in BINS], [int((neg & (t >= lo) & (t < hi)).sum()) for lo, hi in BINS], int(pos.sum()), int(neg.sum())) for tn, (pos, neg) in tasks.items()}
# ------------------------------------------------------------------ result text
f = lambda x: "  nan " if np.isnan(x) else f"{x:6.3f}"
lines = [f"C1 result (elapsed {time.time() - T0:.0f}s, peak host RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB)", "",
         f"fit  : success episodes of {FIT_SRC} -> {len(fit_eps)} episodes, {Nf:,} frames (t % 6 == 0). Lmax={Lmax:.1f}",
         f"test : {TEST_SRC} -> {len(test_eps)} episodes, {Nt:,} frames; success {int(succ.sum()):,} frames / failure {int((~succ).sum()):,} frames",
         "score: higher = more out-of-distribution. AUROC positives = OOD side. 0.5 = chance. 'min' = minimum over the four t bins.",
         "MLP : [z_t, z_u, z_u-z_t] -> 2x512, Huber, 15 epochs, 30-frame lookahead (clamped at the end), seeds 100.., M=5 uses members 0-4, M=10 all. k-NN: distance to the k-th neighbour. Maha: Ledoit-Wolf in PCA space.", ""]
for tn, (cp, cn, np_, nn_) in cnt.items():
    lines.append(f"{tn}: positives {np_:,} / negatives {nn_:,}   per t-bin positives {cp} negatives {cn}   (bins with <30 of either -> nan)")
lines += ["", "tasks: OOD_vs_ID = failure-episode frames (positive) vs success-episode frames. miss_vs_hold = missed-grasp frames of FAILURE episodes vs held-grasp frames of SUCCESS episodes (spec; mixes in the success/failure difference).",
          "       miss_vs_hold_alleps = missed vs held grasp frames regardless of the episode outcome (as E0b; isolates the grasp-phase signal). miss = cmd>0 & gap<=0.03; hold = cmd>0 & 0.04<=gap<=0.071. No miss frames exist at t<60 -> that bin is nan.", ""]
hdr = "metric".ljust(34) + "".join(f"| {tn[:20]:<20}".ljust(0) for tn in tasks)
for tn in tasks:
    lines += [f"=== AUROC: {tn}", "metric".ljust(34) + "   all " + "".join(f"{b:>8}" for b in BIN_NAMES) + "    min"]
    for n in res: lines.append(n.ljust(34) + "".join(f(x) + " " for x in res[n][tn][:1]) + " " + "".join(f(x) + "  " for x in res[n][tn][1:5]) + " " + f(res[n][tn][5]))
    lines.append("")
single = [n for n in res if not n.startswith("comb") and not n.startswith("ref")]
def top(tn, col, names, k=3): return sorted(names, key=lambda n: -np.nan_to_num(res[n][tn][col], nan=-1))[:k]
lines += ["=== summary (auto-generated)",
          "OOD_vs_ID overall, top singles : " + ", ".join(f"{n} {res[n]['OOD_vs_ID'][0]:.3f}" for n in top("OOD_vs_ID", 0, single)),
          "OOD_vs_ID worst-bin, top singles: " + ", ".join(f"{n} {res[n]['OOD_vs_ID'][5]:.3f}" for n in top("OOD_vs_ID", 5, single)),
          "miss_vs_hold overall, top singles: " + ", ".join(f"{n} {res[n]['miss_vs_hold'][0]:.3f}" for n in top("miss_vs_hold", 0, single)),
          "miss_vs_hold worst-bin, top singles: " + ", ".join(f"{n} {res[n]['miss_vs_hold'][5]:.3f}" for n in top("miss_vs_hold", 5, single)),
          "miss_vs_hold_alleps overall, top singles: " + ", ".join(f"{n} {res[n]['miss_vs_hold_alleps'][0]:.3f}" for n in top("miss_vs_hold_alleps", 0, single)),
          "miss_vs_hold_alleps worst-bin, top singles: " + ", ".join(f"{n} {res[n]['miss_vs_hold_alleps'][5]:.3f}" for n in top("miss_vs_hold_alleps", 5, single)),
          "best combination (OOD_vs_ID overall): " + ", ".join(f"{n} {res[n]['OOD_vs_ID'][0]:.3f}" for n in top("OOD_vs_ID", 0, [n for n in res if n.startswith('comb')], 2))]
open(f"{OUT}/c1_result.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
pd.concat([meta, pd.DataFrame(scores)], axis=1).to_parquet(f"{OUT}/c1_scores.parquet")
# ------------------------------------------------------------------ figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib import font_manager as fm
FONT = "/root/.fonts/NotoSansCJKjp-Regular.otf"; JP = os.path.exists(FONT)
if JP: fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
def L(jp, en): return jp if JP else en
def roc(y, s, npts=500):
    o = np.argsort(-s, kind="stable"); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(~yy); ix = np.unique(np.linspace(0, len(s) - 1, npts).astype(int))
    return np.r_[0, fp[ix] / fp[-1]], np.r_[0, tp[ix] / tp[-1]]
show = {"ens_std_M5": ("#1f6feb", "-"), "ens_std_M10": ("#0a3069", "-"), "knn1_euc": ("#2da44e", "-"), "knn5_euc": ("#1a7f37", "-"), "knn20_euc": ("#116329", "-"),
        "knn1_cos": ("#2da44e", "--"), "knn5_cos": ("#1a7f37", "--"), "knn20_cos": ("#116329", "--"), "maha_pca64": ("#fb8500", "-"), "maha_pca128": ("#bc4c00", "-"),
        "comb[ens_std_M10+knn5_euc]": ("#d1242f", "-"), "comb[ens_std_M10+knn5_cos]": ("#d1242f", "--"), "ref_t (time only)": ("gray", ":")}
fig = plt.figure(figsize=(19, 17)); gs = fig.add_gridspec(3, 6, hspace=.42, wspace=1.3)
for c, (tn, ttl) in enumerate([("OOD_vs_ID", L("(1) ROC: 失敗エピソード（OOD）vs 成功エピソード（ID）のフレーム", "(1) ROC: failure-episode frames (OOD) vs success-episode frames (ID)")),
                               ("miss_vs_hold", L("(2) ROC: 空振り（失敗側）vs 保持（成功側）", "(2) ROC: missed grasp (failure side) vs held grasp (success side)"))]):
    ax = fig.add_subplot(gs[0, c * 3:(c + 1) * 3]); pos, neg = tasks[tn]; m = pos | neg
    for n, (col, ls) in show.items():
        fp, tp = roc(pos[m], scores[n][m]); ax.plot(fp, tp, color=col, ls=ls, lw=1.6, label=f"{n} ({res[n][tn][0]:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=.6); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(ttl, fontsize=11); ax.legend(fontsize=7, loc="lower right")
names = list(show); disp = lambda n: n.replace("comb[ens_std_M10+", "comb:ens10+").replace("]", "").replace("ens_std_", "ens_").replace(" (time only)", "")
for c, (tn, ttl) in enumerate([("OOD_vs_ID", L("(3) AUROC: 失敗 vs 成功、t の区間ごと", "(3) AUROC failure vs success, by t bin")),
                               ("miss_vs_hold", L("(4) AUROC: 空振り vs 保持、t の区間ごと", "(4) AUROC missed vs held grasp, by t bin"))]):
    ax = fig.add_subplot(gs[1, c * 3:(c + 1) * 3]); A = np.array([res[n][tn][:5] for n in names], float)
    im = ax.imshow(A, vmin=.35, vmax=1, cmap="RdYlGn", aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax.text(j, i, "nan" if np.isnan(A[i, j]) else f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels(["all"] + BIN_NAMES, fontsize=8); ax.set_yticks(range(len(names))); ax.set_yticklabels([disp(n) for n in names], fontsize=7); ax.set_title(ttl, fontsize=11)
pos, neg = tasks["OOD_vs_ID"]
for c, n in enumerate(["ens_std_M10", "knn5_euc", "maha_pca64"]):
    ax = fig.add_subplot(gs[2, c * 2:(c + 1) * 2]); v = scores[n]; v = np.clip(v, 1e-6, None); bins = np.logspace(np.log10(np.percentile(v, .5)), np.log10(np.percentile(v, 99.9)), 50)
    for lab, mk, col in ((L("成功（ID）", "success (ID)"), succ, "#1f6feb"), (L("失敗（OOD）", "failure (OOD)"), ~succ, "#d1242f"), (L("保持", "held grasp"), hold & succ, "#2da44e"), (L("空振り", "missed grasp"), miss & ~succ, "k")):
        ax.hist(v[mk], bins=bins, density=True, histtype="step", color=col, lw=1.8, label=f"{lab} (n={int(mk.sum()):,})")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel(n); ax.set_ylabel(L("密度", "density")); ax.legend(fontsize=7); ax.set_title(f"({5 + c}) {L('スコアの分布', 'score distribution')}: {n}", fontsize=10)
fig.suptitle(L("C1: RL token の特徴量で「成功エピソードの分布の外」を見つける指標の比較", "C1: scores for detecting 'outside the success-episode distribution' from RL-token features")
             + (L("", "  (Japanese font /root/.fonts/NotoSansCJKjp-Regular.otf not present in this environment -> English labels)") if not JP else ""), fontsize=13)
fig.savefig(f"{OUT}/c1_ood.png", dpi=110, bbox_inches="tight"); log("DONE; figure saved")
