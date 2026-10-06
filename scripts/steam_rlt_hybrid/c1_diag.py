"""C1 diagnostics (not part of the requested table): (1) why does Mahalanobis(PCA64) give AUROC < 0.5?  (2) is the ensemble-std result affected by the
end-of-episode clamping of the 30-frame partner?   Reads tmp/hybrid/zrl (read only) and c1_scores.parquet."""
import json, os, time, numpy as np, pandas as pd, torch
from scipy.stats import rankdata
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"; OUT = f"{R}/tmp/hybrid_env1"
FIT = ["r1_m0", "r1_m1", "r1_m2", "r2_m1", "r2_m2", "r3_m0", "r3_m1", "r3_m2"]; TEST = ["r4_m0", "r4_m1", "r4_m2", "eval_m0", "eval_m1", "eval_m2"]
def auroc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum())
def sp(tag): return f"{D}/libero10_task0_eval_v2_m{tag[-1]}" if tag.startswith("eval") else f"{D}/libero10_task0_train_v2_{tag}"
def load(tags, success_only):
    Zs, S = [], []
    for tag in tags:
        z = np.load(f"{Z}/{tag}.npy"); m = pd.read_parquet(f"{Z}/{tag}_meta.parquet"); L = [json.loads(l)["length"] for l in open(f"{sp(tag)}/meta/episodes.jsonl")]
        for e, g in m.groupby("ep"):
            g = g.sort_values("t"); suc = bool(g.succ.any())
            if (success_only and not suc) or len(g) != (L[int(e)] + 5) // 6: continue
            Zs.append(z[g.index.values].astype(np.float32)); S += [suc] * len(g)
    return np.concatenate(Zs), np.array(S)
Zf, _ = load(FIT, True); Zt, succ = load(TEST, False); mu, sd = Zf.mean(0), Zf.std(0) + 1e-6
Zf = torch.from_numpy((Zf - mu) / sd).cuda().double(); Zt = torch.from_numpy((Zt - mu) / sd).cuda().double(); Nf = len(Zf)
m0 = Zf.mean(0); Xc = Zf - m0; ev, V = torch.linalg.eigh(Xc.T @ Xc / Nf); ev = ev.flip(0); V = V.flip(1); tot = ev.sum()
lines = ["C1 diagnostics", "", "== (1) Mahalanobis in the top-p PCA space (fit = success frames of r1-r3; test = r4 + eval)",
         "p     var_expl  mean d^2 on FIT frames (in-sample; ~p expected)  ID(test success) mean d^2   OOD(test failure) mean d^2   AUROC(maha)  AUROC(PCA residual norm)"]
for p in (16, 64, 128, 256, 512):
    Vp = V[:, :p]; X = Xc @ Vp; S = X.T @ X / Nf
    Sig = S  # shrinkage alpha is ~0.0005 for p<=128 (see c1.py); plain covariance is used here
    Y = (Zt - m0) @ Vp; d2 = (Y @ torch.linalg.inv(Sig) * Y).sum(1); d2f = (X @ torch.linalg.inv(Sig) * X).sum(1)
    res = ((Zt - m0) ** 2).sum(1) - (Y ** 2).sum(1); d = d2.sqrt().cpu().numpy(); r = res.clamp_min(0).sqrt().cpu().numpy()
    lines.append(f"{p:<5d} {float(ev[:p].sum() / tot):7.3f}   {float(d2f.mean()):10.1f}                              {float(d2[torch.from_numpy(succ).cuda()].mean()):12.1f}              {float(d2[~torch.from_numpy(succ).cuda()].mean()):12.1f}            {auroc(~succ, d):8.3f}     {auroc(~succ, r):8.3f}")
lines += ["", "reading: AUROC < 0.5 means failure frames have SMALLER Mahalanobis distance than success frames in that subspace (they sit closer to the fit mean).", ""]
# (2) end clamping
sc = pd.read_parquet(f"{OUT}/c1_scores.parquet"); G = sc.groupby(["src", "ep"]).t.transform("max") // 6 + 1; late = sc.t.values >= (G.values - 5) * 6
fail = ~sc.succ.values; lines += ["== (2) frames whose 30-frame partner is clamped (last 5 grid steps of an episode)",
    f"share of frames clamped: success {late[~fail].mean():.3f}, failure {late[fail].mean():.3f}"]
cols = ["ens_std_M5", "ens_std_M10", "knn5_euc", "knn20_euc", "maha_pca128"]
lines.append("score            AUROC OOD_vs_ID: all frames | excluding clamped frames | clamped frames only")
for c in cols:
    v = sc[c].values; lines.append(f"{c:<16} {auroc(fail, v):6.3f}                 {auroc(fail[~late], v[~late]):6.3f}                    {auroc(fail[late], v[late]):6.3f}")
open(f"{OUT}/c1_diag.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
