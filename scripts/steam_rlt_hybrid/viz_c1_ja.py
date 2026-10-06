"""Japanese figure of the C1 result (made from tmp/hybrid_env1/c1_scores.parquet)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import rankdata
for f in ("/root/.fonts/NotoSansCJKjp-Regular.otf", "/root/.fonts/NotoSansCJKjp-Bold.otf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
R = "/workspace/libero_RL"; d = pd.read_parquet(f"{R}/tmp/hybrid_env1/c1_scores.parquet")
def auroc(y, s):
    y = np.asarray(y, bool); r = rankdata(s); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if min(n1, n0) >= 30 else np.nan
def roc(y, s, n=300):
    y = np.asarray(y, bool); o = np.argsort(-s); yy = y[o]; tp = np.cumsum(yy) / yy.sum(); fp = np.cumsum(~yy) / (~yy).sum(); ix = np.linspace(0, len(s) - 1, n).astype(int); return fp[ix], tp[ix]
M = {"アンサンブル（M=10）": "ens_std_M10", "kNN（k=20）": "knn20_euc", "マハラノビス（PCA128）": "maha_pca128", "組み合わせ（M=10 + kNN20）": "comb[ens_std_M10+knn20_euc]", "時間 t だけ（参考）": "ref_t (time only)"}
col = dict(zip(M, ["#d1242f", "#1f6feb", "#8b949e", "#2da44e", "#bf8700"]))
ood = ~d.succ.values; close = d.cmd.values > 0
miss = close & (d.gap.values <= 0.03); hold = close & (d.gap.values >= 0.04) & (d.gap.values <= 0.071)
bins = [(0, 60), (60, 120), (120, 180), (180, 10**6)]; bl = ["t<60", "60-120", "120-180", "t≥180"]
fig, ax = plt.subplots(2, 3, figsize=(18, 10.5)); ax = ax.ravel()
for n, c in M.items(): fp, tp = roc(ood, d[c].values); ax[0].plot(fp, tp, c=col[n], lw=2, label=f"{n}  AUROC {auroc(ood, d[c].values):.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("偽陽性率（成功エピソードのフレームを、範囲外と判定）"); ax[0].set_ylabel("検出率（失敗エピソードのフレームを検出）"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("(a) ROC 曲線: 失敗 vs 成功のフレーム（テスト 90,606 フレーム）")
for k, (n, c) in enumerate((("アンサンブルのばらつき（M=10）", "ens_std_M10"), ("kNN 距離（k=20）", "knn20_euc"))):
    a = ax[1 + k]; v = d[c].values; lo, hi = np.percentile(v, [0.5, 99.5]); b = np.linspace(lo, hi, 60) if k == 1 else np.logspace(np.log10(max(lo, 1e-2)), np.log10(hi), 60)
    a.hist(v[~ood], bins=b, density=True, alpha=.6, color="#1f6feb", label="成功エピソード"); a.hist(v[ood], bins=b, density=True, alpha=.6, color="#d1242f", label="失敗エピソード")
    if k == 0: a.set_xscale("log")
    a.set_xlabel(n); a.set_ylabel("密度"); a.legend(); a.set_title(f"({'bc'[k]}) {n} の分布")
w = .16; x = np.arange(4)
for i, (n, c) in enumerate(M.items()):
    ax[3].bar(x + (i - 2) * w, [auroc(ood[(d.t >= lo) & (d.t < hi)], d[c].values[(d.t >= lo) & (d.t < hi)]) for lo, hi in bins], w, color=col[n], label=n)
ax[3].set_xticks(x); ax[3].set_xticklabels(bl); ax[3].set_ylim(0.4, 1.13); ax[3].axhline(.5, c="k", lw=.8, ls=":"); ax[3].set_ylabel("AUROC"); ax[3].set_xlabel("ステップ t の区間")
ax[3].set_title("(d) 区間ごとの AUROC: 失敗 vs 成功\n序盤（t<60）は、すべての指標が下がる"); ax[3].legend(fontsize=7, loc="upper center", ncol=3)
sel = miss | hold
for i, (n, c) in enumerate(M.items()):
    vals = []
    for lo, hi in bins:
        m = sel & (d.t.values >= lo) & (d.t.values < hi); vals.append(auroc(miss[m], d[c].values[m]) if m.any() else np.nan)
    ax[4].bar(x + (i - 2) * w, vals, w, color=col[n], label=n)
ax[4].set_xticks(x); ax[4].set_xticklabels(bl); ax[4].set_ylim(0.4, 1.0); ax[4].axhline(.5, c="k", lw=.8, ls=":"); ax[4].set_ylabel("AUROC"); ax[4].set_xlabel("ステップ t の区間（t<60 は、空振りがなく、なし）")
ax[4].set_title("(e) 区間ごとの AUROC: 空振り vs 保持（成否を問わない版）\nアンサンブルが、わずかに上。t≥180 が最も低い")
p = [16, 64, 128, 256, 512]; ax[5].plot(p, [0.090, 0.257, 0.868, 0.946, 0.956], "o-", c="#8b949e", label="マハラノビス距離"); ax[5].plot(p, [0.932, 0.949, 0.951, 0.953, 0.948], "s-", c="#2da44e", label="PCA の残差ノルム")
ax[5].axhline(.5, c="k", lw=.8, ls=":"); ax[5].set_xscale("log", base=2); ax[5].set_xlabel("PCA の主成分の数"); ax[5].set_ylabel("AUROC（失敗 vs 成功）"); ax[5].legend()
ax[5].set_title("(f) マハラノビスは、主成分が少ないと 0.5 を下回る\n失敗は、分散の小さい方向に外れている（c1_diag.txt の値）")
fig.suptitle("C1: 「成功エピソードの分布の外か」を判定する指標の比較（環境1の結果）", fontsize=15); fig.tight_layout(); fig.savefig(f"{R}/tmp/hybrid/c1_ood_ja.png", dpi=115); print("saved")
