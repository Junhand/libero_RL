import sys, json, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("/root/.fonts/NotoSansCJKjp-Regular.otf", "/root/.fonts/NotoSansCJKjp-Bold.otf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(2)
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"
sys.path.insert(0, f"{R}/RLinf")
import importlib.util
spec = importlib.util.spec_from_file_location("os_", f"{R}/tmp/rlinf_dev/rlinf/algorithms/rlt/offset_score.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sc = m.OffsetScorer(f"{R}/tmp/hybrid/offset_mlp.pt", torch.device("cpu")); Lmax = sc.lmax
df = pd.read_parquet(f"{R}/tmp/hybrid/e0b_frames.parquet")
# ---- (a) predicted vs true offset on the independent eval rollouts
rows = []
for i in range(3):
    tag = f"eval_m{i}"; z = np.load(f"{Z}/{tag}.npy").astype(np.float32); mt = pd.read_parquet(f"{Z}/{tag}_meta.parquet")
    L = [json.loads(l)["length"] for l in open(f"{D}/libero10_task0_eval_v2_m{i}/meta/episodes.jsonl")]
    for e in mt.ep.unique():
        mm = mt[mt.ep == e].sort_values("t"); idx = mm.index.values; G = len(idx); succ = bool(mm.succ.iloc[0])
        for d in (-5, -3, -1, 1, 3, 5):
            ii = np.arange(max(0, -d), min(G, G - d)); 
            if len(ii) == 0: continue
            p = sc(torch.from_numpy(z[idx[ii]]), torch.from_numpy(z[idx[ii + d]])).numpy()
            rows += [(succ, d * 6 * Lmax / L[e], v) for v in p]
a = pd.DataFrame(rows, columns=["succ", "true", "pred"])
fig, ax = plt.subplots(2, 2, figsize=(13, 10)); ax = ax.ravel()
for s, c, lab in ((True, "#1f6feb", "成功エピソード"), (False, "#d1242f", "失敗エピソード")):
    q = a[a.succ == s]; q = q.sample(min(len(q), 4000), random_state=0)
    ax[0].scatter(q.true, q.pred, s=4, alpha=0.25, c=c, label=f"{lab}（{len(a[a.succ == s]):,} 対）")
lim = [a.true.min() - 5, a.true.max() + 5]; ax[0].plot(lim, lim, "k--", lw=1); ax[0].axhline(0, c="gray", lw=.5); ax[0].axvline(0, c="gray", lw=.5)
sg = {s: (np.sign(a[a.succ == s].pred) == np.sign(a[a.succ == s].true)).mean() for s in (True, False)}
ax[0].set_title(f"(a) 予測したオフセット vs 正解（eval 64 エピソード）\n前後の判定の正解率: 成功 {sg[True]:.3f} / 失敗 {sg[False]:.3f}")
ax[0].set_xlabel("正解（長さで正規化した時間差）"); ax[0].set_ylabel("予測"); ax[0].legend(markerscale=4)
# ---- (b) score by gripper state
cats = {"開いている": df[~df.close].score, "閉じる指令・保持\n（隙間 0.04–0.071）": df[df.hold].score, "閉じる指令・空振り\n（隙間 ≤ 0.03）": df[df.miss].score}
bp = ax[1].boxplot(list(cats.values()), tick_labels=list(cats.keys()), showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], ("#8b949e", "#2da44e", "#d1242f")): patch.set_facecolor(c); patch.set_alpha(.6)
for k, (n, v) in enumerate(cats.items(), 1): ax[1].text(k, v.median(), f" 平均 {v.mean():.1f}\n n={len(v):,}", va="bottom", ha="left", fontsize=9)
ax[1].set_ylabel("スコア（前向き 30 フレームのオフセット）"); ax[1].set_title("(b) 把持の状態ごとのスコア（未使用 + eval の 62,599 フレーム）\n保持と空振りの AUROC = 0.832")
# ---- (c) episode mean score
em = df.groupby(["src", "ep"]).agg(succ=("succ", "first"), mean=("score", "mean")).reset_index()
bins = np.linspace(em["mean"].min(), em["mean"].max(), 40)
ax[2].hist(em[em.succ]["mean"], bins=bins, alpha=.7, color="#1f6feb", label=f"成功 {int(em.succ.sum())} 件"); ax[2].hist(em[~em.succ]["mean"], bins=bins, alpha=.7, color="#d1242f", label=f"失敗 {int((~em.succ).sum())} 件")
ax[2].set_xlabel("エピソードの平均スコア"); ax[2].set_ylabel("エピソード数"); ax[2].legend()
ax[2].set_title("(c) エピソードの平均スコア（AUROC ≈ 0.999）\n注: 失敗エピソードはすべて上限 520 ステップまで続く（長い）")
# ---- (d) time course of example episodes
ev = df[df.src.str.startswith("eval")]
ex = [("成功", ev[ev.succ].groupby(["src", "ep"]).size().index[0]), ("失敗", ev[~ev.succ].groupby(["src", "ep"]).size().index[0])]
cols = {"成功": "#1f6feb", "失敗": "#d1242f"}
for lab, (src, ep) in ex:
    q = ev[(ev.src == src) & (ev.ep == ep)].sort_values("t")
    ax[3].plot(q.t, q.score, "-o", ms=3, c=cols[lab], label=f"{lab}（{src} ep{ep}、{int(q['T'].iloc[0])} ステップ）")
    mm = q[q.miss]; ax[3].scatter(mm.t, mm.score, marker="x", c="k", s=40, zorder=5, label="空振り（×）" if lab == "失敗" else None)
ax[3].axhline(0, c="gray", lw=.5); ax[3].set_xlabel("ステップ"); ax[3].set_ylabel("スコア"); ax[3].legend(fontsize=9)
ax[3].set_title("(d) スコアの時間変化の例（eval の 1 エピソードずつ）")
fig.suptitle("E0: RL token から時間オフセットを予測（小さな MLP）— 結果", fontsize=15); fig.tight_layout()
fig.savefig(f"{R}/tmp/hybrid/e0_overview.png", dpi=130)
print("saved; pairs", len(a), "episodes", len(em))
