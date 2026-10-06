"""Figure of c2.py (test values at the validation-selected lambda). Reads c2_res.pkl; run alone: python c2_plot.py"""
import os, pickle, time, numpy as np
T0 = time.time(); IN = "/workspace/libero_RL/tmp/hybrid_env2"; OUT = "/workspace/libero_RL/tmp/hybrid"
RES = pickle.load(open(f"{IN}/c2_res.pkl", "rb"))["RES"]
class CFG(dict): pass
CFG = {n: None for n in RES}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
JP = [f for f in ("/root/.fonts/NotoSansCJKjp-Regular.otf", "/root/.fonts/NotoSansCJKjp-Bold.otf") if os.path.exists(f)]
for f in JP: fm.fontManager.addfont(f)
if JP: plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
T = (lambda ja, en: ja) if JP else (lambda ja, en: en)          # no Japanese font in this environment -> English labels
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"; C1, C2 = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 of the reference palette
names = list(CFG); x = np.arange(len(names)); w = 0.36
fig, axs = plt.subplots(1, 2, figsize=(15, 5.8), facecolor=SURF)
panels = [(T("偽陽性の割合(低いほど良い)", "False-positive share (lower is better)"), ("fp_fail", "fp_miss"),
           (T("失敗エピソードのフレーム", "frames of failed episodes"), T("空振りのフレーム", "miss frames")), (0, None)),
          (T("AUROC(高いほど良い)", "AUROC (higher is better)"), ("au_hold", "au_succ"),
           (T("保持 vs 空振り", "hold vs miss"), T("成功 vs 失敗エピソードのフレーム", "success vs failure episodes (frames)")), (0.5, 1.0))]
for ax, (ttl, ks, labs, lim) in zip(axs, panels):
    ax.set_facecolor(SURF)
    for j, (k, lab, c) in enumerate(zip(ks, labs, (C1, C2))):
        vals = [RES[n]["test"][k] for n in names]; bars = ax.bar(x + (j - .5) * w, vals, w - 0.03, color=c, label=lab, zorder=3)
        for b, n, v in zip(bars, names, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}\n(lam={RES[n]['sel'][k].split('=')[1]})", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=10, color=INK); ax.set_title(ttl, fontsize=12, color=INK, loc="left", pad=30)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(colors=MUTED)
    if lim[0] is not None: ax.set_ylim(bottom=lim[0])
    if lim[1] is not None: ax.set_ylim(top=lim[1])
    else: ax.set_ylim(top=max(RES[n]["test"][k] for n in names for k in ks) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK2)
fig.suptitle(T("C2: アンサンブルの種類ごとの、テスト(データの元が別)での値 — λ は検証データで選択", "C2: test values (separate data sources) per ensemble variant - lambda chosen on validation"), fontsize=14, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/c2_summary_ja.png", dpi=130, facecolor=SURF)
print("saved figure; elapsed %.0fs" % (time.time() - T0))
