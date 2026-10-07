"""Compare STEAM critics on the eval rollouts (64 episodes): success-vs-failure separation and hold-vs-miss separation of the continuous advantage."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
SH = "/workspace/libero_RL/tmp/steam_lora/shadow_eval"; tags = sys.argv[1:] or ["n32e2000_eval", "frozen2000_eval"]
def auroc(y, s):
    y = np.asarray(y, bool); r = rankdata(s); n1 = y.sum(); n0 = (~y).sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
frames = {t: [] for t in tags}; ep_rows = []
for m in range(3):
    D = f"{SH}/eval_v2_m{m}"; eps = [json.loads(l) for l in open(f"{D}/meta/episodes.jsonl")]
    adv = {t: pd.read_parquet(f"{D}/meta/advantages_{t}.parquet") for t in tags}
    for e in range(len(eps)):
        d = pd.read_parquet(f"/workspace/libero_RL/data/smolvla_recap/libero10_task0_eval_v2_m{m}/data/chunk-000/episode_{e:06d}.parquet", columns=["frame_index", "state", "actions", "is_success"])
        S = np.stack(d.state.values); A = np.stack(d.actions.values); succ = bool(d.is_success.any())
        for t in tags:
            a = adv[t][adv[t].episode_index == e].sort_values("frame_index")
            fi = a.frame_index.values; v = a.advantage_continuous.values
            gap = (S[:, 6] - S[:, 7])[fi]; cmd = A[:, 6][fi]
            frames[t].append(pd.DataFrame(dict(src=m, ep=e, succ=succ, t=fi, adv=v, gap=gap, cmd=cmd)))
F = {t: pd.concat(frames[t]).reset_index(drop=True) for t in tags}
for t in tags:
    f = F[t]; em = f.groupby(["src", "ep"]).agg(succ=("succ", "first"), mean=("adv", "mean"), p10=("adv", lambda x: np.percentile(x, 10)))
    close = f.cmd > 0; miss = close & (f.gap <= 0.03); hold = close & (f.gap >= 0.04) & (f.gap <= 0.071); sm = miss | hold
    print(f"[{t}] episodes {len(em)} (success {int(em.succ.sum())}), frames {len(f)}  | episode AUROC (mean adv): {auroc(em.succ, em['mean']):.3f}  (p10): {auroc(em.succ, em.p10):.3f}"
          f" | frame AUROC success-vs-failure frames: {auroc(f.succ, f.adv):.3f} | hold-vs-miss AUROC: {auroc(hold[sm], f.adv[sm]):.3f} (miss n={int(miss.sum())}, hold n={int(hold.sum())})")
if len(tags) == 2:
    a, b = F[tags[0]], F[tags[1]]; j = a.merge(b, on=["src", "ep", "t"], suffixes=("_a", "_b"))
    print(f"agreement of the two critics on {len(j)} common frames: Spearman {spearmanr(j.adv_a, j.adv_b)[0]:.3f}; mean advantage a {j.adv_a.mean():.3f}, b {j.adv_b.mean():.3f}")
# bootstrap over episodes (64 episodes resampled with replacement) for the AUROC differences
if len(tags) == 2:
    rng = np.random.default_rng(0); keys = sorted({(s, e) for s, e in zip(F[tags[0]].src, F[tags[0]].ep)})
    idx = {t: {k: g.index.values for k, g in F[t].groupby(["src", "ep"])} for t in tags}
    def stats(t, ks):
        ii = np.concatenate([idx[t][k] for k in ks]); f = F[t].loc[ii]
        close = f.cmd > 0; miss = close & (f.gap <= 0.03); hold = close & (f.gap >= 0.04) & (f.gap <= 0.071); sm = miss | hold
        return auroc(f.succ, f.adv), auroc(hold[sm], f.adv[sm])
    D1, D2 = [], []
    for _ in range(200):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        try: a = stats(tags[0], ks); b = stats(tags[1], ks)
        except Exception: continue
        D1.append(b[0] - a[0]); D2.append(b[1] - a[1])
    for name, D in (("success-vs-failure frame AUROC", D1), ("hold-vs-miss AUROC", D2)):
        print(f"bootstrap {tags[1]} - {tags[0]}: {name}: mean {np.mean(D):+.3f}  95% interval [{np.percentile(D, 2.5):+.3f}, {np.percentile(D, 97.5):+.3f}]  (n={len(D)})")
