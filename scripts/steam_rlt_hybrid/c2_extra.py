"""Supplementary to c2.py (no retraining): lambda beyond the grid (32, 64, 128) and 'std only' (score = -std), using c2_predictions.pkl.
Same splits, same metrics. Reads c2_predictions.pkl; writes c2_result_extra.txt."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"; OUT = f"{R}/tmp/hybrid_env2"
VAL_SRC = ["r3_m2", "r4_m0"]; TEST_SRC = ["r4_m1", "r4_m2", "eval_m0", "eval_m1", "eval_m2"]
def src_dir(tag): return f"{D}/libero10_task0_eval_v2_m{tag[-1]}" if tag.startswith("eval_") else f"{D}/libero10_task0_train_v2_{tag}"
def table(tags):
    S, G, C = [], [], []
    for tag in tags:
        m = pd.read_parquet(f"{Z}/{tag}_meta.parquet"); L = [json.loads(l)["length"] for l in open(f"{src_dir(tag)}/meta/episodes.jsonl")]; idx = m.groupby("ep").indices
        for e in range(len(L)):
            if e not in idx: continue
            ii = idx[e]; mm = m.iloc[ii]
            if len(mm) != (L[e] + 5) // 6: continue
            ii = ii[np.argsort(mm.t.values)]; S.append(np.full(len(ii), bool(m.succ.iloc[ii[0]]))); G.append(m.gap.values[ii]); C.append(m.cmd.values[ii])
    return dict(succ=np.concatenate(S), gap=np.concatenate(G), cmd=np.concatenate(C))
VAL, TEST = table(VAL_SRC), table(TEST_SRC)
def auroc(y, s):
    y = np.asarray(y).astype(bool); r = rankdata(s); n1 = y.sum(); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))
def metrics(s, tab):
    succ, close = tab["succ"], tab["cmd"] > 0; fail = ~succ
    miss = close & (tab["gap"] <= 0.03); hold = close & (tab["gap"] >= 0.04) & (tab["gap"] <= 0.071); mh = miss | hold
    pos = s >= np.percentile(s[succ], 30)
    return [float(pos[fail].mean()), float(pos[miss].mean()), auroc(hold[mh], s[mh]), auroc(succ, s)]
PRED = pickle.load(open(f"{OUT}/c2_predictions.pkl", "rb")); lines = []
def P(x): lines.append(x)
P("Supplementary: lambda beyond the c2.py grid, and std only. columns: fp_fail fp_miss au_hold au_succ (TEST sources; VALIDATION in the first block)")
for name, (Pv, Pt) in PRED.items():
    P(f"\n=== {name} (M=5)")
    for lab, f in (("lam=16", lambda p: p.mean(0) - 16 * p.std(0)), ("lam=32", lambda p: p.mean(0) - 32 * p.std(0)), ("lam=64", lambda p: p.mean(0) - 64 * p.std(0)),
                   ("lam=128", lambda p: p.mean(0) - 128 * p.std(0)), ("-std only", lambda p: -p.std(0)), ("mean only", lambda p: p.mean(0))):
        v = metrics(f(Pv[:5]), VAL); t = metrics(f(Pt[:5]), TEST)
        P(f"{lab:10s} | VAL {' '.join(f'{x:.3f}' for x in v)} | TEST {' '.join(f'{x:.3f}' for x in t)}")
open(f"{OUT}/c2_result_extra.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
