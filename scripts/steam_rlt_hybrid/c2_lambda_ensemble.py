"""C2: how to choose lambda for 'mean - lambda * std' of an offset-MLP ensemble on RL tokens, and which ensemble variant helps.
Split by data SOURCE (not by episode). lambda and M are chosen on the validation sources; the test sources are only read at the chosen setting.
All outputs go to tmp/hybrid_env2/. Inputs (tmp/hybrid/zrl) are read only.
  RLinf/.venv-smolvla/bin/python tmp/hybrid_env2/c2.py"""
import json, os, sys, time, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import rankdata
T0 = time.time()
R = "/workspace/libero_RL"; Z = f"{R}/tmp/hybrid/zrl"; D = f"{R}/data/smolvla_recap"; OUT = f"{R}/tmp/hybrid_env2"
TRAIN_SRC = ["r1_m0", "r1_m1", "r1_m2", "r2_m1", "r2_m2", "r3_m0", "r3_m1"]      # successful episodes only
VAL_SRC = ["r3_m2", "r4_m0"]                                                       # all episodes: choose lambda and M
TEST_SRC = ["r4_m1", "r4_m2", "eval_m0", "eval_m1", "eval_m2"]                     # all episodes: final values at the chosen setting
LAMBDAS = [0, 0.5, 1, 2, 4, 8, 16]; M = 5; EPOCHS = 15; BATCH = 1024; DS = [d for d in range(-5, 6) if d != 0]; LOOK = 5   # 5 grid steps = 30 frames
NOISE, DROP = 0.1, 0.1
dev = "cuda"

def src_dir(tag): return f"{D}/libero10_task0_eval_v2_m{tag[-1]}" if tag.startswith("eval_") else f"{D}/libero10_task0_train_v2_{tag}"
def load_src(tag):
    """Episodes of one source: z (float32, grid frames t%6==0, sorted by t), succ (meta, episode-level), gap, cmd, T."""
    z = np.load(f"{Z}/{tag}.npy", mmap_mode="r"); m = pd.read_parquet(f"{Z}/{tag}_meta.parquet")
    L = [json.loads(l)["length"] for l in open(f"{src_dir(tag)}/meta/episodes.jsonl")]; idx = m.groupby("ep").indices; eps = []
    for e in range(len(L)):
        if e not in idx: continue
        ii = idx[e]; mm = m.iloc[ii]; G = (L[e] + 5) // 6
        if len(mm) != G: continue                                   # incomplete grid
        o = np.argsort(mm.t.values); ii = ii[o]
        eps.append(dict(src=tag, ep=e, T=L[e], succ=bool(m.succ.iloc[ii[0]]), z=np.asarray(z[ii], dtype=np.float32),
                        gap=m.gap.values[ii].astype(np.float32), cmd=m.cmd.values[ii].astype(np.float32)))
    return eps
tr_eps = [e for t in TRAIN_SRC for e in load_src(t) if e["succ"]]
val_eps = [e for t in VAL_SRC for e in load_src(t)]; test_eps = [e for t in TEST_SRC for e in load_src(t)]
def info(name, es): print(f"{name}: {len(es)} episodes ({sum(e['succ'] for e in es)} success), {sum(len(e['z']) for e in es):,} frames", flush=True)
info("train (success only)", tr_eps); info("val", val_eps); info("test", test_eps)
LMAX = float(np.percentile([e["T"] for e in tr_eps], 95)); print("Lmax", LMAX)
allz = np.concatenate([e["z"] for e in tr_eps]); MU, SD = allz.mean(0), allz.std(0) + 1e-6; del allz
norm = lambda z: ((z - MU) / SD).astype(np.float32)

# ---- training data: frames on the GPU once; pairs as index arrays (one set per episode, so bootstrap can resample episodes)
Zg = torch.from_numpy(np.concatenate([norm(e["z"]) for e in tr_eps])).to(dev)
P1, P2, PY = [], [], []; off = 0
for e in tr_eps:
    G = len(e["z"]); a, b, y = [], [], []
    for i in range(G):
        for d in DS:
            if 0 <= i + d < G: a.append(off + i); b.append(off + i + d); y.append(d * 6 * LMAX / e["T"])
    P1.append(torch.tensor(a, device=dev)); P2.append(torch.tensor(b, device=dev)); PY.append(torch.tensor(y, dtype=torch.float32, device=dev)); off += G
print("train frames", off, "pairs", sum(len(p) for p in P1), flush=True)

class Net(nn.Module):
    def __init__(s, h=512, L=2, p=0.0):
        super().__init__(); ls = [nn.Linear(960 * 3, h), nn.GELU(), nn.Dropout(p)]
        for _ in range(L - 1): ls += [nn.Linear(h, h), nn.GELU(), nn.Dropout(p)]
        s.f = nn.Sequential(*ls, nn.Linear(h, 1))
    def forward(s, a, b): return s.f(torch.cat([a, b, b - a], -1)).squeeze(-1)

def train_member(h, L, p, noise, boot, seed):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    ids = rs.integers(0, len(tr_eps), len(tr_eps)) if boot else np.arange(len(tr_eps))
    I1 = torch.cat([P1[i] for i in ids]); I2 = torch.cat([P2[i] for i in ids]); Y = torch.cat([PY[i] for i in ids]); N = len(Y)
    net = Net(h, L, p).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * (N // BATCH)); net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(N, device=dev)
        for i in range(N // BATCH):
            ix = perm[i * BATCH:(i + 1) * BATCH]; a, b = Zg[I1[ix]], Zg[I2[ix]]
            if noise > 0: a = a + noise * torch.randn_like(a); b = b + noise * torch.randn_like(b)
            loss = nn.functional.smooth_l1_loss(net(a, b), Y[ix]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()

# ---- evaluation frames (val / test): score of frame t = f(z_t, z_{t+30 frames}), clamped at the episode end
def eval_table(es):
    z = np.concatenate([norm(e["z"]) for e in es]); u = []; o = 0
    for e in es: G = len(e["z"]); u.append(np.minimum(np.arange(G) + LOOK, G - 1) + o); o += G
    cat = lambda k: np.concatenate([np.full(len(e["z"]), e[k]) if k == "succ" else e[k] for e in es])
    return dict(z=torch.from_numpy(z).to(dev), u=torch.from_numpy(np.concatenate(u)).to(dev), succ=cat("succ").astype(bool), gap=cat("gap"), cmd=cat("cmd"))
VAL, TEST = eval_table(val_eps), eval_table(test_eps)
def predict(net, tab):
    out = []
    with torch.no_grad():
        for i in range(0, len(tab["u"]), 8192): out.append(net(tab["z"][i:i + 8192], tab["z"][tab["u"][i:i + 8192]]).cpu().numpy())
    return np.concatenate(out)

def auroc(y, s):
    y = np.asarray(y).astype(bool); r = rankdata(s); n1 = y.sum(); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))
def metrics(s, tab):
    succ, close = tab["succ"], tab["cmd"] > 0; fail = ~succ
    miss = close & (tab["gap"] <= 0.03); hold = close & (tab["gap"] >= 0.04) & (tab["gap"] <= 0.071); mh = miss | hold
    pos = s >= np.percentile(s[succ], 30)                                        # 70% of the success-episode frames are positive
    return dict(fp_fail=float(pos[fail].mean()), fp_miss=float(pos[miss].mean()), au_hold=auroc(hold[mh], s[mh]), au_succ=auroc(succ, s))
KEYS = ["fp_fail", "fp_miss", "au_hold", "au_succ"]; LOWER = {"fp_fail": True, "fp_miss": True, "au_hold": False, "au_succ": False}
def score(Pm, lam): return Pm.mean(0) - lam * Pm.std(0)
def table(Pm, tab):
    rows = {f"lam={l}": metrics(score(Pm, l), tab) for l in LAMBDAS}; rows["min"] = metrics(Pm.min(0), tab); return rows

# ---- variants
CFG = {"V0 seeds only": dict(archs=[(512, 2)] * 5, p=0, noise=0, boot=False),
       "V1 bootstrap": dict(archs=[(512, 2)] * 10, p=0, noise=0, boot=True),            # 10 members: first 5 are V1; all 10 used for the M study
       "V2 architectures": dict(archs=[(256, 2), (512, 2), (1024, 2), (512, 3), (256, 3)], p=0, noise=0, boot=False),
       "V3 noise+dropout": dict(archs=[(512, 2)] * 5, p=DROP, noise=NOISE, boot=False),
       "V4 bootstrap+V3": dict(archs=[(512, 2)] * 5, p=DROP, noise=NOISE, boot=True)}
PRED = {}
for name, c in CFG.items():
    pv, pt = [], []
    for k, (h, L) in enumerate(c["archs"]):
        net = train_member(h, L, c["p"], c["noise"], c["boot"], 1000 + k)
        pv.append(predict(net, VAL)); pt.append(predict(net, TEST)); del net
        print(f"{name} member {k} ({h}x{L}) done  [{time.time() - T0:.0f}s]", flush=True)
    PRED[name] = (np.stack(pv), np.stack(pt))
pickle.dump(PRED, open(f"{OUT}/c2_predictions.pkl", "wb"))

# ---- selection: per metric, the lambda with the best VALIDATION value; the test value is read at that lambda only
def pick(tv, key):
    names = [f"lam={l}" for l in LAMBDAS]; vals = [tv[n][key] for n in names]
    return names[int(np.argmin(vals) if LOWER[key] else np.argmax(vals))]
RES = {}; lines = []
def P(*a): lines.append(" ".join(str(x) for x in a))
def fmt(r): return "  ".join(f"{r[k]:.3f}" for k in KEYS)
P("C2: lambda selection on validation sources, final values on test sources (the splits differ by data source; independence of initial states between sources is NOT checked)")
P(f"train: {len(tr_eps)} success episodes ({', '.join(TRAIN_SRC)}); val: {len(val_eps)} episodes ({', '.join(VAL_SRC)}); test: {len(test_eps)} episodes ({', '.join(TEST_SRC)})")
P(f"val frames {len(VAL['u']):,} (success-episode frames {int(VAL['succ'].sum()):,}); test frames {len(TEST['u']):,} (success-episode frames {int(TEST['succ'].sum()):,})")
P("columns: fp_fail = share of FAILURE-episode frames that become positive (lower is better); fp_miss = share of MISS frames (cmd>0, gap<=0.03) that become positive (lower);")
P("         au_hold = AUROC hold (cmd>0, 0.04<=gap<=0.071) vs miss (higher); au_succ = AUROC success-episode frames vs failure-episode frames (higher).")
P("positive = score >= the 30th percentile of the success-episode frames (70% of them are positive). score = mean - lambda*std over M members; 'min' = minimum over members.")
for name in CFG:
    Pv, Pt = PRED[name]; Pv5, Pt5 = Pv[:M], Pt[:M]; tv, tt = table(Pv5, VAL), table(Pt5, TEST)
    P(f"\n=== {name} (M={M})"); P(f"{'':10s} | VALIDATION: {'  '.join(KEYS)} | TEST: {'  '.join(KEYS)}")
    for r in tv: P(f"{r:10s} | {fmt(tv[r])} | {fmt(tt[r])}")
    sel = {k: pick(tv, k) for k in KEYS}; RES[name] = dict(sel=sel, test={k: tt[sel[k]][k] for k in KEYS}, val={k: tv[sel[k]][k] for k in KEYS},
        test_lam4={k: tt["lam=4"][k] for k in KEYS}, test_lam0={k: tt["lam=0"][k] for k in KEYS}, test_min={k: tt["min"][k] for k in KEYS},
        test_oracle={k: (min if LOWER[k] else max)(tt[f"lam={l}"][k] for l in LAMBDAS) for k in KEYS}, oracle_lam={k: pick(tt, k) for k in KEYS})
    P(f"selected lambda (validation): " + ", ".join(f"{k}: {sel[k]}" for k in KEYS))
    P("TEST value at the selected lambda:   " + "  ".join(f"{k}={RES[name]['test'][k]:.3f}" for k in KEYS))
    P("TEST value if lambda were chosen ON THE TEST sources (optimistic): " + "  ".join(f"{k}={RES[name]['test_oracle'][k]:.3f}({RES[name]['oracle_lam'][k]})" for k in KEYS))
# ---- M study (V1 bootstrap, 10 members): min and mean - lambda*std, lambda picked on validation
P("\n=== M study, V1 bootstrap (members 0..M-1; lambda chosen on validation; test at that lambda)")
P(f"{'M':>3s} {'rule':8s} | " + "  ".join(KEYS) + " | selected lambda per metric")
Pv, Pt = PRED["V1 bootstrap"]; MS = {}
for m in (1, 3, 5, 10):
    tv, tt = table(Pv[:m], VAL), table(Pt[:m], TEST)
    P(f"{m:3d} {'min':8s} | {fmt(tt['min'])} |")
    sel = {k: pick(tv, k) for k in KEYS}; MS[m] = {k: tt[sel[k]][k] for k in KEYS}
    P(f"{m:3d} {'mean-lam*std':8s} | " + "  ".join(f"{MS[m][k]:.3f}" for k in KEYS) + " | " + ", ".join(f"{k}: {sel[k]}" for k in KEYS))
    MS[m]["min"] = tt["min"]
# ---- optimism: V0 (same design as the previous run E0e) at the previous protocol (lambda=4 chosen on the evaluation data itself) vs validation-selected
P("\n=== Optimism check on V0 (seeds only, the previous design)")
r = RES["V0 seeds only"]
for k in KEYS: P(f"{k:8s}: test at lambda=4 {r['test_lam4'][k]:.3f} | test at validation-selected {r['test'][k]:.3f} ({r['sel'][k]}) | test-oracle (lambda chosen on test) {r['test_oracle'][k]:.3f} ({r['oracle_lam'][k]}) | single member (lambda=0, mean) {r['test_lam0'][k]:.3f}")
P(f"\nelapsed {time.time() - T0:.0f}s")
open(f"{OUT}/c2_result.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines[-40:]))
pickle.dump(dict(RES=RES, MS=MS), open(f"{OUT}/c2_res.pkl", "wb"))

# ---- figure: test values at the validation-selected lambda, per variant
exec(open(f"{OUT}/c2_plot.py").read())          # figure: see c2_plot.py
