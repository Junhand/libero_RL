"""Second-grasp conversion paired by the initial state. All recordings use use_ordered_reset_state_ids=True, so the k-th
episode file (sorted by episode_index of the first row) is the same initial state in every run; episodes are matched by that order."""
import sys, glob, numpy as np, pyarrow.parquet as pq
from diag_analyze import runs
def load(root):
    eps = []
    for f in glob.glob(f"{root}/rank_*/id_*/data/chunk-*/file-*.parquet"):
        t = pq.read_table(f, columns=["state", "actions", "is_success", "episode_index"]).to_pandas()
        for e, g in t.groupby("episode_index", sort=True):
            if len(g) < 100: continue
            S = np.stack(g.state.values); A = np.stack(g.actions.values); gap = S[:, 6] - S[:, 7]
            H = runs((A[:, 6] > 0) & (gap > 0.04) & (gap < 0.071), 8)
            eps.append((int(e), len(H), bool(g.is_success.values[-1]), S[0]))
    return sorted(eps, key=lambda x: x[0])
R = {n: load(n if "/" in n else f"diag/{n}") for n in sys.argv[1:]}
names = list(R); N = {n: len(v) for n, v in R.items()}; print("episodes:", N)
# alignment check: the first-frame states must agree between runs for the same position
ref = R[names[0]]
for n in names[1:]:
    d = [np.abs(a[3] - b[3])[:3].max() for a, b in zip(ref, R[n])]; print(f"alignment {names[0]} vs {n}: max |Δ eef pos at t=0| median {np.median(d):.4f}, max {np.max(d):.4f}")
K = min(N.values()); both = [i for i in range(K) if all(R[n][i][1] >= 1 for n in names)]
for n in names:
    v = R[n][:K]; k1 = [i for i in range(K) if v[i][1] >= 1]; k2 = [i for i in k1 if v[i][1] >= 2]
    print(f"{n}: first hold {len(k1)}/{K} | 2nd|1st {len(k2)}/{len(k1)} = {len(k2)/max(len(k1),1):.2f} | success {np.mean([x[2] for x in v]):.2f}")
print("initial states with a first hold in ALL runs:", len(both))
for n in names:
    c = sum(R[n][i][1] >= 2 for i in both); print(f"  {n}: 2nd|1st on those = {c}/{len(both)} = {c/max(len(both),1):.2f}")
print("per-state (1 = 2nd hold, . = no 2nd hold, - = no first hold):")
for n in names: print(f"  {n:>22}", "".join("1" if x[1] >= 2 else ("." if x[1] == 1 else "-") for x in R[n][:K]))
