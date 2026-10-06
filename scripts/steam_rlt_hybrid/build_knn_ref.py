"""Reference set + thresholds for the kNN confidence gate (same recipe as C1: fit = success episodes of r1_m0..r3_m2)."""
import numpy as np, pandas as pd, torch, json
R = "/workspace/libero_RL/tmp/hybrid"; Z = f"{R}/zrl"
FIT = ["r1_m0", "r1_m1", "r1_m2", "r2_m1", "r2_m2", "r3_m0", "r3_m1", "r3_m2"]; TEST = ["r4_m0", "r4_m1", "r4_m2", "eval_m0", "eval_m1", "eval_m2"]
def load(tags, succ_only=None):
    zs, ms = [], []
    for t in tags:
        z = np.load(f"{Z}/{t}.npy"); m = pd.read_parquet(f"{Z}/{t}_meta.parquet"); m["src"] = t
        ms.append(m); zs.append(z)
    m = pd.concat(ms).reset_index(drop=True); z = np.concatenate(zs).astype(np.float32); return z, m
zf, mf = load(FIT); keep = mf.succ.values; zf = zf[keep]; print("fit frames", len(zf), "episodes", mf[keep].groupby(["src", "ep"]).ngroups)
mu, sd = zf.mean(0), zf.std(0) + 1e-6; zs = (zf - mu) / sd
torch.save(dict(z=torch.from_numpy(zs), mu=torch.from_numpy(mu), sd=torch.from_numpy(sd)), f"{R}/knn_ref.pt")
zt, mt = load(TEST); zt = (zt - mu) / sd
ref = torch.from_numpy(zs).cuda(); dist = np.zeros(len(zt), np.float32)
for i in range(0, len(zt), 2048):
    d = torch.cdist(torch.from_numpy(zt[i:i+2048]).cuda(), ref); dist[i:i+2048] = d.topk(20, dim=1, largest=False).values[:, -1].cpu().numpy()
mt["dist"] = dist; ph = mt.t <= 150            # the actor acts until the first grasp (before ~step 150)
idf = mt[ph & mt.succ]; ood = mt[ph & ~mt.succ]
res = {"phase_frames_ID": int(len(idf)), "phase_frames_OOD": int(len(ood))}
for p in (50, 80, 90, 95, 99):
    thr = float(np.percentile(idf.dist, p)); res[f"p{p}"] = dict(threshold=round(thr, 3), id_kept=round(float((idf.dist <= thr).mean()), 3), ood_kept=round(float((ood.dist <= thr).mean()), 3))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{R}/knn_thresholds.json", "w"), indent=1)
