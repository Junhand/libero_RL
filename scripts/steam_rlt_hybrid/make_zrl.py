"""Compute RL tokens (960-d) from cached image-token features with the Stage-1 encoder; keep only frames on the t%6==0 grid."""
import sys, glob, os, numpy as np, pandas as pd, torch
sys.path.insert(0, "/workspace/libero_RL/RLinf")
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
R = "/workspace/libero_RL/tmp/rlt_stage1"
ck = torch.load(f"{R}/rlt_stage1_ckpt.pt", map_location="cpu")
mod = RLTTokenTransformer(input_dim=960, embed_dim=960, prefix_seq_len=128, num_layers=2, num_heads=8)
mod.load_state_dict(ck["net"]); mod.eval().cuda().requires_grad_(False); scale = float(ck["scale"])
print("scale", scale, flush=True)
out = "/workspace/libero_RL/tmp/hybrid/zrl"; os.makedirs(out, exist_ok=True)
tags = sorted({os.path.basename(f)[:-4] for d in ("feats", "feats_rest", "feats_demo", "feats_demo_rest") for f in glob.glob(f"{R}/{d}/*.npy")})
for tag in tags:
    zs, metas = [], []
    for d in ("feats", "feats_rest", "feats_demo", "feats_demo_rest"):
        f = f"{R}/{d}/{tag}.npy"
        if not (os.path.exists(f) and os.path.exists(f"{R}/{d}/{tag}.done")): continue
        m = pd.read_parquet(f"{R}/{d}/{tag}_meta.parquet"); X = np.load(f, mmap_mode="r")
        idx = np.where(m.t.values % 6 == 0)[0]
        for i in range(0, len(idx), 256):
            b = torch.from_numpy(np.ascontiguousarray(X[idx[i:i+256]])).cuda().float() / scale
            with torch.no_grad(): zs.append(mod.encode_flat(b).float().cpu().numpy().astype(np.float16))
        metas.append(m.iloc[idx])
    if not zs: continue
    z = np.concatenate(zs); m = pd.concat(metas).reset_index(drop=True)
    np.save(f"{out}/{tag}.npy", z); m.to_parquet(f"{out}/{tag}_meta.parquet")
    print(tag, z.shape, "episodes", m.ep.nunique(), flush=True)
