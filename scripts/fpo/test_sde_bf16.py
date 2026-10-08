"""Flow-SDE PPO (model): ODE equivalence, rollout vs recompute, gradients, perturbation response, SDE sanity."""
import sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
R = "/workspace/libero_RL"; D = f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
cfg = OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
    smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=None, chunk_size=50, advantage_prompt=False,
    sde=dict(enabled=True, noise_level=0.5, num_steps=5, prefix_bf16=True, value_hidden_dims=[512, 256]),
    rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"})))
m = get_model(cfg).cuda().eval()
print("wrapper:", type(m).__name__, "| trainable params (M):", round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 2), "of", round(sum(p.numel() for p in m.parameters()) / 1e6, 1))
print("trainable groups:", sorted({".".join(n.split(".")[:4]) for n, p in m.named_parameters() if p.requires_grad})[:12])
def dec(path, T):
    out = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            out.append(fr.to_ndarray(format="rgb24"))
            if len(out) > T: break
    return np.stack(out)
TS = [0, 60, 120, 200]
a = dec(f"{D}/videos/chunk-000/image/episode_000000.mp4", max(TS)); w = dec(f"{D}/videos/chunk-000/wrist_image/episode_000000.mp4", max(TS))
S = np.stack(pd.read_parquet(f"{D}/data/chunk-000/episode_000000.parquet", columns=["state"]).state.values)
env_obs = dict(main_images=torch.from_numpy(a[TS]), wrist_images=torch.from_numpy(w[TS]), states=torch.from_numpy(S[TS].astype(np.float32)), task_descriptions=[TASK] * len(TS))


import random, time
random.seed(0)
sync = torch.cuda.synchronize
# (a) rollout vs recompute, same batch
actions, res = m.predict_action_batch(env_obs, mode="train"); fi = res["forward_inputs"]
out = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
print("(a) same batch: max |logprob diff| %.2e, |value diff| %.2e" % ((out["logprobs"].cpu() - res["prev_logprobs"]).abs().max().item(), (out["values"].cpu().reshape(-1) - res["prev_values"].reshape(-1)).abs().max().item()))
# (b) rollout on 2 samples (two calls), recompute in one batch of 4: batch-shape dependence of the bf16 prefix
def sub(o, idx): return dict(main_images=o["main_images"][idx], wrist_images=o["wrist_images"][idx], states=o["states"][idx], task_descriptions=[o["task_descriptions"][0]] * len(idx))
random.seed(1); _, r1 = m.predict_action_batch(sub(env_obs, [0, 1]), mode="train")
random.seed(2); _, r2 = m.predict_action_batch(sub(env_obs, [2, 3]), mode="train")
cat = {k: torch.cat([r1["forward_inputs"][k], r2["forward_inputs"][k]]) for k in r1["forward_inputs"]}
o4 = m(forward_inputs=cat, compute_logprobs=True, compute_values=True)
prev = torch.cat([r1["prev_logprobs"], r2["prev_logprobs"]]); pv = torch.cat([r1["prev_values"], r2["prev_values"]]).reshape(-1)
d = (o4["logprobs"].cpu() - prev).sum(-1)
print("(b) rollout in batches of 2, recompute in a batch of 4: per-chunk log-ratio at identical weights:", [round(x, 4) for x in d.tolist()], "| value diff max %.2e" % (o4["values"].cpu() - pv).abs().max().item())
# (c) eval path
with torch.no_grad():
    ea, _ = m.predict_action_batch(env_obs, mode="eval")
print("(c) eval actions", tuple(ea.shape), "finite", bool(torch.isfinite(ea).all()))
# (d) rollout speed (3 envs), fp32 prefix vs bf16 prefix
for flag in (False, True):
    m.prefix_bf16 = flag
    o3 = sub(env_obs, [0, 1, 2])
    with torch.no_grad():
        for _ in range(2): m.predict_action_batch(o3, mode="train")
        sync(); t = time.time()
        for _ in range(8): m.predict_action_batch(o3, mode="train")
        sync()
    print("(d) rollout chunk (3 envs, 5 steps), prefix_bf16=%s: %.0f ms" % (flag, (time.time() - t) / 8 * 1000))
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
