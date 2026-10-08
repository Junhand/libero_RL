"""Flow-SDE PPO (model): ODE equivalence, rollout vs recompute, gradients, perturbation response, SDE sanity."""
import sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
R = "/workspace/libero_RL"; D = f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
cfg = OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
    smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=None, chunk_size=50, advantage_prompt=False,
    sde=dict(enabled=True, noise_level=0.5, num_steps=5, value_hidden_dims=[512, 256]),
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

import random
from rlinf.models.embodiment.smolvla import sde
# ---- (a) a deterministic loop built from sample_mean_std equals lerobot's own Euler sampler (same noise)
from rlinf.models.embodiment.pi0_fast.lerobot_io import build_lerobot_batch_from_env_obs
batch = m._prepare_lerobot_batch(build_lerobot_batch_from_env_obs(env_obs, image_size=m.image_size))
images, img_masks, lt, lm, state = m._model_inputs(batch)
cfgp = m.policy.config; B = state.shape[0]
noise = torch.randn(B, cfgp.chunk_size, cfgp.max_action_dim, device=state.device)
with torch.no_grad():
    ref = m.policy.model.sample_actions(images, img_masks, lt, lm, state, noise=noise)
    feats, pm, pkv = sde.prefix_cache(m.policy.model, images, img_masks, lt, lm, state)
    x = noise; ts = sde.get_timesteps(m.denoise_steps, x.device)
    for k in range(m.denoise_steps):
        idx = torch.full((B,), k, device=x.device, dtype=torch.long); t = torch.full((B,), float(ts[k]), device=x.device)
        v = m.policy.model.denoise_step(pm, pkv, x, t).float()
        mean, std = sde.sample_mean_std(x, v, idx, sde=False, noise_level=0.5, num_steps=m.denoise_steps); x = mean
print(f"(a) ODE loop vs lerobot sample_actions: max abs diff {(x - ref).abs().max().item():.2e}")
# ---- (b) rollout vs recompute
random.seed(0)
actions, result = m.predict_action_batch(env_obs, mode="train")
fi = result["forward_inputs"]
print("actions", tuple(actions.shape), "| prev_logprobs", tuple(result["prev_logprobs"].shape), "| prev_values", tuple(result["prev_values"].shape), "| chosen step", fi["sde_step"][0].item())
print("forward_inputs:", {k: (tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in fi.items()})
out = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
dl = (out["logprobs"].cpu() - result["prev_logprobs"]).abs().max().item(); dv = (out["values"].cpu().reshape(-1) - result["prev_values"].reshape(-1)).abs().max().item()
print(f"(b) recompute vs rollout: max |logprobs diff| {dl:.2e}, max |values diff| {dv:.2e}; logprobs mean {out['logprobs'].mean().item():.4f}")
# ---- (c) gradients; zero update -> ratio 1; response to a small perturbation of the action expert
m.train(); m.zero_grad(set_to_none=True)
out = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
(out["logprobs"].sum() + out["values"].mean()).backward()
g = {}
for n, p in m.named_parameters():
    if p.requires_grad: g.setdefault(".".join(n.split(".")[:3]) if "policy" in n else n.split(".")[0], []).append(p.grad is not None and p.grad.abs().sum().item() > 0)
print("(c) trainable tensors with non-zero grad (ok / total):", {k: (sum(v), len(v)) for k, v in g.items()})
print("    frozen params with grad:", sum(1 for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None))
base = out["logprobs"].detach().sum(-1)
with torch.no_grad():
    for p in m.policy.model.action_out_proj.parameters(): p.add_(1e-3 * torch.randn_like(p))
    pert = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)["logprobs"].sum(-1)
print("    log-ratio after a 1e-3 weight perturbation (per chunk):", [round(v, 3) for v in (pert - base).tolist()])
# ---- (d) SDE sanity: for a fixed observation, stochastic step moves the final action only slightly vs ODE
m.eval()
with torch.no_grad():
    random.seed(1); torch.manual_seed(1)
    ode = sde.prefix_cache  # keep the namespace used above
    diffs = []
    for chosen in (0, 4, 9):
        random.seed(0)
        x = noise.clone(); xs = noise.clone()
        for k in range(m.denoise_steps):
            idx = torch.full((B,), k, device=x.device, dtype=torch.long); t = torch.full((B,), float(ts[k]), device=x.device)
            v = m.policy.model.denoise_step(pm, pkv, xs, t).float()
            mean, std = sde.sample_mean_std(xs, v, idx, sde=(k == chosen), noise_level=0.5, num_steps=m.denoise_steps)
            xs = mean + torch.randn_like(xs) * std
        diffs.append((xs - ref)[:, :10, :7].abs().mean().item())
print("(d) mean |SDE action - ODE action| on executed steps, stochastic step 0/4/9:", [round(d, 3) for d in diffs], "| ODE action scale", round(ref[:, :10, :7].abs().mean().item(), 3))
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
