"""FPO S0: unit test of smolvla_fpo_cfm_loss against lerobot's own training forward (policy.model.forward) with fixed (time, noise)."""
import os, sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
from rlinf.models.embodiment.smolvla.fpo import smolvla_fpo_cfm_loss
from rlinf.models.embodiment.pi0_fast.lerobot_io import build_lerobot_batch_from_env_obs
R = "/workspace/libero_RL"; D = f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
if os.environ.get("STRICT") == "1":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
print("tf32 matmul:", torch.backends.cuda.matmul.allow_tf32, "| precision:", torch.get_float32_matmul_precision())
cfg = OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
    smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=0, chunk_size=50, advantage_prompt=False,
    rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"})))
m = get_model(cfg).cuda().eval(); pol = m.policy; mod = pol.model
if os.environ.get("EXPERT_FP32") == "1":
    mod.float()
print("expert dtype:", next(mod.vlm_with_expert.lm_expert.parameters()).dtype, "| vlm dtype:", next(mod.vlm_with_expert.vlm.parameters()).dtype)
print("policy dtype:", next(mod.parameters()).dtype, "| chunk_size", pol.config.chunk_size, "max_action_dim", pol.config.max_action_dim, "use_cache", pol.config.use_cache, "num_steps", pol.config.num_steps)
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
obs = dict(main_images=torch.from_numpy(a[TS]), wrist_images=torch.from_numpy(w[TS]), states=torch.from_numpy(S[TS].astype(np.float32)), task_descriptions=[TASK] * len(TS))
with torch.no_grad():
    b = m._prepare_lerobot_batch(build_lerobot_batch_from_env_obs(obs, image_size=m.image_size))
    images, img_masks = pol.prepare_images(b); state = pol.prepare_state(b)
    lang_tokens = b["observation.language.tokens"]; lang_masks = b["observation.language.attention_mask"]
    act7 = pol.predict_action_chunk(b)                       # [B, 50, 7] normalised policy sample
B, C, D32 = len(TS), pol.config.chunk_size, pol.config.max_action_dim
actions = torch.zeros(B, C, D32, device=act7.device); actions[:, : act7.shape[1], :7] = act7[:, :C, :7].float()
N, H = 4, 10
g = torch.Generator(device="cuda").manual_seed(123)
timesteps = torch.rand(B, N, device="cuda", generator=g).clamp(1e-3, 1 - 1e-3); noise = torch.randn(B, N, C, D32, device="cuda", generator=g)
print("actions", tuple(actions.shape), "timesteps", tuple(timesteps.shape), "noise", tuple(noise.shape))
# ---- T1: equality with the reference training forward
with torch.no_grad():
    mine = smolvla_fpo_cfm_loss(mod, images, img_masks, lang_tokens, lang_masks, state, actions, timesteps, noise, action_dim=7, num_steps=H)
    ref = torch.stack([mod.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise=noise[:, n], time=timesteps[:, n])[:, :H, :7].mean(-1) for n in range(N)], dim=-1)
print("mine", tuple(mine.shape), "ref", tuple(ref.shape))
err = (mine - ref).abs(); rel = err / ref.abs().clamp_min(1e-8)
print(f"T1 max abs diff {err.max().item():.3e}, max rel diff {rel.max().item():.3e}; loss scale mean {ref.mean().item():.4f} (min {ref.min().item():.4f}, max {ref.max().item():.4f})")
print("T1 PASS" if torch.allclose(mine, ref, rtol=1e-3, atol=1e-5) else "T1 FAIL")
# ---- T2: determinism
with torch.no_grad():
    mine2 = smolvla_fpo_cfm_loss(mod, images, img_masks, lang_tokens, lang_masks, state, actions, timesteps, noise, action_dim=7, num_steps=H)
print("T2 deterministic:", torch.equal(mine, mine2))
# ---- T3: gradients flow only to the action expert (VLM frozen)
groups = {}
for n, p in mod.named_parameters(): groups.setdefault(n.split(".")[0] + "." + (n.split(".")[1] if n.startswith("vlm_with_expert") else ""), []).append((n, p))
print("T3 trainable parameter groups (requires_grad):", {k: (sum(p.numel() for _, p in v if p.requires_grad) / 1e6, sum(p.numel() for _, p in v) / 1e6) for k, v in groups.items()})
mod.zero_grad(set_to_none=True)
with torch.enable_grad():
    loss = smolvla_fpo_cfm_loss(mod, images, img_masks, lang_tokens, lang_masks, state, actions, timesteps, noise, action_dim=7, num_steps=H)
    loss.mean().backward()
with_grad = {k: sum(1 for _, p in v if p.grad is not None and p.grad.abs().sum() > 0) for k, v in groups.items()}
tot = {k: sum(1 for _, p in v if p.requires_grad) for k, v in groups.items()}
print("T3 params with non-zero grad / trainable:", {k: (with_grad[k], tot[k]) for k in groups})
vlm_grad = sum(1 for n, p in mod.named_parameters() if ".vlm." in n and p.grad is not None)
print("T3 VLM params that received a gradient:", vlm_grad)
# ---- T4: sensitivity: a tiny perturbation of the expert changes the loss by a small, nonzero amount (log-ratio scale)
with torch.no_grad():
    L0 = smolvla_fpo_cfm_loss(mod, images, img_masks, lang_tokens, lang_masks, state, actions, timesteps, noise, action_dim=7, num_steps=H)
    for scale in (1e-5, 1e-4, 1e-3):
        saved = [p.detach().clone() for p in mod.parameters() if p.requires_grad]
        torch.manual_seed(0)
        for p in mod.parameters():
            if p.requires_grad: p.add_(torch.randn_like(p) * scale * p.abs().mean())
        L1 = smolvla_fpo_cfm_loss(mod, images, img_masks, lang_tokens, lang_masks, state, actions, timesteps, noise, action_dim=7, num_steps=H)
        i = 0
        for p in mod.parameters():
            if p.requires_grad: p.copy_(saved[i]); i += 1
        lr = (L0 - L1)
        print(f"T4 perturbation {scale:g}: log-ratio (L_old - L_new) mean {lr.mean().item():+.2e}, std {lr.std().item():.2e}, max |.| {lr.abs().max().item():.2e}; ratio-1 max {(lr.exp() - 1).abs().max().item():.2e}")
torch.cuda.synchronize(); print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
