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

import time
def rep(x, k): return [t.repeat_interleave(k, 0) for t in x] if isinstance(x, list) else x.repeat_interleave(k, 0)
for Bm in (4, 8, 16, 32):
    k = Bm // B
    im, imk, lt, lm, st, ac = rep(images, k), rep(img_masks, k), rep(lang_tokens, k), rep(lang_masks, k), rep(state, k), rep(actions, k)
    ts = timesteps.repeat_interleave(k, 0); nz = noise.repeat_interleave(k, 0)
    torch.cuda.reset_peak_memory_stats(); mod.zero_grad(set_to_none=True)
    for it in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.enable_grad():
            loss = smolvla_fpo_cfm_loss(mod, im, imk, lt, lm, st, ac, ts, nz, action_dim=7, num_steps=H); loss.mean().backward()
        torch.cuda.synchronize(); dt = time.time() - t0
    print(f"micro-batch {Bm:2d} chunks x N=4: forward+backward {dt:.3f} s ({dt / Bm * 1000:.0f} ms/chunk), peak GPU {torch.cuda.max_memory_allocated() // 2**20} MiB")
    mod.zero_grad(set_to_none=True)
