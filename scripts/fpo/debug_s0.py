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
    mod.vlm_with_expert.lm_expert.to(torch.float32)
import collections
print("param dtype histogram:", dict(collections.Counter(str(p.dtype) for p in mod.parameters())), "| buffers:", dict(collections.Counter(str(b.dtype) for b in mod.buffers())))
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

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
print("param dtypes:", {k: str(next(v.parameters()).dtype) for k, v in (("vlm", mod.vlm_with_expert.vlm), ("lm_expert", mod.vlm_with_expert.lm_expert), ("action_out_proj", mod.action_out_proj), ("state_proj", mod.state_proj))})
def v_joint(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
    time_expanded = time[:, None, None]; x_t = time_expanded * noise + (1 - time_expanded) * actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = mod.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad_masks, suffix_att_masks = mod.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1); att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d = make_att_2d_masks(pad_masks, att_masks); pos = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = mod.vlm_with_expert.forward(attention_mask=att_2d, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
    return mod.action_out_proj(suffix_out[:, -C:].to(torch.float32)), x_t
def v_cached(images, img_masks, lang_tokens, lang_masks, state, x_t, time):
    prefix_embs, prefix_pad_masks, prefix_att_masks = mod.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks); pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, pkv = mod.vlm_with_expert.forward(attention_mask=att, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
    return mod.denoise_step(prefix_pad_masks, pkv, x_t, time)
with torch.no_grad():
    t = timesteps[:, 0]; e = noise[:, 0]
    vj, xt = v_joint(images, img_masks, lang_tokens, lang_masks, state, actions, e, t)
    vc = v_cached(images, img_masks, lang_tokens, lang_masks, state, xt, t)
    d = (vj - vc).abs(); print(f"v joint vs cached: max abs {d.max().item():.3e}, mean abs {d.mean().item():.3e}, |v| mean {vj.abs().mean().item():.3e}")
    print("per-sample max abs diff:", [round(d[i].max().item(), 4) for i in range(B)])
    # batch-size dependence of the joint path itself (sample 0 alone vs in the batch of 4)
    vj1, _ = v_joint(images[:1] if not isinstance(images, list) else [x[:1] for x in images], img_masks[:1] if not isinstance(img_masks, list) else [x[:1] for x in img_masks], lang_tokens[:1], lang_masks[:1], state[:1], actions[:1], e[:1], t[:1])
    print(f"joint path, sample 0 alone vs in batch: max abs {(vj1[0] - vj[0]).abs().max().item():.3e}")
    vc1 = v_cached([x[:1] for x in images], [x[:1] for x in img_masks], lang_tokens[:1], lang_masks[:1], state[:1], xt[:1], t[:1])
    print(f"cached path, sample 0 alone vs in batch: max abs {(vc1[0] - vc[0]).abs().max().item():.3e}")
    print("steps/dims where |joint-cached| is largest (sample 0):", (d[0].mean(-1)).argsort(descending=True)[:5].tolist(), "dims mean:", [round(x, 4) for x in d[0].mean(0)[:9].tolist()])
