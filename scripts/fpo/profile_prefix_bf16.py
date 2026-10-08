"""Where does a co-training update spend its time? (micro-batch 10 PPO forward/backward, SFT data loading, SFT forward/backward, optimizer step; TF32 off/on)"""
import sys, time, torch, numpy as np
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.data.datasets.smolvla_sft import build_smolvla_demo_dataloader
R = "/workspace/libero_RL"
cfg = OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
    smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=None, chunk_size=50, advantage_prompt=False,
    sde=dict(enabled=True, noise_level=0.5, num_steps=5, value_hidden_dims=[512, 256]),
    rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"})))
m = get_model(cfg).cuda().train()
full = OmegaConf.create(dict(actor=dict(model=dict(smolvla=dict(chunk_size=50)), micro_batch_size=10, sft_batch_size=8, sft_num_workers=0)))
dl = build_smolvla_demo_dataloader(full, f"{R}/data/smolvla_recap/v30/libero10_task0_sft_target3", seed=0)
def sync(): torch.cuda.synchronize()
def timeit(fn, n=8, warm=2):
    for _ in range(warm): fn()
    sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t) / n

import contextlib
def ac(on): return torch.autocast("cuda", dtype=torch.bfloat16) if on else contextlib.nullcontext()
it = iter(dl); batch = next(it)
ds = dl.dataset; idx = np.linspace(0, len(ds) - 1, 10).astype(int); items = [ds[int(i)] for i in idx]
env_obs = dict(main_images=torch.stack([(x["observation.images.image"].permute(1, 2, 0) * 255).byte() for x in items]),
               wrist_images=torch.stack([(x["observation.images.image2"].permute(1, 2, 0) * 255).byte() for x in items]),
               states=torch.stack([x["observation.state"] for x in items]).float(), task_descriptions=[items[0]["task"]] * 10)
with torch.no_grad(): _, res = m.predict_action_batch(env_obs, mode="train")
fi = res["forward_inputs"]
def sync(): torch.cuda.synchronize()
def timeit(fn, n=8, warm=2):
    for _ in range(warm): fn()
    sync(); t = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t) / n
def ppo(on):
    def f():
        m.zero_grad(set_to_none=True)
        with ac(on): o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
        (o["logprobs"].float().sum() + o["values"].float().mean()).backward()
    return f
def sft(on):
    def f():
        m.zero_grad(set_to_none=True)
        with ac(on): l = m(data=batch, forward_type=ForwardType.SFT)["loss"]
        l.float().backward()
    return f


from rlinf.models.embodiment.smolvla import sde
from rlinf.models.embodiment.smolvla import smolvla_sde_model as sm
import transformers
orig_prefix = sde.prefix_cache
def prefix_bf16(model, images, img_masks, lang_tokens, lang_masks, state):
    """Frozen VLM prefix under bf16 autocast, KV cache cast back to fp32 for the (fp32) trainable action expert."""
    from transformers import DynamicCache
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, pm, pkv = orig_prefix(model, images, img_masks, lang_tokens, lang_masks, state)
    new = DynamicCache()
    for i, layer in enumerate(pkv.layers): new.update(layer.keys.float(), layer.values.float(), i)
    return feats.float(), pm, new
def run(mode):
    sde.prefix_cache = prefix_bf16 if mode == "prefix_bf16" else orig_prefix
    sm.sde.prefix_cache = sde.prefix_cache
def logp(mode):
    run(mode)
    with torch.no_grad(): return m(forward_inputs=fi, compute_logprobs=True, compute_values=True)["logprobs"].float().sum(-1)
def ppo_time(mode):
    run(mode)
    def f():
        m.zero_grad(set_to_none=True); o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
        (o["logprobs"].float().sum() + o["values"].float().mean()).backward()
    return timeit(f) * 1000
def sft_time(mode):
    run(mode)
    def f():
        m.zero_grad(set_to_none=True); m(data=batch, forward_type=ForwardType.SFT)["loss"].backward()
    return timeit(f) * 1000
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
print("PPO mb10 fp32: %.0f ms | prefix in bf16 (expert fp32): %.0f ms" % (ppo_time("fp32"), ppo_time("prefix_bf16")))
m.zero_grad(set_to_none=True); run("fp32")
o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True); (o["logprobs"].float().sum()).backward()
params = [p for p in m.parameters() if p.requires_grad and p.grad is not None]; grads = [p.grad.clone() for p in params]
b32, bp = logp("fp32"), logp("prefix_bf16")
print("same weights: fp32 vs prefix_bf16 logprob mean |diff| %.4f (fp32 logprob scale %.1f)" % ((b32 - bp).abs().mean().item(), b32.abs().mean().item()))
gen = torch.Generator(device="cuda").manual_seed(0)
for k, step in (("0.1 Adam-sized sign step", 5e-7), ("random +-5e-6", None)):
    with torch.no_grad():
        saved = [p.clone() for p in params]
        for p, g in zip(params, grads):
            if step is not None: p.sub_(step * g.sign())
            else: p.add_(5e-6 * torch.sign(torch.randn(p.shape, device=p.device, generator=gen)))
    r32 = logp("fp32") - b32; rp = logp("prefix_bf16") - bp
    print(f"{k}: fp32 log-ratio mean |.| {r32.abs().mean().item():.4f} | error of prefix_bf16 log-ratio vs fp32: {(rp - r32).abs().mean().item():.5f}")
    with torch.no_grad():
        for p, s0 in zip(params, saved): p.copy_(s0)
