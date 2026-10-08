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
print("speed (ms): " )
for name, tf32, bf in (("fp32", False, False), ("TF32", True, False), ("bf16 autocast", True, True)):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    print(f"  {name:14s} PPO mb10 {timeit(ppo(bf))*1000:.0f} | SFT b8 {timeit(sft(bf))*1000:.0f}")
# ---- noise of the per-chunk log-ratio when the weights move by a few Adam-sized steps
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
def logp(on):
    with torch.no_grad(), ac(on): return m(forward_inputs=fi, compute_logprobs=True, compute_values=True)["logprobs"].float().sum(-1)
base32, base16 = logp(False), logp(True)
print("same weights, bf16 vs fp32 logprob (per chunk, sum of 70): mean |diff|", (base32 - base16).abs().mean().item())
ppo(False)()  # gradients
params = [p for p in m.parameters() if p.requires_grad and p.grad is not None]
for k, step in (("1 step", 5e-6), ("10 steps", 5e-5)):
    with torch.no_grad():
        saved = [p.clone() for p in params]
        for p in params: p.sub_(step * p.grad.sign())
    r32 = logp(False) - base32; r16 = logp(True) - base16
    print(f"  weights moved by {k} (lr 5e-6 x sign): fp32 log-ratio std {r32.std().item():.3f} (mean |.| {r32.abs().mean().item():.3f}) | bf16-vs-fp32 error of the log-ratio: mean |err| {(r16 - r32).abs().mean().item():.3f}")
    with torch.no_grad():
        for p, s0 in zip(params, saved): p.copy_(s0)
print("(PPO clip radius is 0.2 in the log-ratio)")
