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

def logp(mode):
    torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32"); torch.backends.cudnn.allow_tf32 = (mode == "tf32")
    with torch.no_grad(), ac(mode == "bf16"): return m(forward_inputs=fi, compute_logprobs=True, compute_values=True)["logprobs"].float().sum(-1)
torch.backends.cuda.matmul.allow_tf32 = False
m.zero_grad(set_to_none=True)
o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True); (o["logprobs"].float().sum()).backward()
params = [p for p in m.parameters() if p.requires_grad and p.grad is not None]
grads = [p.grad.clone() for p in params]
base = {k: logp(k) for k in ("fp32", "tf32", "bf16")}
gen = torch.Generator(device="cuda").manual_seed(0)
for k, step in (("1 Adam-sized sign step", 5e-6), ("0.1 step", 5e-7), ("random +-5e-6", None)):
    with torch.no_grad():
        saved = [p.clone() for p in params]
        for p, g in zip(params, grads):
            if step is not None: p.sub_(step * g.sign())
            else: p.add_(5e-6 * torch.sign(torch.randn(p.shape, device=p.device, generator=gen)))
    r = {kk: logp(kk) - base[kk] for kk in base}
    print(f"{k}: fp32 log-ratio mean |.| {r['fp32'].abs().mean().item():.4f} | error vs fp32: TF32 {(r['tf32']-r['fp32']).abs().mean().item():.4f}, bf16 {(r['bf16']-r['fp32']).abs().mean().item():.4f}")
    with torch.no_grad():
        for p, s0 in zip(params, saved): p.copy_(s0)
