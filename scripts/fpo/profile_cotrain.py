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
it = iter(dl)
def load():
    global it
    try: return next(it)
    except StopIteration: it = iter(dl); return next(it)
batches = [load() for _ in range(4)]  # decoded batches for the GPU timings
t_data = timeit(load, n=12, warm=1)
# PPO micro-batch of 10: stored rollout inputs from demo frames
ds = dl.dataset; idx = np.linspace(0, len(ds) - 1, 10).astype(int); items = [ds[int(i)] for i in idx]
env_obs = dict(main_images=torch.stack([(x["observation.images.image"].permute(1, 2, 0) * 255).byte() for x in items]),
               wrist_images=torch.stack([(x["observation.images.image2"].permute(1, 2, 0) * 255).byte() for x in items]),
               states=torch.stack([x["observation.state"] for x in items]).float(), task_descriptions=[items[0]["task"]] * 10)
with torch.no_grad(): _, res = m.predict_action_batch(env_obs, mode="train")
fi = res["forward_inputs"]
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-9)
def ppo():
    m.zero_grad(set_to_none=True); o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
    (o["logprobs"].sum() + o["values"].mean()).backward()
def sft():
    m.zero_grad(set_to_none=True); m(data=batches[0], forward_type=ForwardType.SFT)["loss"].backward()
def both():
    m.zero_grad(set_to_none=True)
    o = m(forward_inputs=fi, compute_logprobs=True, compute_values=True); l = o["logprobs"].sum() + o["values"].mean()
    l = l + m(data=batches[1], forward_type=ForwardType.SFT)["loss"]; l.backward()
def step(): opt.step()
def rollout_chunk():
    with torch.no_grad(): m.predict_action_batch(env_obs, mode="train")
print(f"SFT data loading (batch 8, 0 workers): {t_data*1000:.0f} ms / batch")
for tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    r = {k: timeit(f) * 1000 for k, f in dict(ppo_fwd_bwd_mb10=ppo, sft_fwd_bwd_b8=sft, ppo_plus_sft=both, optimizer_step=step, rollout_chunk_10envs=rollout_chunk).items()}
    mb = 288  # micro-batches per update: 720 entries / 10 x 4 epochs
    est_off = mb * (r["ppo_fwd_bwd_mb10"] + t_data * 1000 + r["sft_fwd_bwd_b8"]) / 1000 + 24 * r["optimizer_step"] / 1000
    est_no = mb * r["ppo_fwd_bwd_mb10"] / 1000 + 24 * r["optimizer_step"] / 1000
    print(f"TF32={tf32}: " + ", ".join(f"{k} {v:.0f} ms" for k, v in r.items()) + f" | est. training per update: PPO only {est_no:.0f} s, with SFT {est_off:.0f} s")
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
