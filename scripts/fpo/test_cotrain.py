"""Co-training pieces: demo dataset -> SFT loss of the Flow-SDE wrapper -> gradients; PPO + SFT loss in one backward."""
import sys, time, torch
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
t0 = time.time(); dl = build_smolvla_demo_dataloader(full, f"{R}/data/smolvla_recap/v30/libero10_task0_sft_target3", seed=0)
print("dataset frames:", len(dl.dataset), "| batches/epoch:", len(dl), f"| build {time.time()-t0:.1f}s")
t0 = time.time(); batch = next(iter(dl)); print(f"first batch {time.time()-t0:.1f}s:", {k: (tuple(v.shape) if torch.is_tensor(v) else (type(v).__name__, len(v))) for k, v in batch.items()})
out = m(data=batch, forward_type=ForwardType.SFT)
print("sft loss:", out["loss"].item())
m.zero_grad(set_to_none=True); out["loss"].backward()
ok = sum(1 for n, p in m.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
tot = sum(1 for n, p in m.named_parameters() if p.requires_grad)
print("trainable tensors with grad from SFT loss:", ok, "/", tot, "| frozen with grad:", sum(1 for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None))
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], float("inf"))
print("grad norm of the SFT loss (batch 8, beta=1):", round(gn.item(), 3))
# a few more batches for a stable estimate
g = []
it = iter(dl)
for _ in range(5):
    b = next(it); m.zero_grad(set_to_none=True); l = m(data=b, forward_type=ForwardType.SFT)["loss"]; l.backward()
    g.append(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], float("inf")).item()); 
print("SFT grad norms over 5 batches:", [round(x, 3) for x in g], "| losses last:", round(l.item(), 4))
