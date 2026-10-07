"""FPO S1 (model): rollout outputs vs default_forward recompute, gradients, trainable parameters."""
import sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
R = "/workspace/libero_RL"; D = f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
cfg = OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
    smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=None, chunk_size=50, advantage_prompt=False,
    fpo=dict(enabled=True, n_action_samples=4, value_hidden_dims=[512, 256]),
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
actions, result = m.predict_action_batch(env_obs, mode="train")
fi = result["forward_inputs"]
print("actions", tuple(actions.shape), actions.dtype, "| prev_logprobs", tuple(result["prev_logprobs"].shape), result["prev_logprobs"].dtype, "| prev_values", tuple(result["prev_values"].shape))
print("forward_inputs:", {k: (tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in fi.items()})
# ---- recompute with default_forward (as the actor does) and compare
out = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
dl = (out["logprobs"].cpu() - result["prev_logprobs"]).abs().max().item(); dv = (out["values"].cpu().reshape(-1) - result["prev_values"].reshape(-1)).abs().max().item()
print(f"recompute vs rollout: max |logprobs diff| {dl:.2e}, max |values diff| {dv:.2e}; logprobs mean {out['logprobs'].mean().item():.4f}")
print("S1-model T1", "PASS" if dl < 1e-4 and dv < 1e-4 else "FAIL")
# ---- gradients through default_forward
m.train(); m.zero_grad(set_to_none=True)
out = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
(out["logprobs"].mean() + out["values"].mean()).backward()
g = {}
for n, p in m.named_parameters():
    if p.requires_grad: g.setdefault(".".join(n.split(".")[:3]) if "policy" in n else n.split(".")[0], []).append(p.grad is not None and p.grad.abs().sum().item() > 0)
print("trainable tensors with non-zero grad (ok / total):", {k: (sum(v), len(v)) for k, v in g.items()})
nograd = sum(1 for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None)
print("frozen params with grad:", nograd)
# ---- ratio at zero update is exactly 1; second call is deterministic
out2 = m(forward_inputs=fi, compute_logprobs=True, compute_values=True)
print("deterministic:", torch.equal(out["logprobs"].detach(), out2["logprobs"].detach()))
print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
