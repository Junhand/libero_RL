"""Does the critic of the 2-phase run (F, step 750) distinguish actions, per phase?
Replay rows are in insertion order: rows [0, N_D) come from run D (phase 1 only), the rest from the 2-phase run.
Phase labels for the rest: per episode the first segment is phase 1; a terminal with reward>0 in phase 1 starts phase 2."""
import sys, re, glob, os, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.mlp_policy import get_model as get_mlp
LOG = "/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_101403"
CK = f"{LOG}/rlt-phase2-utd5-2phase/checkpoints/global_step_750/actor"
R = f"{CK}/sac_components/replay_buffer/rank_0"
t = open(f"{LOG}/run.log", errors="ignore").read(); N = r"(-?[0-9.]+(?:e[+-]?[0-9]+)?)"
gm = [float(x) for x in re.findall(r"rlt/global_min_replay_size[ =]+" + N, t)]; tc = [float(x) for x in re.findall(r"replay/transition_count[ =]+" + N, t)]
N_D = int(gm[0] - tc[0]); print("rows from D:", N_D)
files = {}
for p in glob.glob(f"{R}/trajectory_*.pt"):
    files[int(os.path.basename(p).split("_")[1])] = p
n = len(files); print("rows total:", n)
def load(i):
    d = torch.load(files[i], map_location="cpu", weights_only=False); o = d["curr_obs"]
    return (o["z_rl"].reshape(-1), o["proprio"].reshape(-1), o["ref_chunk"].reshape(-1), d["actions"].reshape(-1), d["rewards"].reshape(-1), d["dones"].reshape(-1).any().item())
with ThreadPoolExecutor(32) as ex: rows = list(ex.map(load, range(n)))
Z = torch.stack([r[0] for r in rows]); P = torch.stack([r[1] for r in rows]); REF = torch.stack([r[2] for r in rows]); A = torch.stack([r[3] for r in rows])
RW = torch.stack([r[4] for r in rows]); DONE = np.array([r[5] for r in rows])
rew = RW.sum(1).numpy()
# phase labels
phase = np.ones(n, np.int8); ph = 1
for i in range(N_D, n):
    phase[i] = ph
    if DONE[i]: ph = 2 if (ph == 1 and rew[i] > 0) else 1
print("phase counts (new rows): phase1 %d, phase2 %d" % ((phase[N_D:] == 1).sum(), (phase[N_D:] == 2).sum()))
print("terminal rows with reward>0: phase1 %d, phase2 %d" % ((DONE & (rew > 0) & (phase == 1))[N_D:].sum(), (DONE & (rew > 0) & (phase == 2))[N_D:].sum()))
cfg = OmegaConf.create(dict(model_type="rlt_mlp_policy", z_dim=960, proprio_dim=8, action_dim=7, num_action_chunks=10, ref_num_action_chunks=10, add_q_head=True, q_head_type="default", fixed_std=0.02))
net = get_mlp(cfg, torch.float32); print(net.load_state_dict(torch.load(f"{CK}/model_state_dict/full_weights.pt", map_location="cpu"))); net = net.cuda().eval()
def Qmin(z, p, a):
    with torch.no_grad():
        q = net.sac_q_forward({"z_rl": z, "proprio": p}, a); q = torch.cat(list(q), -1) if isinstance(q, (list, tuple)) else q
        return q.min(-1).values
rng = torch.Generator().manual_seed(0)
for name, mask in (("phase 1 rows (new run)", (phase == 1) & (np.arange(n) >= N_D)), ("phase 2 rows (new run)", phase == 2)):
    idx = torch.from_numpy(np.where(mask)[0]); idx = idx[torch.randperm(len(idx), generator=rng)[:4000]]
    z, p, a, ref = Z[idx].cuda(), P[idx].cuda(), A[idx].cuda(), REF[idx].cuda().reshape(len(idx), -1)
    with torch.no_grad():
        act, _, _ = net.sac_forward({"z_rl": z, "proprio": p, "ref_chunk": ref.reshape(-1, 10, 7)}, deterministic=True)
    q_exec = Qmin(z, p, a); q_ref = Qmin(z, p, ref.clamp(-1, 1)); q_act = Qmin(z, p, act)
    noise = (a + 0.1 * torch.randn_like(a)).clamp(-1, 1); q_noise = Qmin(z, p, noise); q_rand = Qmin(z, p, torch.rand_like(a) * 2 - 1)
    a_req = a.clone().requires_grad_(True); q = net.sac_q_forward({"z_rl": z, "proprio": p}, a_req); q = torch.cat(list(q), -1) if isinstance(q, (list, tuple)) else q; q.min(-1).values.sum().backward(); g = a_req.grad.reshape(len(idx), -1).norm(dim=1)
    y = torch.from_numpy(rew[idx.numpy()]).cuda()
    print(f"\n== {name} (n={len(idx)})")
    print("  Q over states: mean %.3f, std %.3f" % (q_exec.mean(), q_exec.std()))
    print("  Q(s,a_exec)-Q(s,ref):   mean %+.4f, std %.4f" % ((q_exec - q_ref).mean(), (q_exec - q_ref).std()))
    print("  Q(s,actor)-Q(s,ref):    mean %+.4f, std %.4f" % ((q_act - q_ref).mean(), (q_act - q_ref).std()))
    print("  Q(s,a+0.1 noise)-Q(s,a): mean %+.4f, std %.4f | Q(s,random)-Q(s,a): mean %+.4f" % ((q_noise - q_exec).mean(), (q_noise - q_exec).std(), (q_rand - q_exec).mean()))
    print("  |dQ/da| (L2 over 70 dims): median %.4f" % g.median())
    print("  action effect / state effect  (std of Q(s,actor)-Q(s,ref)) / (std of Q over states): %.3f" % ((q_act - q_ref).std() / q_exec.std()))
    pos = y > 0; print("  rows with reward>0: n=%d, Q mean %.3f | rows with reward=0: Q mean %.3f" % (pos.sum(), q_exec[pos].mean() if pos.any() else float('nan'), q_exec[~pos].mean()))

# ---- calibration: realized discounted return vs the critic's prediction, per phase ----
gamma = 0.99
disc = torch.tensor([gamma ** k for k in range(10)])
rt = (RW * disc).sum(1).numpy()                     # discounted chunk reward, as in forward_critic
ret = np.zeros(n); nxt = 0.0
for i in range(n - 1, -1, -1):
    ret[i] = rt[i] + (0.0 if DONE[i] else gamma ** 10 * nxt) if True else 0
    nxt = ret[i]
    if DONE[i - 1] if i > 0 else False: nxt = 0.0
Qe = np.zeros(n); Qr = np.zeros(n)
for s in range(0, n, 4096):
    sl = slice(s, s + 4096)
    z, p, a, ref = Z[sl].cuda(), P[sl].cuda(), A[sl].cuda(), REF[sl].cuda()
    Qe[sl] = Qmin(z, p, a).cpu().numpy(); Qr[sl] = Qmin(z, p, ref.clamp(-1, 1)).cpu().numpy()
print("\n==== calibration of the critic against realized (discounted) returns, rows of the new run ====")
for name, mask in (("phase 1", (phase == 1) & (np.arange(n) >= N_D)), ("phase 2", phase == 2)):
    y = ret[mask]; q = Qe[mask]; d = (Qe - Qr)[mask]
    X = np.c_[np.ones(mask.sum()), q, d]; b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b
    b1, *_ = np.linalg.lstsq(X[:, :2], y, rcond=None)
    print(f"{name}: n={mask.sum()} | mean realized return {y.mean():.3f} vs mean Q(s,a_exec) {q.mean():.3f} (bias {q.mean()-y.mean():+.3f}) | corr(Q, realized) {np.corrcoef(q,y)[0,1]:.3f}")
    print(f"   regression realized ~ a + b*Q(s,a_exec): slope b = {b1[1]:.2f} (1 = calibrated)")
    print(f"   adding the predicted action advantage d=Q(s,a_exec)-Q(s,ref): coefficient on d = {b[2]:+.2f} (1 = the critic's action effects are real, 0 = no information) | mean d = {d.mean():+.3f}")
