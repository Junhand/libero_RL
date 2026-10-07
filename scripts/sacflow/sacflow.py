#!/usr/bin/env python
"""SAC Flow on smolVLA -- a minimal STANDALONE prototype (single process, one GPU; not integrated into RLinf's Ray/FSDP workers).

actor : smolVLA's action expert (lm_expert + action_* layers, ~99.8M params; VLM + vision tower frozen). The action chunk is the K-step Euler rollout of the
        expert, with fixed-sigma Gaussian noise added to the action at every step (as RLinf's JaxFlowTActor 'fixed noise std' variant). Gradients flow through all K steps.
critic: Q(s,a) = 2 MLP heads (min) on [mean-pooled and last-token hidden state of the frozen VLM prefix] + executed chunk (10x7, normalised). EMA target.
loss  : actor  = -lambda * Q(s, a_theta) + BC(a_theta, a_replay)   (TD3+BC-style normalisation = SAC Flow offline-to-online proximity term, paper Eq. 12)
        critic = TD (gamma over chunk decisions), reward = 1 at success.  The entropy term is dropped (alpha=0): with a fixed noise std the path density does not depend on theta.
data  : online LIBERO-10 Task 0 (the 50 fixed init states), 3 parallel envs, chunk-level transitions, episode-based uint8 replay."""
import os, sys, time, json, copy, argparse, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", ROBOT_PLATFORM="LIBERO", LIBERO_TYPE="standard", HF_HUB_OFFLINE="1")
R = "/workspace/libero_RL"; sys.path.insert(0, f"{R}/RLinf")
C, A = 10, 7          # executed chunk (env steps) and action dim
TASK = "put both the alphabet soup and the tomato sauce in the basket"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True); p.add_argument("--hours", type=float, default=3.2); p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=520); p.add_argument("--iter_eps", type=int, default=3); p.add_argument("--warm_eps", type=int, default=150)
    p.add_argument("--min_eps", type=int, default=12); p.add_argument("--updates", type=int, default=16); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--K", type=int, default=10); p.add_argument("--sigma_roll", type=float, default=0.02); p.add_argument("--sigma_train", type=float, default=0.05)
    p.add_argument("--q_weight", type=float, default=0.06); p.add_argument("--bc_weight", type=float, default=1.0); p.add_argument("--drift_max", type=float, default=0.1)
    p.add_argument("--abort_succ", type=float, default=0.12); p.add_argument("--final_save", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99); p.add_argument("--tau", type=float, default=0.01); 
    p.add_argument("--actor_lr", type=float, default=2e-6); p.add_argument("--critic_lr", type=float, default=3e-4); p.add_argument("--actor_warmup", type=int, default=50)
    p.add_argument("--buf_eps", type=int, default=160); p.add_argument("--save_every", type=int, default=20); p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

class Critic(nn.Module):
    def __init__(s, fdim=1920, adim=C * A, hid=512):
        super().__init__(); s.ln = nn.LayerNorm(fdim)
        s.q = nn.ModuleList([nn.Sequential(nn.Linear(fdim + adim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1)) for _ in range(2)])
    def forward(s, feat, a):
        z = torch.cat([s.ln(feat), a.flatten(1)], -1); return torch.cat([q(z) for q in s.q], -1)      # [B, 2]

class Buffer:     # episode-based replay; transition = (episode, chunk index t), next obs = index t+1
    def __init__(s, max_eps): s.eps = collections.deque(maxlen=max_eps)
    def add(s, ep): s.eps.append(ep)
    def n_trans(s): return sum(e["T"] for e in s.eps)
    def sample(s, B, rng):
        cum = np.cumsum([e["T"] for e in s.eps]); idx = rng.integers(0, cum[-1], B); items = []
        for i in idx:
            k = int(np.searchsorted(cum, i, side="right")); items.append((s.eps[k], int(i - (cum[k - 1] if k > 0 else 0))))
        def obs(off):
            im = np.stack([e["img"][t + off] for e, t in items])
            return dict(main_images=torch.from_numpy(im[:, 0]), wrist_images=torch.from_numpy(im[:, 1]), states=torch.from_numpy(np.stack([e["st"][t + off] for e, t in items])), task_descriptions=[TASK] * B)
        a = torch.from_numpy(np.stack([e["act"][t] for e, t in items]).astype(np.float32)).cuda()
        r = torch.tensor([e["rew"][t] for e, t in items], dtype=torch.float32).cuda(); d = torch.tensor([float(e["term"][t]) for e, t in items], dtype=torch.float32).cuda()
        return obs(0), obs(1), a, r, d

class Agent:
    def __init__(s, args):
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.smolvla import get_model
        from rlinf.models.embodiment.pi0_fast.lerobot_io import build_lerobot_batch_from_env_obs
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        s.build_batch, s.LT, s.LM, s.mask2d = build_lerobot_batch_from_env_obs, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, make_att_2d_masks
        cfg = OmegaConf.load(f"{R}/RLinf/examples/embodiment/config/model/smolvla.yaml"); cfg.model_path = f"{R}/models/smolvla_libero"
        cfg.smolvla.vlm_model_name = f"{R}/models/SmolVLM2-500M-Video-Instruct"
        # fp32 master weights: the default load is a bf16/fp32 mix, and bf16 weights would swallow lr=1e-5 updates
        s.m = get_model(cfg, torch_dtype=torch.float32).cuda().eval(); s.pol = s.m.policy; s.model = s.pol.model; s.a = args; s.K = args.K
        is_expert = lambda n: ("lm_expert" in n) or ("action_" in n and "vlm" not in n)
        s.params = []
        for n, p in s.m.named_parameters():
            p.requires_grad_(is_expert(n))
            if is_expert(n): s.params.append((n, p))
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        lin = s.model.action_out_proj; _f = lin.forward
        def _fp32_out(x, _f=_f):
            with torch.autocast("cuda", enabled=False): return _f(x.float())
        lin.forward = _fp32_out      # velocity output in fp32 even under bf16 autocast (parameter names/keys unchanged)
        s.critic = Critic().cuda(); s.target = copy.deepcopy(s.critic).requires_grad_(False)
        s.copt = torch.optim.AdamW(s.critic.parameters(), args.critic_lr, weight_decay=0.0)
        s.aopt = torch.optim.AdamW([p for _, p in s.params], args.actor_lr, betas=(0.9, 0.95), weight_decay=0.0)
        s.ref = [p.detach().clone() for _, p in s.params]      # frozen initial expert (for the pathwise anchor)
        s.rng = np.random.default_rng(args.seed); s.n_actor_updates = 0
        print(f"actor params {sum(p.numel() for _, p in s.params) / 1e6:.1f}M | critic {sum(p.numel() for p in s.critic.parameters()) / 1e6:.2f}M", flush=True)
    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def prefix(s, env_obs):
        b = s.m._prepare_lerobot_batch(s.build_batch(env_obs)); images, im = s.pol.prepare_images(b); state = s.pol.prepare_state(b)
        pe, pm, pa = s.model.embed_prefix(images, im, b[s.LT], b[s.LM], state=state)
        outs, pkv = s.model.vlm_with_expert.forward(attention_mask=s.mask2d(pm, pa), position_ids=torch.cumsum(pm, 1) - 1, past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        h = outs[0].float(); w = pm.unsqueeze(-1).float()
        return dict(pm=pm, pkv=pkv, feat=torch.cat([(h * w).sum(1) / w.sum(1).clamp_min(1), h[:, -1]], -1))
    def make_noise(s, B, sigma):      # x_1 and the per-step noises, so that the reference and the actor can be run with the SAME noise
        x1 = torch.randn(B, s.model.config.chunk_size, s.model.config.max_action_dim, device="cuda")
        return x1, ([torch.randn_like(x1) for _ in range(s.K)] if sigma > 0 else None)
    def flow(s, pk, sigma, grad, noise=None):      # K-step noisy Euler rollout of the action expert (x: t=1 noise -> t=0 action), differentiable when grad=True
        B = pk["pm"].shape[0]; dt = -1.0 / s.K; x, eps = noise if noise is not None else s.make_noise(B, sigma)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(s.K):
                v = s.model.denoise_step(pk["pm"], pk["pkv"], x, torch.full((B,), 1.0 + k * dt, dtype=torch.float32, device="cuda")); x = x + dt * v
                if eps is not None: x = x + sigma * eps[k]
        return x
    @torch.no_grad()
    def ref_flow(s, pk, sigma, noise):      # same rollout with the INITIAL expert weights (temporarily swapped in)
        cur = [p.detach().clone() for _, p in s.params]
        for (_, p), r in zip(s.params, s.ref): p.copy_(r)
        try: return s.flow(pk, sigma, False, noise)
        finally:
            for (_, p), c in zip(s.params, cur): p.copy_(c)
    @torch.no_grad()
    def act(s, env_obs):      # rollout: returns normalised chunk [B,C,A] and the unnormalised env actions
        pk = s.prefix(env_obs); a = s.flow(pk, s.a.sigma_roll, False)[:, :C, :A]
        return a, s.m.postprocessor(a.float().cpu())
    def update(s, buf, do_actor):
        a = s.a; obs, nobs, ab, r, d = buf.sample(a.batch, s.rng); st = {}
        cur, nxt = s.prefix(obs), s.prefix(nobs)
        with torch.no_grad():
            an = s.flow(nxt, a.sigma_train, False)[:, :C, :A]; y = r + a.gamma * (1 - d) * s.target(nxt["feat"], an).min(-1).values
        q = s.critic(cur["feat"], ab); closs = F.mse_loss(q, y[:, None].expand_as(q)); s.copt.zero_grad(set_to_none=True); closs.backward()
        st["critic_grad"] = float(nn.utils.clip_grad_norm_(s.critic.parameters(), 10.0)); s.copt.step()
        with torch.no_grad():
            for pt, pc in zip(s.target.parameters(), s.critic.parameters()): pt.lerp_(pc, a.tau)
        st.update(closs=float(closs), q_data=float(q.mean()), y=float(y.mean()), r=float(r.mean()))
        if do_actor:
            for p in s.critic.parameters(): p.requires_grad_(False)
            noise = s.make_noise(a.batch, a.sigma_train); aref = s.ref_flow(cur, a.sigma_train, noise)[:, :C, :A]
            ap = s.flow(cur, a.sigma_train, True, noise)[:, :C, :A]; qpi = s.critic(cur["feat"], ap).min(-1).values
            dq = torch.autograd.grad(qpi.sum(), ap, retain_graph=True)[0]; drift = ((ap - aref) ** 2).mean().sqrt().detach()
            guard = float(drift < a.drift_max)      # stop following Q while the policy is further than drift_max (rms, normalised units) from the initial one
            bc = ((ap - aref) ** 2).flatten(1).sum(-1).mean()
            lam = a.q_weight / dq.abs().mean().detach().clamp_min(1e-6)      # Q-gradient normalised to q_weight per element => equilibrium drift ~ q_weight / (2 * bc_weight) (normalised action units)
            aloss = -lam * guard * qpi.mean() + a.bc_weight * bc
            for g in s.aopt.param_groups: g["lr"] = a.actor_lr * min(1.0, (s.n_actor_updates + 1) / a.actor_warmup)
            s.aopt.zero_grad(set_to_none=True); aloss.backward(); st["actor_grad"] = float(nn.utils.clip_grad_norm_([p for _, p in s.params], 1.0)); s.aopt.step(); s.n_actor_updates += 1
            for p in s.critic.parameters(): p.requires_grad_(True)
            st.update(aloss=float(aloss), bc=float(bc), q_pi=float(qpi.mean()), drift=float(drift), guard=guard, dQda=float(dq.abs().mean()), lam=float(lam))
        return st

class Collector:
    def __init__(s, args, agent, buf):
        from omegaconf import OmegaConf
        from rlinf.envs.sim.libero.libero_env import LiberoEnv
        cfg = OmegaConf.load(f"{R}/RLinf/examples/embodiment/config/env/libero_10.yaml")
        cfg.total_num_envs = args.num_envs; cfg.max_episode_steps = args.max_episode_steps; cfg.max_steps_per_rollout_epoch = args.max_episode_steps
        cfg.use_rel_reward = False; cfg.reward_coef = 1.0; cfg.task_id_filter = [0]; cfg.video_cfg.save_video = False; cfg.video_cfg.video_base_dir = f"{R}/tmp/sacflow/video"
        cfg.is_eval = False; cfg.auto_reset = False; cfg.ignore_terminations = False; cfg.seed = args.seed
        s.env = LiberoEnv(cfg, num_envs=args.num_envs, seed_offset=0, total_num_processes=1, worker_info=None); s.agent, s.buf, s.n = agent, buf, args.num_envs
        s.obs, _ = s.env.reset(); s.cur = [s.new_ep() for _ in range(s.n)]; s.done_eps = 0; s.succ = collections.deque(maxlen=30); s.lens = collections.deque(maxlen=30); s.env_steps = 0
        for i in range(s.n): s.push_obs(i)
    @staticmethod
    def new_ep(): return dict(img=[], st=[], act=[], rew=[], term=[])
    def push_obs(s, i):
        e = s.cur[i]; e["img"].append(np.stack([s.obs["main_images"][i].numpy(), s.obs["wrist_images"][i].numpy()])); e["st"].append(s.obs["states"][i].numpy().astype(np.float32))
    def step(s):
        a_norm, a_env = s.agent.act(s.obs); ol, rw, te, tr, _ = s.env.chunk_step(a_env); s.env_steps += C * s.n; s.obs = ol[-1]
        finished = []
        for i in range(s.n):
            e = s.cur[i]; term = bool(te[i].any()); trunc = bool(tr[i].any()); e["act"].append(a_norm[i].float().cpu().numpy().astype(np.float16)); e["rew"].append(1.0 if term else 0.0); e["term"].append(term)
            s.push_obs(i)       # next observation (final one if the episode ends here)
            if term or trunc: finished.append((i, term))
        for i, term in finished:
            e = s.cur[i]; s.buf.add(dict(img=np.stack(e["img"]), st=np.stack(e["st"]), act=np.stack(e["act"]), rew=np.array(e["rew"], np.float32), term=np.array(e["term"]), T=len(e["act"]), succ=term))
            s.succ.append(float(term)); s.lens.append(len(e["act"]) * C); s.done_eps += 1; s.cur[i] = s.new_ep()
        if finished:
            s.obs, _ = s.env.reset(env_idx=[i for i, _ in finished])
            for i, _ in finished: s.push_obs(i)

def main():
    args = get_args(); out = f"{R}/tmp/sacflow/runs/{args.tag}"; os.makedirs(out, exist_ok=True); torch.manual_seed(args.seed)
    run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb; run = wandb.init(project="libero_RL", name=f"sacflow_smolvla_{args.tag}", config=vars(args), dir=out)
        except Exception as ex: print("wandb disabled:", ex, flush=True)
    agent = Agent(args); buf = Buffer(args.buf_eps); col = Collector(args, agent, buf)
    json.dump(vars(args), open(f"{out}/args.json", "w")); t0 = time.time(); it = 0; low = 0; logf = open(f"{out}/log.jsonl", "a")
    print(f"start {time.strftime('%H:%M:%S')} | budget {args.hours} h | GPU peak after load {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    while time.time() - t0 < args.hours * 3600:
        ti = time.time(); n0 = col.done_eps
        while col.done_eps - n0 < args.iter_eps: col.step()
        tc = time.time() - ti; ups = []; tu = time.time()
        if len(buf.eps) >= args.min_eps:
            for _ in range(args.updates): ups.append(agent.update(buf, do_actor=(col.done_eps >= args.warm_eps)))
        tu = time.time() - tu; mean = lambda k: float(np.mean([u[k] for u in ups if k in u])) if any(k in u for u in ups) else None
        rec = dict(it=it, hours=round((time.time() - t0) / 3600, 3), eps=col.done_eps, trans=buf.n_trans(), env_steps=col.env_steps, succ30=(float(np.mean(col.succ)) if col.succ else None), len30=(float(np.mean(col.lens)) if col.lens else None),
                   updates=len(ups), actor_updates=agent.n_actor_updates, t_collect=round(tc, 1), t_update=round(tu, 1), gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                   **{k: mean(k) for k in ("closs", "q_data", "y", "r", "aloss", "bc", "q_pi", "drift", "guard", "dQda", "lam", "critic_grad", "actor_grad")})
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)
        if run is not None: run.log({k: v for k, v in rec.items() if v is not None}, step=it)
        if it % args.save_every == 0 and agent.n_actor_updates > 0: torch.save({n: p.detach().cpu() for n, p in agent.params}, f"{out}/expert_it{it:04d}.pt")
        low = low + 1 if (agent.n_actor_updates > 0 and len(col.succ) >= 30 and np.mean(col.succ) < args.abort_succ) else 0
        if low >= 8: open(f"{out}/ABORT", "w").write(f"collapse: succ30<{args.abort_succ} for 8 iterations at it {it}"); print("ABORT: success collapsed", flush=True); break
        it += 1
    torch.save({n: p.detach().cpu() for n, p in agent.params}, f"{out}/expert_final.pt")
    if args.final_save: torch.save(agent.m.state_dict(), f"{out}/full_weights_final.pt")
    open(f"{out}/DONE", "w").write(time.strftime("%F %T")); print("DONE", flush=True)
    if run is not None: run.finish()

if __name__ == "__main__":
    main()
