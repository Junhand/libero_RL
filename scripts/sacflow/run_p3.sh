#!/usr/bin/env bash
# SAC Flow(smolVLA) prototype run p3: pathwise anchor to the frozen initial expert (same noise), critic-only warm-up of 150 episodes,
# Q-gradient normalised (equilibrium drift ~ q_weight/(2*bc_weight) = 0.03), drift guard 0.1, auto-abort when the success rate collapses. The python process is NOT exec'd, so its exit status is recorded.
cd /workspace/libero_RL/tmp/sacflow
source <(grep '^export WANDB_API_KEY=' ~/.bashrc 2>/dev/null)
ulimit -c 0
PYTHONUNBUFFERED=1 /workspace/libero_RL/RLinf/.venv-smolvla/bin/python -X faulthandler sacflow.py --tag p3 --hours 3.6 --num_envs 3 --max_episode_steps 520 \
  --iter_eps 3 --warm_eps 150 --min_eps 12 --updates 16 --batch 16 --K 10 --sigma_roll 0.02 --sigma_train 0.05 \
  --gamma 0.99 --tau 0.01 --q_weight 0.06 --bc_weight 1.0 --drift_max 0.1 --abort_succ 0.12 \
  --actor_lr 2e-6 --critic_lr 3e-4 --actor_warmup 50 --buf_eps 50 --save_every 30 --seed 0
echo "exit=$? at $(date +%T)" > runs/p3/exit
