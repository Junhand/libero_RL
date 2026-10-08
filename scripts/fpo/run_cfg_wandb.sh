#!/bin/bash
# run_cfg.sh with Weights & Biases enabled (metrics + one success and one failure rollout video per update).
# The API key is read from ~/.bashrc (WANDB_API_KEY) without being printed. Usage: EXP=<run name> CONFIG=... EPOCHS=... ./run_cfg_wandb.sh [hydra overrides...]
eval "$(grep -E '^(export )?WANDB_API_KEY=' ~/.bashrc)"; export WANDB_API_KEY
[ -n "$WANDB_API_KEY" ] || { echo "WANDB_API_KEY not found in ~/.bashrc" >&2; exit 1; }
export WANDB_ENTITY_OVERRIDE=""   # default entity of the key
exec "$(dirname "$0")/run_cfg.sh" 'runner.logger.logger_backends=[tensorboard,wandb]' runner.logger.project_name=${WANDB_PROJECT:-smolvla-libero-ppo} \
  runner.logger.experiment_name=${EXP:-ppo_sde_$(date +%m%d_%H%M)} runner.rollout_video.enabled=True "$@"
