#!/bin/bash
# run_cfg_env.sh (per-environment DEV/TAG/CONFIG) with Weights & Biases enabled: metrics + one success and one failure rollout video per update.
# WANDB_API_KEY is read from ~/.bashrc of the machine it runs on, without being printed.
#   DEV=<dev copy> TAG=<tag> CONFIG=<config> EPOCHS=<updates> EXP=<wandb run name> ./run_cfg_env_wandb.sh [hydra overrides...]
eval "$(grep -E '^(export )?WANDB_API_KEY=' ~/.bashrc)"; export WANDB_API_KEY
[ -n "$WANDB_API_KEY" ] || { echo "WANDB_API_KEY not found in ~/.bashrc" >&2; exit 1; }
exec "$(dirname "$0")/run_cfg_env.sh" 'runner.logger.logger_backends=[tensorboard,wandb]' runner.logger.project_name=${WANDB_PROJECT:-smolvla-libero-ppo} \
  runner.logger.experiment_name=${EXP:-${TAG:-run}_$(date +%m%d_%H%M)} runner.rollout_video.enabled=True "$@"
