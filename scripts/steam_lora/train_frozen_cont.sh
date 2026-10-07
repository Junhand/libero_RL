#!/usr/bin/env bash
# Continuation of the both-frozen critic training from the step-1000 head weights (optimizer/scheduler restart: lr 1e-4 -> 1e-6 over 1000 steps, new data seed).
cd /workspace/libero_RL/tmp/steam_lora
SAVE=500 ./run_variant.sh steam_frozen2000b 1000 actor.model.freeze_vision_encoder=true actor.model.freeze_language_model=true \
  actor.optim.lr=1e-4 actor.optim.value_lr=1e-4 actor.optim.lr_warmup_steps=20 data.seed=1 +actor.model.model_path=/workspace/libero_RL/tmp/rlinf_steam_dev/logs/steam_sft/steam_value_model_sft-20261006-17:27:55/steam_frozen2000/checkpoints/global_step_1000/actor
