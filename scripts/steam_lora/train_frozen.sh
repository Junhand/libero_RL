#!/usr/bin/env bash
# STEAM critic training with BOTH the vision encoder and the language model frozen (only the projectors / fusion norm / head train).
# Usage: [SAVE=1000] [LR=2e-4] ./train_frozen.sh <name> <steps>
NAME=${1:?name}; STEPS=${2:?steps}
cd /workspace/libero_RL/tmp/steam_lora
SAVE=${SAVE:-1000} ./run_variant.sh $NAME $STEPS actor.model.freeze_vision_encoder=true actor.model.freeze_language_model=true \
  actor.optim.lr=${LR:-2e-4} actor.optim.value_lr=${LR:-2e-4}
