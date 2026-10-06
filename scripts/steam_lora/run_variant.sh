#!/usr/bin/env bash
# STEAM critic training in a variant (full / freeze / LoRA). Usage: ./run_variant.sh <name> <steps> [extra hydra overrides...]
# Same data and settings as tmp/steam/run_expF.sh (sft_target3 x4 + train_v2 rollouts, success episodes only, N=32, k=32, 1 member, micro 8 / global 64),
# but run from the development copy of RLinf (LoRA support) and from scratch.
set -uo pipefail
NAME=${1:?name}; STEPS=${2:?steps}; shift 2
export RAY_memory_usage_threshold=0.98
DEV=/workspace/libero_RL/tmp/rlinf_steam_dev
cd $DEV && source /workspace/libero_RL/RLinf/.venv/bin/activate
export TMPDIR=/workspace/libero_RL/tmp/sl; mkdir -p $TMPDIR
export HF_HUB_OFFLINE=1
D=/workspace/libero_RL/data/smolvla_recap; M=/workspace/libero_RL/models
PATHS="{dataset_path:$D/libero10_task0_sft_target3,type:sft,weight:4.0}"
for ds in $(ls -d $D/libero10_task0_train_v2_r*_m* | sort); do PATHS="$PATHS,{dataset_path:$ds,type:rollout,weight:1.0}"; done
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft \
  "data.train_data_paths=[$PATHS]" \
  "data.camera_keys=[image,wrist_image]" data.k=32 +data.length_scale_enabled=true data.only_success=true data.seed=0 \
  actor.seed=0 actor.model.num_bins=32 actor.model.ensemble_size=1 actor.model.ensemble_head_seed_base=0 \
  actor.model.vision_repo_id=$M/siglip-so400m-patch14-384 \
  actor.model.language_repo_id=$M/gemma-3-270m actor.model.tokenizer_path=$M/gemma-3-270m \
  actor.micro_batch_size=8 actor.global_batch_size=64 data.train_num_workers=4 data.prefetch_factor=2 \
  runner.max_steps=$STEPS actor.optim.total_training_steps=$STEPS actor.optim.lr_warmup_steps=50 \
  runner.save_interval=${SAVE:-100000} runner.logger.logger_backends=[tensorboard] runner.logger.experiment_name=$NAME "$@"
echo "exit=$?" > /workspace/libero_RL/tmp/steam_lora/$NAME.exit
