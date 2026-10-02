#!/usr/bin/env bash
# Step2 value model SFT on the recollected (success-terminated) data. Usage: MB=<micro batch> bash scripts/run_step2_v2.sh MAX_STEPS [extra hydra args]
# env: MB (default 8; 24GB GPU needs 8 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True), SAVE/VAL (default 560), EVS (eval samples per set)
set -euo pipefail
MAX=${1:-20}; shift || true
cd /workspace/libero_RL/RLinf
source .venv/bin/activate
D=/workspace/libero_RL/data/smolvla_recap
MODELS=/workspace/libero_RL/models
T="{dataset_path:$D/libero10_task0_sft,type:sft,weight:1.0,robot_type:libero,model_type:pi05}"
for r in 1 2 3 4; do for m in 0 1 2; do
  [ -d $D/libero10_task0_train_v2_r${r}_m$m ] && T="$T,{dataset_path:$D/libero10_task0_train_v2_r${r}_m$m,type:rollout,weight:1.0,robot_type:libero,model_type:pi05}"
done; done
E=""; for m in 0 1 2; do E="$E{dataset_path:$D/libero10_task0_eval_v2_m$m,max_samples:${EVS:-3000},robot_type:libero,model_type:pi05},"; done
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft \
  "data.train_data_paths=[$T]" "data.eval_data_paths=[${E%,}]" data.tag=fail300 \
  actor.model.siglip_path=$MODELS/siglip2-so400m-patch14-224 \
  actor.model.gemma3_path=$MODELS/gemma-3-270m \
  actor.model.tokenizer_path=$MODELS/gemma-3-270m \
  actor.micro_batch_size=${MB:-8} actor.global_batch_size=256 \
  runner.max_steps=$MAX runner.save_interval=${SAVE:-560} runner.val_check_interval=${VAL:-560} \
  "runner.logger.logger_backends=[tensorboard]" \
  runner.logger.project_name=libero_RL runner.logger.experiment_name=step2_value_sft_v2 "$@"
