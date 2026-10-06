#!/bin/bash
# RLT Stage 2 smoke run using the development copy of RLinf (tmp/rlinf_dsrl).
ROOT=/workspace/libero_RL
DEV=$ROOT/tmp/rlinf_dsrl
cd $DEV && source $ROOT/RLinf/.venv-smolvla/bin/activate
export EMBODIED_PATH=$DEV/examples/embodiment REPO_PATH=$DEV PYTHONPATH=$DEV
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true WANDB__DISABLE_STATS=true WANDB_SILENT=true
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
LOG=$ROOT/tmp/rlt_stage2_dsrl/logs/smoke_$(date +%H%M%S); mkdir -p $LOG
python $EMBODIED_PATH/train_embodied_agent.py --config-path $EMBODIED_PATH/config --config-name libero_10_smolvla_rlt_stage2 \
  runner.logger.log_path=$LOG runner.max_epochs=${EPOCHS:-4} runner.save_interval=-1 \
  rollout.rlt_feature_model.model_path=$ROOT/models/smolvla_libero \
  rollout.rlt_feature_model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
  rollout.rlt_feature_model.smolvla.rlt_token_ckpt=$ROOT/tmp/rlt_stage1/rlt_stage1_ckpt.pt \
  rollout.rlt_feature_model.smolvla.noise_seed=${NOISE_SEED:-910000} \
  algorithm.rlt_schedule.warmup_min_size=40 algorithm.rlt_schedule.max_updates_per_train_step=10 \
  algorithm.rlt_schedule.warmup_post_collect_updates=10 \
  algorithm.actor_weight_schedule.warmup_updates=5 algorithm.actor_weight_schedule.ramp_updates=10 \
  algorithm.replay_buffer.min_buffer_size=20 actor.micro_batch_size=32 actor.global_batch_size=32 \
  env.train.seed=${ENV_SEED:-910000} \
  "$@" > $LOG/run.log 2>&1
echo "exit=$?" >> $LOG/run.log
echo $LOG > $ROOT/tmp/rlt_stage2_dsrl/last_log_dir
