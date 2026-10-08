#!/bin/bash
# 50-episode evaluation (50 fixed initial states of Task 0, 2 envs x 25 passes) of a Flow-SDE PPO checkpoint, in fp32 through the
# SmolVLAForFlowSDE wrapper (deterministic ODE with the trained number of Euler steps).
#   DEV=<dev copy> ./eval_sde.sh <name> <seed> [full_weights.pt]     (no weights = the base weights through the same wrapper)
ROOT=/workspace/libero_RL; DEV=${DEV:-$ROOT/tmp/rlinf_dev}; NAME=${1:?name}; SEED=${2:?seed}; CKW=${3:-}
source $ROOT/RLinf/.venv-smolvla/bin/activate; export HF_HUB_OFFLINE=1
OUT=$ROOT/tmp/hybrid_fpo/eval_${NAME}_seed${SEED}.log
CK=""; [ -n "$CKW" ] && CK="runner.ckpt_path=$CKW"
cd $DEV; bash evaluations/run_eval.sh libero libero_10_smolvla_eval rollout.model.model_path=$ROOT/models/smolvla_libero \
  rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct rollout.model.add_value_head=True \
  +rollout.model.smolvla.sde.enabled=True +rollout.model.smolvla.sde.num_steps=${NSTEPS:-10} +rollout.model.smolvla.sde.noise_level=0.5 \
  $CK rollout.model.smolvla.noise_seed=$SEED +env.eval.task_id_filter=[0] \
  env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 env.eval.video_cfg.save_video=False > $OUT 2>&1
echo "exit=$?" >> $OUT; echo $OUT
