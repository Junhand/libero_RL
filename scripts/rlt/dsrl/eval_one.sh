#!/bin/bash
# One evaluation under the base-model protocol (README step 4), using the evaluation-only mode (evaluations/run_eval.sh).
#   ./eval_one.sh A <seed>             SmolVLA alone, original pipeline (shared RLinf tree, libero_10_smolvla_eval)
#   ./eval_one.sh B <seed>             SmolVLA alone through the RLT code path (dev copy, libero_10_smolvla_rlt_eval, switch disabled)
#   ./eval_one.sh C <seed> <full_weights.pt>   RLT actor + SmolVLA (dev copy, libero_10_smolvla_rlt_eval)
# 50 fixed initial states of Task 0, each once; 2 parallel envs x 25 passes; videos at 10 fps.
ROOT=/workspace/libero_RL; WHICH=${1:?A|B|C}; SEED=${2:?seed}; CKW=${3:-}
source $ROOT/RLinf/.venv-smolvla/bin/activate; export HF_HUB_OFFLINE=1
OUT=$ROOT/tmp/rlt_stage2_dsrl/eval_${TAG:-$WHICH}_seed${SEED}.log; EXTRA=${EXTRA:-}
COMMON="env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 env.eval.video_cfg.save_video=True +env.eval.video_cfg.fps=10"
VLA="rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct"
FEAT="rollout.rlt_feature_model.model_path=$ROOT/models/smolvla_libero rollout.rlt_feature_model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct rollout.rlt_feature_model.smolvla.rlt_token_ckpt=$ROOT/tmp/rlt_stage1/rlt_stage1_ckpt.pt rollout.rlt_feature_model.smolvla.noise_seed=$SEED"
case $WHICH in
  A) cd $ROOT/RLinf; bash evaluations/run_eval.sh libero libero_10_smolvla_eval rollout.model.model_path=$ROOT/models/smolvla_libero $VLA \
       rollout.model.smolvla.noise_seed=$SEED +env.eval.task_id_filter=[0] $COMMON > $OUT 2>&1 ;;
  B) cd $ROOT/tmp/rlinf_dsrl; bash evaluations/run_eval.sh libero libero_10_smolvla_rlt_eval $FEAT env.eval.rlt_policy_switch.enable=False $COMMON > $OUT 2>&1 ;;
  C) cd $ROOT/tmp/rlinf_dsrl; bash evaluations/run_eval.sh libero libero_10_smolvla_rlt_eval $FEAT runner.ckpt_path=$CKW $COMMON $EXTRA > $OUT 2>&1 ;;
esac
echo "exit=$?" >> $OUT; echo $OUT
