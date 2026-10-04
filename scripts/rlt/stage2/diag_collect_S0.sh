#!/bin/bash
# S0: base-protocol evaluation (same seed, every step recorded) of an actor used ONLY until the first grasp (num_phases=1);
# SmolVLA alone does the rest. Same weights as the 2-phase runs, no training.
#   ./diag_collect_S0.sh <name> <full_weights.pt>      -> diag/<name>, diag/eval_<name>.log
ROOT=/workspace/libero_RL; W=${1:?name}; CKW=${2:?full_weights.pt}; SEED=${SEED:-930000}
source $ROOT/RLinf/.venv-smolvla/bin/activate; export HF_HUB_OFFLINE=1; cd $ROOT/tmp/rlinf_dev
FEAT="rollout.rlt_feature_model.model_path=$ROOT/models/smolvla_libero rollout.rlt_feature_model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct rollout.rlt_feature_model.smolvla.rlt_token_ckpt=$ROOT/tmp/rlt_stage1/rlt_stage1_ckpt.pt rollout.rlt_feature_model.smolvla.noise_seed=$SEED"
SAVE=$ROOT/tmp/rlt_stage2/diag/$W
COL="+env.eval.data_collection.enabled=True +env.eval.data_collection.save_dir=$SAVE +env.eval.data_collection.export_format=lerobot +env.eval.data_collection.only_success=False +env.eval.data_collection.robot_type=panda +env.eval.data_collection.fps=10 env.eval.use_ordered_reset_state_ids=True"
COMMON="env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 env.eval.video_cfg.save_video=False"
[ -e "$SAVE" ] && { echo "exists: $SAVE"; exit 1; }
bash evaluations/run_eval.sh libero libero_10_smolvla_rlt_eval $FEAT runner.ckpt_path=$CKW env.eval.rlt_policy_switch.num_phases=1 $COMMON $COL > $ROOT/tmp/rlt_stage2/diag/eval_$W.log 2>&1
echo "exit=$?" >> $ROOT/tmp/rlt_stage2/diag/eval_$W.log
