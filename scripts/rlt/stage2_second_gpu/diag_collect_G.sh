#!/bin/bash
# Re-run the base-protocol evaluation (same seed) while recording every step (state, action, success) for diagnosis.
#   ./diag_collect.sh D   -> RLT actor (stage 1 long, step 500)      B -> SmolVLA only through the RLT path
ROOT=/workspace/libero_RL; W=${1:?G}; SEED=930000
source $ROOT/RLinf/.venv-smolvla/bin/activate; export HF_HUB_OFFLINE=1; cd $ROOT/tmp/rlinf_dev
CKW=$ROOT/tmp/rlt_stage2_b/logs/smoke_102505/rlt-phase2-std005-2phase/checkpoints/global_step_750/actor/model_state_dict/full_weights.pt
FEAT="rollout.rlt_feature_model.model_path=$ROOT/models/smolvla_libero rollout.rlt_feature_model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct rollout.rlt_feature_model.smolvla.rlt_token_ckpt=$ROOT/tmp/rlt_stage1/rlt_stage1_ckpt.pt rollout.rlt_feature_model.smolvla.noise_seed=$SEED"
SAVE=$ROOT/tmp/rlt_stage2_b/diag/$W
COL="+env.eval.data_collection.enabled=True +env.eval.data_collection.save_dir=$SAVE +env.eval.data_collection.export_format=lerobot +env.eval.data_collection.only_success=False +env.eval.data_collection.robot_type=panda +env.eval.data_collection.fps=10 env.eval.use_ordered_reset_state_ids=True"
COMMON="env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 env.eval.video_cfg.save_video=False"
[ -e "$SAVE" ] && { echo "exists: $SAVE"; exit 1; }
EXTRA="runner.ckpt_path=$CKW env.eval.rlt_policy_switch.num_phases=2"
bash evaluations/run_eval.sh libero libero_10_smolvla_rlt_eval $FEAT $EXTRA $COMMON $COL > $ROOT/tmp/rlt_stage2_b/diag/eval_$W.log 2>&1
echo "exit=$?" >> $ROOT/tmp/rlt_stage2_b/diag/eval_$W.log
