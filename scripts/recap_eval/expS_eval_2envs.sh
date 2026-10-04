#!/bin/bash
# Evaluation of experiment S (global_step_1450), 2 runs x 50 trials, with advantage_prompt=true (as the training script does),
# 2 envs x 25 passes (10 envs run out of memory in the 25GB container).
ROOT=/workspace/libero_RL; OUT=$ROOT/tmp/recap_expS_eval; mkdir -p $OUT
W=$ROOT/RLinf/logs/20261004-09:55:21-libero10_task0_recap_smolvla/step4_expS_outcome_1450/checkpoints/global_step_1450/actor/model_state_dict/full_weights.pt
ls -la $W > /dev/null || exit 1
for k in 1 2; do
  ray stop --force >/dev/null 2>&1; sleep 5; L=$OUT/eval_expS_run$k.log; echo "[$(date +%T)] start run $k"
  ( cd $ROOT/RLinf && source .venv-smolvla/bin/activate && HF_HUB_OFFLINE=1 bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
      rollout.model.model_path=$ROOT/models/smolvla_libero runner.ckpt_path=$W rollout.model.smolvla.advantage_prompt=true \
      rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
      "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 ) > $L 2>&1
  echo "exit=$?" >> $L
  echo "[$(date +%T)] done run $k: $(tr '\r' '\n' < $L | grep -oE "'eval/success_once': array\([0-9.]+" | tail -1)"
done; echo "[$(date +%T)] all done"
