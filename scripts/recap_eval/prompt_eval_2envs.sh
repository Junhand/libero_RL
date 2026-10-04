#!/bin/bash
# Work 1: evaluate checkpoints B (global_step_1000) and C (global_step_2000) with advantage_prompt=true/false (50 trials each, 4 runs; 2 envs x 25 passes because 10 envs run out of memory in the 25GB container).
#   nohup bash recap_prompt_eval.sh > recap_prompt_eval.out 2>&1 &
ROOT=/workspace/libero_RL; OUT=$ROOT/tmp/recap_prompt_eval; mkdir -p $OUT
declare -A CK=(
 [B]=$ROOT/RLinf/logs/20261003-09:38:15-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt
 [C]=$ROOT/RLinf/logs/20261003-20:10:02-libero10_task0_recap_smolvla/step4_recap_noweight_resume1000/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt )
for W in B C; do ls -la ${CK[$W]} > /dev/null || { echo "missing ${CK[$W]}"; exit 1; }; done
for W in B C; do for P in true false; do
  ray stop --force >/dev/null 2>&1; sleep 5; L=$OUT/eval_${W}_prompt_${P}.log; echo "[$(date +%T)] start $W prompt=$P"
  ( cd $ROOT/RLinf && source .venv-smolvla/bin/activate && HF_HUB_OFFLINE=1 bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
      rollout.model.model_path=$ROOT/models/smolvla_libero runner.ckpt_path=${CK[$W]} rollout.model.smolvla.advantage_prompt=$P \
      rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
      "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000 ) > $L 2>&1
  echo "exit=$?" >> $L
  echo "[$(date +%T)] done $W prompt=$P: $(tr '\r' '\n' < $L | grep -oE "'eval/success_once': array\([0-9.]+" | tail -1)"
done; done
echo "[$(date +%T)] all done"
