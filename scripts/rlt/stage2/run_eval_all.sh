#!/bin/bash
# A, B, C under the same protocol with one noise seed; each result (metrics + video clips) is uploaded to wandb.
cd /workspace/libero_RL/tmp/rlt_stage2
SEED=${1:?seed}
CKW=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_223023/rlt2stage-std0.02-v2-s2-task/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
declare -A NAME=( [A]="eval-A-smolvla-base-seed$SEED" [B]="eval-B-smolvla-via-rlt-path-seed$SEED" [C]="eval-C-rlt-actor-step400-seed$SEED" )
declare -A DESC=( [A]="SmolVLA alone, original eval pipeline (libero_10_smolvla_eval)" [B]="SmolVLA alone through the RLT code path (actor switch disabled)" [C]="RLT actor (2-stage training, global_step_400, fixed_std 0.02) + SmolVLA; actor acts until the first grasp" )
for W in A B C; do
  echo "[$(date +%T)] start $W"; ray stop --force >/dev/null 2>&1; sleep 5
  LOG=$(./eval_one.sh $W $SEED $CKW | tail -1)
  echo "[$(date +%T)] done $W: $(grep -h '^exit=' $LOG | tail -1) | $LOG"
  python3 upload_eval_to_wandb.py $LOG "${NAME[$W]}" "${DESC[$W]}" 2>&1 | tail -4
done
echo "[$(date +%T)] all done"
