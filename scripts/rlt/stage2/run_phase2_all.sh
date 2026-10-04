#!/bin/bash
# 0b: the stage-1 actor (D, step 500) with the 2-phase gate, no training -> evaluation + wandb.
# 1 : training with the 2-phase gate, resumed from D (weights + replay) -> final evaluation + wandb.
cd /workspace/libero_RL/tmp/rlt_stage2
D=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp/checkpoints/global_step_500
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
echo "[$(date +%T)] 0b: evaluate D with the 2-phase gate (no training)"; ray stop --force >/dev/null 2>&1; sleep 5
LOG=$(EXTRA="env.eval.rlt_policy_switch.num_phases=2" TAG=E ./eval_one.sh C 930000 $D/actor/model_state_dict/full_weights.pt | tail -1)
echo "[$(date +%T)] 0b done: $(grep -h '^exit=' $LOG | tail -1) | success_once=$(grep -o "eval/success_once': array([0-9.]*" $LOG | tail -1 | grep -o '[0-9.]*$')"
python3 upload_eval_to_wandb.py $LOG "eval-E-stage1-actor-with-2phase-gate-no-training-seed930000" "Stage-1 actor (D, step 500) with the 2-phase gate (the actor also acts in the second grasp phase), NO training for the second phase" 2>&1 | tail -2
echo "[$(date +%T)] 1: training with the 2-phase gate"; ray stop --force >/dev/null 2>&1; sleep 5
INIT_RESUME=$D ENV_SEED=960000 NOISE_SEED=960000 EVAL_SEED=930000 ./run_phase2_long.sh rlt-phase2-utd5 750
echo "[$(date +%T)] all done"
