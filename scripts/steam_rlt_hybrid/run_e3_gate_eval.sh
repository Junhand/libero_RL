#!/bin/bash
# E3: D's actor (rlt-s1cont-utd5, 500 epochs) evaluated with a kNN confidence gate (actor acts only where the RL token is familiar).
cd /workspace/libero_RL/tmp/rlt_stage2
CK=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt
REF=/workspace/libero_RL/tmp/hybrid/knn_ref.pt
run() { # name threshold
  echo "[$(date +%T)] start $1 (threshold $2)"
  TAG=e3_$1 EXTRA="rollout.rlt_gate.enable=True rollout.rlt_gate.ref_path=$REF rollout.rlt_gate.k=20 rollout.rlt_gate.threshold=$2 rollout.rlt_gate.log_path=/workspace/libero_RL/tmp/rlt_stage2/gate_$1.jsonl" ./eval_one.sh C 930000 $CK > /dev/null
  echo "[$(date +%T)] done $1: $(grep -h 'success_once' eval_e3_$1_seed930000.log | tail -1 | grep -o 'success_once=[0-9.]*')"
}
run g95 14.367
run g90 9.938
run g99 24.291
run gall 1000000000   # threshold so large that the gate never blocks: must reproduce D (0.62)
run gnone 0           # the actor is never used: the VLA alone through the RLT path (~0.46)
echo chain-done
