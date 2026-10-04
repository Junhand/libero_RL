#!/bin/bash
# Dedicated second-phase actor: the learned actor (and critic) acts ONLY in the second grasp phase and is trained from scratch
# (reference warm-up, then BC->Q ramp as in stage 1); the first phase is done by a frozen actor (PRE_ACTOR weights, eval mode).
# This variant adds the miss penalty (MISS_PENALTY, default 0.2) to the reward of the second phase.
# Final evaluation: frozen first-phase actor + this second-phase actor, base protocol.
#   ./run_stage1_long.sh <tag> <epochs>
# Re-runs after a crash resume from the newest complete checkpoint (weights + replay), as in run_two_stage.sh.
cd /workspace/libero_RL/tmp/rlt_stage2_b
TAG=${1:?tag}; E1=${2:?epochs}; MAX_RETRY=${MAX_RETRY:-3}; SEED=${EVAL_SEED:-930000}; INIT_RESUME=${INIT_RESUME:-}; PRE_ACTOR=${PRE_ACTOR:?frozen first-phase actor full_weights.pt}
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
say() { echo "[$(date +%T)] $*"; }

# UTD = update_epoch / train_every_transitions = 25 / 5 = 5; the per-epoch cap must not limit it (~34 transitions x 5 = 170).
S1="algorithm.update_epoch=25 algorithm.rlt_schedule.warmup_min_size=1000 algorithm.rlt_schedule.max_updates_per_train_step=400 algorithm.rlt_schedule.warmup_post_collect_updates=1000 algorithm.actor_weight_schedule.warmup_updates=1000 algorithm.actor_weight_schedule.ramp_updates=1000 algorithm.replay_buffer.min_buffer_size=1000 actor.micro_batch_size=128 actor.global_batch_size=512"
SAVE=50; LOGGER="runner.logger.logger_backends=[tensorboard]"
RESUME_SCHED="algorithm.rlt_schedule.warmup_post_collect_updates=0 algorithm.actor_weight_schedule.warmup_updates=0 algorithm.actor_weight_schedule.ramp_updates=0"

latest_ckpt() {
  local best="" bestn=-1 d n
  for d in /workspace/libero_RL/tmp/rlt_stage2_b/logs/smoke_*/$1/checkpoints/global_step_*; do
    [ -f "$d/actor/model_state_dict/full_weights.pt" ] || continue
    [ -f "$d/actor/sac_components/target_model/checkpoint_rank_0.pt" ] || continue
    [ -f "$d/actor/sac_components/replay_buffer/rank_0/metadata.json" ] || continue
    [ -f "$d/actor/sac_components/replay_buffer/rank_0/trajectory_index.json" ] || continue
    n=${d##*_}; if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$d; fi
  done
  echo "$best"
}

NAME=$TAG-p2only-miss; attempt=0; envs=3
while [ $attempt -le $MAX_RETRY ]; do
  attempt=$((attempt + 1)); resume_args=""; ck=$(latest_ckpt "$NAME")
  if [ -n "$ck" ] && [ "${ck##*_}" -lt "$E1" ]; then
    say "attempt $attempt: resuming from $ck (weights + replay), envs=$envs"
    resume_args="runner.resume_dir=$ck algorithm.resume_load_replay=True $RESUME_SCHED"
  elif [ -n "$INIT_RESUME" ]; then
    say "attempt $attempt: continuing from the earlier run $INIT_RESUME (weights + replay), envs=$envs"
    resume_args="runner.resume_dir=$INIT_RESUME algorithm.resume_load_replay=True $RESUME_SCHED"
  else
    say "attempt $attempt: starting from scratch, envs=$envs"
  fi
  EPOCHS=$E1 ./run_smoke.sh runner.save_interval=$SAVE env.train.total_num_envs=$envs env.eval.total_num_envs=2 \
    runner.logger.project_name=libero_RL $LOGGER runner.logger.experiment_name=$NAME \
    env.train.rlt_policy_switch.mode=grasp_subtask env.train.rlt_policy_switch.num_phases=2 env.train.rlt_policy_switch.actor_phase=1 env.train.rlt_policy_switch.pre_actor=True rollout.rlt_pre_actor.ckpt_path=$PRE_ACTOR env.train.rlt_policy_switch.miss_penalty=${MISS_PENALTY:-0.2} actor.model.fixed_std=${FIXED_STD:-0.05} env.train.max_episode_steps=400 env.train.max_steps_per_rollout_epoch=400 \
    algorithm.rlt_phase_return=False algorithm.rlt_segment_terminal=True $S1 $resume_args
  log=$(cat last_log_dir); code=$(grep -h '^exit=' $log/run.log | tail -1)
  say "attempt $attempt finished: $code | $log"
  ray stop --force >/dev/null 2>&1; sleep 5
  ck=$(latest_ckpt "$NAME")
  if [ "$code" = "exit=0" ] && [ -n "$ck" ] && [ "${ck##*_}" -ge "$E1" ]; then break; fi
  if grep -q "low on memory" $log/run.log; then envs=2; say "out of memory -> retry with 2 envs"; fi
  sleep 30
done
CK=$(latest_ckpt "$NAME"); [ -n "$CK" ] && [ "${CK##*_}" -ge "$E1" ] || { say "training did not finish"; exit 1; }

say "=== evaluation of $CK (base protocol, seed $SEED)"
LOG=$(EXTRA="env.eval.rlt_policy_switch.num_phases=2 env.eval.rlt_policy_switch.actor_phase=1 env.eval.rlt_policy_switch.pre_actor=True rollout.rlt_pre_actor.ckpt_path=$PRE_ACTOR" TAG=I ./eval_one.sh C $SEED $CK/actor/model_state_dict/full_weights.pt | tail -1)
say "evaluation finished: $(grep -h '^exit=' $LOG | tail -1) | $LOG"
say "all done"
