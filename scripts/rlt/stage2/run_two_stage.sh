#!/bin/bash
# Two-stage RLT training with automatic re-runs after a crash.
#
#   Stage 1: first-grasp subtask (reward 1 at the confirmed grasp; the episode ends there).
#   Stage 2: full task (task-completion reward; the actor acts until the first grasp), started from
#            the stage-1 actor/critic weights (the replay buffer is not carried over).
#
#   MODE=main  ./run_two_stage.sh <tag> <stage1_epochs> <stage2_epochs>
#   MODE=small ./run_two_stage.sh small 6 6     # same structure at a tiny scale (check before a long run)
#
# Re-runs: if a stage ends with a non-zero exit code, it is started again (up to MAX_RETRY times) from
# the newest complete checkpoint of that stage (weights + replay buffer), or from the stage start if
# there is none. After an out-of-memory failure the retry uses 2 training envs instead of 3.
# Running the whole script again skips stages that already finished (state_<tag>.txt).
cd /workspace/libero_RL/tmp/rlt_stage2
TAG=${1:?tag}; E1=${2:?stage1 epochs}; E2=${3:?stage2 epochs}; MODE=${MODE:-main}
MAX_RETRY=${MAX_RETRY:-3}
STATE=state_$TAG.txt; touch $STATE
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
say() { echo "[$(date +%T)] $*"; }

if [ "$MODE" = main ]; then
  S1="algorithm.rlt_schedule.warmup_min_size=1000 algorithm.rlt_schedule.max_updates_per_train_step=40 algorithm.rlt_schedule.warmup_post_collect_updates=1000 algorithm.actor_weight_schedule.warmup_updates=1000 algorithm.actor_weight_schedule.ramp_updates=1000 algorithm.replay_buffer.min_buffer_size=1000 actor.micro_batch_size=128 actor.global_batch_size=512"
  S2="algorithm.rlt_schedule.warmup_min_size=300 algorithm.rlt_schedule.max_updates_per_train_step=40 algorithm.rlt_schedule.warmup_post_collect_updates=0 algorithm.actor_weight_schedule.warmup_updates=0 algorithm.actor_weight_schedule.ramp_updates=0 algorithm.replay_buffer.min_buffer_size=300 actor.micro_batch_size=128 actor.global_batch_size=512"
  SAVE1=50; SAVE2=25; LOGGER="runner.logger.logger_backends=[tensorboard,wandb]"
else
  S1="algorithm.rlt_schedule.warmup_min_size=60 algorithm.rlt_schedule.max_updates_per_train_step=10 algorithm.rlt_schedule.warmup_post_collect_updates=20 algorithm.actor_weight_schedule.warmup_updates=10 algorithm.actor_weight_schedule.ramp_updates=10 algorithm.replay_buffer.min_buffer_size=60 actor.micro_batch_size=32 actor.global_batch_size=32"
  S2="algorithm.rlt_schedule.warmup_min_size=60 algorithm.rlt_schedule.max_updates_per_train_step=10 algorithm.rlt_schedule.warmup_post_collect_updates=0 algorithm.actor_weight_schedule.warmup_updates=0 algorithm.actor_weight_schedule.ramp_updates=0 algorithm.replay_buffer.min_buffer_size=60 actor.micro_batch_size=32 actor.global_batch_size=32"
  SAVE1=3; SAVE2=2; LOGGER="runner.logger.logger_backends=[tensorboard]"
fi
# After a crash inside a stage the actor is already trained and the replay buffer is restored:
# no VLA-only warm-up and no BC-weight warm-up again (update_step restarts from 0).
RESUME_SCHED="algorithm.rlt_schedule.warmup_post_collect_updates=0 algorithm.actor_weight_schedule.warmup_updates=0 algorithm.actor_weight_schedule.ramp_updates=0"

# Newest complete checkpoint of an experiment name (all attempts), or empty.
latest_ckpt() {
  local best="" bestn=-1 d n
  for d in /workspace/libero_RL/tmp/rlt_stage2/logs/smoke_*/$1/checkpoints/global_step_*; do
    [ -f "$d/actor/model_state_dict/full_weights.pt" ] || continue
    [ -f "$d/actor/sac_components/target_model/checkpoint_rank_0.pt" ] || continue
    [ -f "$d/actor/sac_components/replay_buffer/rank_0/metadata.json" ] || continue
    [ -f "$d/actor/sac_components/replay_buffer/rank_0/trajectory_index.json" ] || continue
    n=${d##*_}; if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$d; fi
  done
  echo "$best"
}

# run_stage <experiment name> <final global step> <save interval> <first-attempt resume dir or ""> <stage args...>
run_stage() {
  local name=$1 total=$2 save=$3 init_resume=$4; shift 4
  local stage_args="$*" attempt=0 envs=3 resume_args ck log code
  while [ $attempt -le $MAX_RETRY ]; do
    attempt=$((attempt + 1))
    resume_args=""
    ck=$(latest_ckpt "$name")
    if [ -n "$ck" ] && [ "${ck##*_}" -lt "$total" ]; then
      say "$name attempt $attempt: resuming from $ck (weights + replay), envs=$envs"
      resume_args="runner.resume_dir=$ck algorithm.resume_load_replay=True $RESUME_SCHED"
    elif [ -n "$init_resume" ]; then
      say "$name attempt $attempt: starting from $init_resume (weights only), envs=$envs"
      resume_args="runner.resume_dir=$init_resume algorithm.resume_load_replay=False"
    else
      say "$name attempt $attempt: starting from scratch, envs=$envs"
    fi
    EPOCHS=$total ./run_smoke.sh runner.save_interval=$save env.train.total_num_envs=$envs env.eval.total_num_envs=2 \
      runner.logger.project_name=libero_RL $LOGGER runner.logger.experiment_name=$name $stage_args $resume_args
    log=$(cat last_log_dir); code=$(grep -h '^exit=' $log/run.log | tail -1)
    say "$name attempt $attempt finished: $code | $log"
    ray stop --force >/dev/null 2>&1; sleep 5
    ck=$(latest_ckpt "$name")
    if [ "$code" = "exit=0" ] && [ -n "$ck" ] && [ "${ck##*_}" -ge "$total" ]; then return 0; fi
    if grep -q "low on memory" $log/run.log; then envs=2; say "$name: out of memory -> retry with 2 envs"; fi
    sleep 30
  done
  say "$name: giving up after $MAX_RETRY retries"; return 1
}

S1NAME=$TAG-s1-grasp; S2NAME=$TAG-s2-task
if ! grep -q "^stage1_done=" $STATE; then
  say "=== stage 1 (grasp subtask), $E1 epochs"
  run_stage $S1NAME $E1 $SAVE1 "" \
    env.train.rlt_policy_switch.mode=grasp_subtask env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 \
    algorithm.rlt_phase_return=False $S1 || exit 1
  CK1=$(latest_ckpt $S1NAME); echo "stage1_done=$CK1" >> $STATE
else
  say "stage 1 already finished (state file)"
fi
CK1=$(grep "^stage1_done=" $STATE | tail -1 | cut -d= -f2)
[ -d "$CK1/actor" ] || { say "stage-1 checkpoint missing: $CK1"; exit 1; }

if ! grep -q "^stage2_done=" $STATE; then
  say "=== stage 2 (full task), $E2 epochs after step $E1"
  run_stage $S2NAME $((E1 + E2)) $SAVE2 "$CK1" \
    env.train.rlt_policy_switch.mode=full_task env.train.max_episode_steps=520 env.train.max_steps_per_rollout_epoch=520 \
    algorithm.rlt_phase_return=True $S2 || exit 1
  echo "stage2_done=$(latest_ckpt $S2NAME)" >> $STATE
fi
say "all done: $(cat $STATE | tr '\n' ' ')"
