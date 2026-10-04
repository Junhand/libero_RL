#!/bin/bash
# Stage 1 only (first-grasp subtask), long run with UTD 5, then the base-protocol evaluation of the final actor.
#   ./run_stage1_long.sh <tag> <epochs>
# Re-runs after a crash resume from the newest complete checkpoint (weights + replay), as in run_two_stage.sh.
cd /workspace/libero_RL/tmp/rlt_stage2
TAG=${1:?tag}; E1=${2:?epochs}; MAX_RETRY=${MAX_RETRY:-3}; SEED=${EVAL_SEED:-930000}; INIT_RESUME=${INIT_RESUME:-}
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
say() { echo "[$(date +%T)] $*"; }

# UTD = update_epoch / train_every_transitions = 25 / 5 = 5; the per-epoch cap must not limit it (~34 transitions x 5 = 170).
S1="algorithm.update_epoch=25 algorithm.rlt_schedule.warmup_min_size=1000 algorithm.rlt_schedule.max_updates_per_train_step=400 algorithm.rlt_schedule.warmup_post_collect_updates=1000 algorithm.actor_weight_schedule.warmup_updates=1000 algorithm.actor_weight_schedule.ramp_updates=1000 algorithm.replay_buffer.min_buffer_size=1000 actor.micro_batch_size=128 actor.global_batch_size=512"
SAVE=50; LOGGER="runner.logger.logger_backends=[tensorboard,wandb]"
RESUME_SCHED="algorithm.rlt_schedule.warmup_post_collect_updates=0 algorithm.actor_weight_schedule.warmup_updates=0 algorithm.actor_weight_schedule.ramp_updates=0"

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

NAME=$TAG-s1-grasp; attempt=0; envs=3
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
    env.train.rlt_policy_switch.mode=grasp_subtask env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 \
    algorithm.rlt_phase_return=False $S1 $resume_args
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
LOG=$(./eval_one.sh C $SEED $CK/actor/model_state_dict/full_weights.pt | tail -1)
say "evaluation finished: $(grep -h '^exit=' $LOG | tail -1) | $LOG"
python3 upload_eval_to_wandb.py $LOG "eval-D-$TAG-step${CK##*_}-seed$SEED" "RLT actor, stage 1 only ($E1 epochs, UTD 5, fixed_std 0.02), actor acts until the first grasp" 2>&1 | tail -3
say "all done"
