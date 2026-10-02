#!/usr/bin/env bash
# Step2 v2: 4000 steps, checkpoint every 300 steps (~1 h at 11.5 s/step, micro batch 16 on RTX 6000 Ada 48GB), eval every 300 steps (300 samples x 3 sets).
# Restarts from the latest checkpoint if the training process dies. Keeps only the last 2 checkpoints.
# Adopt an already running training (e.g. after fixing this script): ADOPT_PID=<pid of run_step2_v2.sh> START=<epoch of the run start> bash scripts/step2_v2_supervisor.sh
cd /workspace/libero_RL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=RLinf/logs/value_sft
# the checkpoint dir uses experiment_name (step2_value_sft_v2); sort by the numeric step (the path has other underscores)
if [ -n "$ADOPT_PID" ]; then FIRST=2; else FIRST=1; START=$(date +%s); : > tmp/step2_v2_events.log; fi
ckpts() { ls -d $LOGS/recap_value_model_sft-*/step2_value_sft_v2/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "${d##*_} $d"; done | sort -n; }
latest_ckpt() { ckpts | tail -1 | cut -d' ' -f2-; }
prune() { ckpts | head -n -2 | cut -d' ' -f2- | while read d; do rm -rf "$d"; echo "$(date +%T) pruned $d" >> tmp/step2_v2_events.log; done; }
[ -n "$ADOPT_PID" ] && echo "$(date +%T) supervisor adopted running training pid $ADOPT_PID" >> tmp/step2_v2_events.log
for attempt in $(seq $FIRST 6); do
  if [ -n "$ADOPT_PID" ] && [ "$attempt" = "$FIRST" ]; then
    P=$ADOPT_PID
  else
    R=$(latest_ckpt); EXTRA=""; [ -n "$R" ] && EXTRA="+runner.resume_dir=$(realpath $R)"
    echo "$(date +%T) attempt $attempt resume=${R:-none}" >> tmp/step2_v2_events.log
    ( cd tmp; MB=16 SAVE=300 VAL=300 EVS=300 bash ../scripts/run_step2_v2.sh 4000 $EXTRA > step2_v2_attempt$attempt.log 2>&1 ) &
    P=$!
  fi
  while kill -0 $P 2>/dev/null; do prune; sleep 120; done
  R=$(latest_ckpt); S=${R##*_}
  if [ "${S:-0}" -ge 4000 ]; then echo "$(date +%T) DONE step $S" >> tmp/step2_v2_events.log; break; fi
  echo "$(date +%T) process ended at ckpt ${S:-none}; restarting" >> tmp/step2_v2_events.log
done
