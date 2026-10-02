#!/usr/bin/env bash
# Step2 v2: 4000 steps, checkpoint every 150 steps (~1 h at 24 s/step), eval every 150 steps (1000 samples x 3 sets).
# Restarts from the latest checkpoint if the training process dies. Keeps the last 2 checkpoints + every 600th step.
cd /workspace/libero_RL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=RLinf/logs/value_sft; START=$(date +%s); : > tmp/step2_v2_events.log
latest_ckpt() { ls -d $LOGS/recap_value_model_sft-*/step2_value_sft/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "$d"; done | sort -t_ -k4 -n | tail -1; }
prune() {
  ls -d $LOGS/recap_value_model_sft-*/step2_value_sft/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "$d"; done | sort -t_ -k4 -n | head -n -2 | while read d; do
    s=${d##*_}; [ $((s % 600)) -eq 0 ] || { rm -rf "$d"; echo "$(date +%T) pruned $d" >> tmp/step2_v2_events.log; }
  done
}
for attempt in 1 2 3 4 5 6; do
  R=$(latest_ckpt); EXTRA=""; [ -n "$R" ] && EXTRA="+runner.resume_dir=$(realpath $R)"
  echo "$(date +%T) attempt $attempt resume=${R:-none}" >> tmp/step2_v2_events.log
  ( cd tmp; MB=8 SAVE=150 VAL=150 EVS=1000 bash ../scripts/run_step2_v2.sh 4000 $EXTRA > step2_v2_attempt$attempt.log 2>&1 ) &
  P=$!
  while kill -0 $P 2>/dev/null; do prune; sleep 120; done
  R=$(latest_ckpt); S=${R##*_}
  if [ "${S:-0}" -ge 4000 ]; then echo "$(date +%T) DONE step $S" >> tmp/step2_v2_events.log; break; fi
  echo "$(date +%T) process ended at ckpt ${S:-none}; restarting" >> tmp/step2_v2_events.log
done
