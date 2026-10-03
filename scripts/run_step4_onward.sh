#!/usr/bin/env bash
# Step3 完了待ち → Step4 (advantage 条件付き SFT、2000 ステップ) → Step9 (評価) を自動で進める。
# 実行: cd /workspace/libero_RL && nohup bash scripts/run_step4_onward.sh > /dev/null 2>&1 &
# イベントは tmp/step4_events.log、Step4 の出力は tmp/step4_attempt*.log、評価は tmp/step9_eval.log。
cd /workspace/libero_RL
ROOT=/workspace/libero_RL
EV=tmp/step4_events.log
D=$ROOT/data/smolvla_recap
SFT_V21=$D/libero10_task0_sft_target3;  SFT_V30=$D/v30/libero10_task0_sft_target3
TRN_V21=$D/libero10_task0_train_v2_merged; TRN_V30=$D/v30/libero10_task0_train_v2_merged
ADV=meta/advantages_fail300_N10_q30.parquet
LOGS=RLinf/logs
MAX_STEPS=2000
log() { echo "$(date +%T) $*" >> $EV; }
: >> $EV; log "start run_step4_onward (pid $$)"

# ---- 1. Step3 の完了待ち（落ちて advantage が無ければ、最大 3 回まで再開）----
step3_ok() { [ -f $SFT_V21/$ADV ] && [ -f $TRN_V21/$ADV ]; }
for try in 1 2 3; do
  while ps -eo cmd | grep -q "[c]ompute_advantages.py"; do sleep 60; done
  if step3_ok; then break; fi
  log "Step3 の出力が見つからない。再実行 try=$try"
  ( cd tmp; bash run_step3_v2.sh > step3_v2_retry$try.log 2>&1 )
done
if ! step3_ok; then log "ABORT: Step3 の advantage が作れなかった"; exit 1; fi
log "Step3 完了: $(RLinf/.venv/bin/python - <<E
import pandas as pd
for p in ["$SFT_V21/$ADV","$TRN_V21/$ADV"]:
    d=pd.read_parquet(p,columns=["advantage"]); print(p.split('/')[-3], len(d), "positive=%.3f"%d.advantage.mean(), end="; ")
E
)"

# ---- 2. Step4 の共通設定 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
TRAIN="data.train_data_paths=[{dataset_path:$SFT_V30,advantages_path:$SFT_V21/$ADV,weight:1.0},{dataset_path:$TRN_V30,advantages_path:$TRN_V21/$ADV,weight:1.0}]"
run_step4() { # run_step4 <micro batch> <max_steps> <save_interval> <log file> [extra hydra args]
  local mb=$1 steps=$2 save=$3 out=$4; shift 4
  ( source RLinf/.venv-smolvla/bin/activate
    bash scripts/run_rlinf_vla_sft.sh libero10_task0_recap_smolvla "$TRAIN" \
      actor.model.model_path=$ROOT/models/smolvla_libero \
      actor.fsdp_config.use_orig_params=true actor.model.precision=fp32 \
      actor.micro_batch_size=$mb actor.global_batch_size=256 data.num_workers=4 \
      runner.max_steps=$steps runner.save_interval=$save \
      actor.optim.total_training_steps=$steps actor.optim.lr_warmup_steps=100 \
      "$@" > $out 2>&1 )
}
sps_of() { tr '\r' '\n' < $1 | grep -oE "time/step=[0-9.]+" | tail -1 | cut -d= -f2; }

# ---- 3. micro batch の選択（32 → 16 → 8 の順に 4 ステップ試す）----
MB=""; SPS=""
for mb in 32 16 8; do
  run_step4 $mb 4 100000 tmp/step4_probe_mb$mb.log
  if grep -q "OutOfMemoryError" tmp/step4_probe_mb$mb.log; then log "probe MB=$mb: OOM"; continue; fi
  s=$(sps_of tmp/step4_probe_mb$mb.log)
  if [ -z "$s" ]; then log "probe MB=$mb: 失敗（OOM 以外。tmp/step4_probe_mb$mb.log を参照）"; continue; fi
  MB=$mb; SPS=$s; log "probe MB=$mb: ${s} 秒/ステップ（最後の 1 ステップ）"; break
done
[ -z "$MB" ] && { log "ABORT: どの micro batch でも動かなかった"; exit 1; }
# 約 1 時間ごとに checkpoint（50 の倍数、3000 を上限）
SAVE=$(python3 -c "s=float('$SPS'); print(min(3000, max(50, round(3600/s/50)*50)))")
log "Step4 本番: MB=$MB SAVE=$SAVE 見込み $(python3 -c "print(round($MAX_STEPS*float('$SPS')/3600,1))") 時間（probe の速度での単純計算）"

# ---- 4. Step4 本番（落ちたら最新の checkpoint から再開、最大 6 回。checkpoint は直近 2 つだけ残す）----
START=$(date +%s)
ckpts() { ls -d $LOGS/*-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "${d##*_} $d"; done | sort -n; }
latest_ckpt() { ckpts | tail -1 | cut -d' ' -f2-; }
prune() { ckpts | head -n -2 | cut -d' ' -f2- | while read d; do rm -rf "$d"; log "pruned $d"; done; }
for attempt in 1 2 3 4 5 6; do
  R=$(latest_ckpt); EXTRA=""; [ -n "$R" ] && EXTRA="+runner.resume_dir=$(realpath $R)"
  log "Step4 attempt $attempt resume=${R:-none}"
  run_step4 $MB $MAX_STEPS $SAVE tmp/step4_attempt$attempt.log $EXTRA &
  P=$!
  while kill -0 $P 2>/dev/null; do prune; sleep 120; done
  R=$(latest_ckpt); S=${R##*_}
  if [ "${S:-0}" -ge $MAX_STEPS ]; then log "DONE Step4 step $S ckpt=$R"; break; fi
  log "Step4 のプロセスが終了（ckpt ${S:-none}）。再開する"
done
R=$(latest_ckpt); S=${R##*_}
[ "${S:-0}" -ge $MAX_STEPS ] || { log "ABORT: Step4 が完了しなかった"; exit 1; }

# ---- 5. Step9 評価（ベースラインと同じ条件: Task 0、固定初期状態 50 個）----
W=$ROOT/$R/actor/model_state_dict/full_weights.pt
[ -f "$W" ] || { log "ABORT: $W がない"; exit 1; }
log "Step9 評価 開始 ckpt=$W"
( source RLinf/.venv-smolvla/bin/activate; cd RLinf
  bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
    rollout.model.model_path=$ROOT/models/smolvla_libero runner.ckpt_path=$W \
    rollout.model.smolvla.advantage_prompt=true \
    "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=10 env.eval.max_steps_per_rollout_epoch=2600 ) > tmp/step9_eval.log 2>&1
log "Step9 評価 終了: $(tr '\r' '\n' < tmp/step9_eval.log | grep -oE "'eval/success_once': array\([0-9.]+" | tail -1)（ベースライン 0.40）"
log "ALL DONE"
