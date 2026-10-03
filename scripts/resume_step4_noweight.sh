#!/usr/bin/env bash
# Step4 を global_step_1000 から再開し、データセットの重みなし(data.balance_dataset_weights=false)に切り替えて 2000 ステップまで進める。
# 終わったら Step9 評価まで自動で行う。落ちたら最新の checkpoint から再開（最大 6 回）、checkpoint は直近 2 つだけ残す。
# 実行: cd /workspace/libero_RL && nohup bash scripts/resume_step4_noweight.sh > /dev/null 2>&1 &
cd /workspace/libero_RL
ROOT=/workspace/libero_RL
EV=tmp/step4_resume_events.log
D=$ROOT/data/smolvla_recap
ADV=meta/advantages_fail300_N10_q30.parquet
LOGS=RLinf/logs
MAX_STEPS=2000; MB=16; SAVE=100
BASE_CKPT=$ROOT/$LOGS/20261003-09:38:15-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_1000
log() { echo "$(date +%T) $*" >> $EV; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
TRAIN="data.train_data_paths=[{dataset_path:$D/v30/libero10_task0_sft_target3,advantages_path:$D/libero10_task0_sft_target3/$ADV,weight:1.0},{dataset_path:$D/v30/libero10_task0_train_v2_merged,advantages_path:$D/libero10_task0_train_v2_merged/$ADV,weight:1.0}]"
run_step4() { # run_step4 <log file> [extra hydra args]
  local out=$1; shift
  ( source RLinf/.venv-smolvla/bin/activate
    bash scripts/run_rlinf_vla_sft.sh libero10_task0_recap_smolvla "$TRAIN" \
      actor.model.model_path=$ROOT/models/smolvla_libero \
      actor.fsdp_config.use_orig_params=true actor.model.precision=fp32 \
      actor.micro_batch_size=$MB actor.global_batch_size=256 data.num_workers=4 \
      runner.max_steps=$MAX_STEPS runner.save_interval=$SAVE \
      actor.optim.total_training_steps=$MAX_STEPS actor.optim.lr_warmup_steps=100 \
      data.balance_dataset_weights=false \
      "$@" > $out 2>&1 )
}
START=$(date +%s)
ckpts() { ls -d $LOGS/*-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "${d##*_} $d"; done | sort -n; }
latest_ckpt() { ckpts | tail -1 | cut -d' ' -f2-; }
prune() { ckpts | head -n -2 | cut -d' ' -f2- | while read d; do rm -rf "$d"; log "pruned $d"; done; }
log "start: resume from $BASE_CKPT, balance_dataset_weights=false, MB=$MB SAVE=$SAVE max_steps=$MAX_STEPS"
for attempt in 1 2 3 4 5 6; do
  R=$(latest_ckpt); [ -z "$R" ] && R=$BASE_CKPT
  log "attempt $attempt resume=$R"
  run_step4 tmp/step4_resume_attempt$attempt.log "+runner.resume_dir=$(realpath $R)" &
  P=$!
  while kill -0 $P 2>/dev/null; do prune; sleep 120; done
  R=$(latest_ckpt); S=${R##*_}
  if [ "${S:-0}" -ge $MAX_STEPS ]; then log "DONE Step4 step $S ckpt=$R"; break; fi
  log "プロセスが終了（ckpt ${S:-none}）。再開する"
done
R=$(latest_ckpt); S=${R##*_}
[ "${S:-0}" -ge $MAX_STEPS ] || { log "ABORT: Step4 が完了しなかった"; exit 1; }
W=$ROOT/$R/actor/model_state_dict/full_weights.pt
[ -f "$W" ] || { log "ABORT: $W がない"; exit 1; }
log "Step9 評価 開始 ckpt=$W"
( source RLinf/.venv-smolvla/bin/activate; cd RLinf
  bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
    rollout.model.model_path=$ROOT/models/smolvla_libero runner.ckpt_path=$W \
    rollout.model.smolvla.advantage_prompt=true \
    rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
    "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=10 env.eval.max_steps_per_rollout_epoch=2600 ) > tmp/step9_eval_final.log 2>&1
log "Step9 評価 終了: $(tr '\r' '\n' < tmp/step9_eval_final.log | grep -oE "'eval/success_once': array\([0-9.]+" | tail -1) (ベースライン 0.40、1000 ステップ時 0.46)"
log "ALL DONE"
