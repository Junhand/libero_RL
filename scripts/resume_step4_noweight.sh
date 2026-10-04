#!/usr/bin/env bash
# Step4 を再開して MAX_STEPS まで進める。sft/rollout の重みは BALANCE で切り替える（既定 true = 50:50、false = 重みなし）。
# 終わったら Step9 評価まで自動で行う。落ちたら最新の checkpoint から再開（最大 6 回）、checkpoint は直近 2 つだけ残す。
# メトリクスは tensorboard と wandb(project libero_RL) の両方に記録する。
# 実行中の学習を引き継ぐ場合: ADOPT_PID=<train_vla_sft.py の PID> START=<学習開始の epoch 秒> bash scripts/resume_step4_noweight.sh
# 実行: cd /workspace/libero_RL && nohup bash scripts/resume_step4_noweight.sh > /dev/null 2>&1 &
cd /workspace/libero_RL
ROOT=/workspace/libero_RL
EV=tmp/step4_resume_events.log
D=$ROOT/data/smolvla_recap
ADV_TAG=${ADV_TAG:-fail300_N10_q30}   # advantage のタグ（q10 など relabel したものに切り替える）
ADV=meta/advantages_$ADV_TAG.parquet
LOGS=RLinf/logs
MAX_STEPS=${MAX_STEPS:-2000}; MB=${MB:-16}; SAVE=${SAVE:-100}; N_EVAL=${N_EVAL:-1}   # N_EVAL: 評価を繰り返す回数（1 回 = 50 試行）
# 環境変数で変えられる設定（既定: 重みなし = balance_dataset_weights=false、global_step_1000 から再開）
# 注意: smolVLA 用ローダ(smolvla_cfg.py)の balance は RLinf 本体(ReCapMixtureDataset)と意味が逆。
#   false = 各フレームが同じ確率で選ばれる（フレーム数の比。RLinf 公式の既定 balance=true と同じ挙動。sft 約 0.13%）
#   true  = データセットごとに weight で選ばれる（sft と rollout が 50:50）
BALANCE=${BALANCE:-false}
EXP_NAME=${EXP_NAME:-step4_recap_framewise_resume1000}
BASE_CKPT=${BASE_CKPT:-$ROOT/$LOGS/20261003-09:38:15-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_1000}
log() { echo "$(date +%T) $*" >> $EV; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
# wandb の API キーは ~/.bashrc の export 行から読む（キーはログに出さない）
eval "$(grep '^export WANDB_API_KEY=' ~/.bashrc)"
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
      data.balance_dataset_weights=$BALANCE \
      "runner.logger.logger_backends=[tensorboard,wandb]" \
      runner.logger.project_name=libero_RL runner.logger.experiment_name=$EXP_NAME \
      "$@" > $out 2>&1 )
}
START=${START:-$(date +%s)}
# 整理・再開の対象はこの学習の保存先だけ（他の実験の checkpoint は絶対に触らない）。引き継ぎ時は RUN_DIR を指定する
RUN_DIR=${RUN_DIR:-'*-libero10_task0_recap_smolvla'}
ckpts() { ls -d $LOGS/$RUN_DIR/$EXP_NAME/checkpoints/global_step_* 2>/dev/null | while read d; do [ $(stat -c %Y "$d") -ge $START ] && echo "${d##*_} $d"; done | sort -n; }
latest_ckpt() { ckpts | tail -1 | cut -d' ' -f2-; }
prune() { ckpts | head -n -2 | cut -d' ' -f2- | while read d; do rm -rf "$d"; log "pruned $d"; done; }
log "start: resume from $BASE_CKPT, balance_dataset_weights=$BALANCE, exp=$EXP_NAME, MB=$MB SAVE=$SAVE max_steps=$MAX_STEPS"
for attempt in 1 2 3 4 5 6; do
  if [ -n "$ADOPT_PID" ] && [ "$attempt" = 1 ]; then
    P=$ADOPT_PID; log "実行中の学習(pid $P)を引き継ぐ"
  else
    R=$(latest_ckpt); [ -z "$R" ] && R=$BASE_CKPT
    log "attempt $attempt resume=$R"
    run_step4 tmp/step4_resume_attempt$attempt.log "+runner.resume_dir=$(realpath $R)" &
    P=$!
  fi
  while kill -0 $P 2>/dev/null; do prune; sleep 120; done
  R=$(latest_ckpt); S=${R##*_}
  if [ "${S:-0}" -ge $MAX_STEPS ]; then log "DONE Step4 step $S ckpt=$R"; break; fi
  log "プロセスが終了（ckpt ${S:-none}）。再開する"
done
R=$(latest_ckpt); S=${R##*_}
[ "${S:-0}" -ge $MAX_STEPS ] || { log "ABORT: Step4 が完了しなかった"; exit 1; }
W=$ROOT/$R/actor/model_state_dict/full_weights.pt
[ -f "$W" ] || { log "ABORT: $W がない"; exit 1; }
W_LOG=tmp/step9_eval_${EXP_NAME}.log
log "Step9 評価 開始 ckpt=$W (${N_EVAL} 回 x 50 試行)"
: > $W_LOG
for k in $(seq 1 $N_EVAL); do
  ( source RLinf/.venv-smolvla/bin/activate; cd RLinf
    bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
      rollout.model.model_path=$ROOT/models/smolvla_libero runner.ckpt_path=$W \
      rollout.model.smolvla.advantage_prompt=true \
      rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
      "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=10 env.eval.max_steps_per_rollout_epoch=2600 ) >> $W_LOG 2>&1
  log "Step9 評価 $k/$N_EVAL 終了: $(tr '\r' '\n' < $W_LOG | grep -oE "'eval/success_once': array\([0-9.]+" | tail -1)"
done
log "ALL DONE （ベースライン 0.40、1000 ステップ 0.46、2000 ステップ 0.44）"
