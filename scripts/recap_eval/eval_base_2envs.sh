#!/usr/bin/env bash
# ベースライン(smolvla_libero)の評価。2 環境 x 25 周、動画あり、ノイズシード固定(動画の撮り直しで同じ結果になるように)
cd /workspace/libero_RL/RLinf && source .venv-smolvla/bin/activate
HF_HUB_OFFLINE=1 bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
  rollout.model.model_path=/workspace/libero_RL/models/smolvla_libero \
  rollout.model.smolvla.advantage_prompt=false rollout.model.smolvla.noise_seed=0 \
  rollout.model.smolvla.vlm_model_name=/workspace/libero_RL/models/SmolVLM2-500M-Video-Instruct \
  "+env.eval.task_id_filter=[0]" env.eval.total_num_envs=2 env.eval.max_steps_per_rollout_epoch=13000
