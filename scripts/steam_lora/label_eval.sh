#!/usr/bin/env bash
# Label the eval rollouts (eval_v2_m0..m2, 64 episodes) with a STEAM critic. Usage: ./label_eval.sh <value_checkpoint_actor_dir> <tag>
CK=${1:?checkpoint}; TAG=${2:?tag}
DEV=/workspace/libero_RL/tmp/rlinf_steam_dev; SH=/workspace/libero_RL/tmp/steam_lora/shadow_eval
cd $DEV && source /workspace/libero_RL/RLinf/.venv/bin/activate
export REPO_PATH=$DEV OFFLINE_RL_CONFIG=$DEV/examples/offline_rl/config PYTHONPATH=$DEV:${PYTHONPATH:-}
export TMPDIR=/workspace/libero_RL/tmp/sl HF_HUB_OFFLINE=1 RAY_memory_usage_threshold=0.98; mkdir -p $TMPDIR
cd $DEV/examples/offline_rl/advantage_labeling/steam/process
python compute_advantages_ensemble.py --config-path $OFFLINE_RL_CONFIG --config-name steam_compute_advantages_ensemble \
  "data.train_data_paths=[{dataset_path:$SH/eval_v2_m0,type:rollout},{dataset_path:$SH/eval_v2_m1,type:rollout},{dataset_path:$SH/eval_v2_m2,type:rollout}]" \
  "data.camera_keys=[image,wrist_image]" data.k=32 \
  advantage.value_checkpoint=$CK advantage.batch_size=16 advantage.label_mode=quantile \
  advantage.rollout_quantile=0.3 advantage.expert_quantile=null advantage.tag=$TAG \
  advantage.num_dataloader_workers_per_gpu=4 2> >(grep -v --line-buffered libdav1d >&2)
echo "exit=$?" > /workspace/libero_RL/tmp/steam_lora/label_$TAG.exit
