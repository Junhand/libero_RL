#!/bin/bash
# FPO smoke / run script (development copy of RLinf). Usage: EPOCHS=<updates> ./run_fpo.sh [hydra overrides...]
ROOT=/workspace/libero_RL; DEV=${DEV:-$ROOT/tmp/rlinf_dev}; TAG=${TAG:-main}
cd $DEV && source $ROOT/RLinf/.venv-smolvla/bin/activate
export EMBODIED_PATH=$DEV/examples/embodiment REPO_PATH=$DEV PYTHONPATH=$DEV
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true WANDB__DISABLE_STATS=true WANDB_SILENT=true
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export RAY_memory_usage_threshold=0.995  # cgroup usage includes reclaimable page cache (~3GB)
export TMPDIR=$ROOT/tmp/sl_$TAG; mkdir -p $TMPDIR
LOG=$ROOT/tmp/hybrid_fpo/logs/${TAG}_run_$(date +%m%d_%H%M%S); mkdir -p $LOG
python $EMBODIED_PATH/train_embodied_agent.py --config-path $EMBODIED_PATH/config --config-name libero_10_smolvla_fpo \
  runner.logger.log_path=$LOG runner.max_epochs=${EPOCHS:-2} \
  actor.model.model_path=$ROOT/models/smolvla_libero actor.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
  "$@" > $LOG/run.log 2>&1
echo "exit=$?" >> $LOG/run.log
echo $LOG > $ROOT/tmp/hybrid_fpo/last_log_dir_$TAG
