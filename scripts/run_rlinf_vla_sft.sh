#!/usr/bin/env bash
# Run RLinf VLA SFT (RLinf/examples/sft/train_vla_sft.py) with Hydra overrides.
#
# Same setup as RLinf/examples/sft/run_vla_sft.sh, which does not forward overrides.
#
# Usage: bash scripts/run_rlinf_vla_sft.sh <config_name> [hydra_overrides...]
# Run inside the RLinf virtual environment.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name> [hydra_overrides...]" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_PATH="$ROOT/RLinf"
export EMBODIED_PATH="$REPO_PATH/examples/sft"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$REPO_PATH:${LIBERO_REPO_PATH:-}:${PYTHONPATH:-}"

CONFIG_NAME="$1"
shift

LOG_DIR="$REPO_PATH/logs/$(date +'%Y%m%d-%H:%M:%S')-$CONFIG_NAME"
mkdir -p "$LOG_DIR"
cmd=(
    python "$EMBODIED_PATH/train_vla_sft.py"
    --config-path "$EMBODIED_PATH/config/"
    --config-name "$CONFIG_NAME"
    "runner.logger.log_path=$LOG_DIR"
    "$@"
)
echo "${cmd[*]}" | tee "$LOG_DIR/run_embodiment.log"
"${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/run_embodiment.log"
