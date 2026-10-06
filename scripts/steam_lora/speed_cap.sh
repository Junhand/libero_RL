#!/usr/bin/env bash
# Speed test without the final checkpoint write: run with a large max_steps, stop (by PID) after >= 32 logged steps.
# Usage: ./speed_cap.sh <name> [hydra overrides...]
NAME=$1; shift
cd /workspace/libero_RL/tmp/steam_lora
./run_variant.sh $NAME 1000 "$@" > $NAME.log 2>&1 &
RUNPID=$!
for i in $(seq 1 120); do                       # up to ~30 min
  sleep 15
  N=$(/workspace/libero_RL/RLinf/.venv/bin/python - <<PY
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
fs = sorted(glob.glob("/workspace/libero_RL/tmp/rlinf_steam_dev/logs/steam_sft/*/tensorboard"), key=os.path.getmtime)
n = 0
if fs:
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0}); ea.Reload()
    if "time/step" in ea.Tags()["scalars"]: n = len(ea.Scalars("time/step"))
print(n)
PY
)
  if [ "${N:-0}" -ge 32 ]; then break; fi
done
echo "[$(date +%T)] $NAME: $N steps logged; stopping"
for p in $(pgrep -f "rlinf_steam_dev/examples/offline_rl/advantage_labeling/steam/train_steam"); do kill $p; done
kill $RUNPID 2>/dev/null
sleep 5; source /workspace/libero_RL/RLinf/.venv/bin/activate; ray stop --force > /dev/null 2>&1
echo "[$(date +%T)] stopped"
