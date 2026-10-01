#!/usr/bin/env bash
# Pre-flight check for rollout collection (RLinf + LIBERO + SmolVLA) on one environment.
# /workspace is shared between environments, but /root, /tmp, /usr (apt) and the GPU are per environment,
# so run this on EACH environment before collecting. It only reads; it never installs anything.
#
# Usage: bash scripts/preflight_collect_env.sh [min_mem_gb=24] [min_gpu_free_gb=12]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/RLinf/.venv-smolvla"
PY="$VENV/bin/python"
MIN_MEM_GB="${1:-24}"
MIN_GPU_GB="${2:-12}"
fail=0
ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [FAIL] $*"; fail=$((fail + 1)); }
warn() { echo "  [WARN] $*"; }

PYCHECK='
import importlib, sys
for m in ("torch", "lerobot", "hydra", "ray", "mujoco", "robosuite", "libero.libero"):
    try:
        importlib.import_module(m); print(f"[OK]   import {m}")
    except Exception as e:
        print(f"[FAIL] import {m}: {type(e).__name__}: {str(e)[:100]}")
import torch
if torch.cuda.is_available():
    f, t = torch.cuda.mem_get_info(0)
    print(f"[OK]   GPU {torch.cuda.get_device_name(0)} free {f/1e9:.1f}/{t/1e9:.1f} GB")
else:
    print("[FAIL] CUDA not available")
'

echo "== host: $(hostname) | $(date '+%F %T')"

echo "1) venv interpreter (the symlink target lives under /root, which is NOT shared)"
if pyver=$("$PY" -c 'import sys; print(sys.version.split()[0])' 2>&1); then
    ok "python $pyver"
else
    bad "$VENV/bin/python does not run -> fix: uv python install 3.12.11"
fi

echo "2) python packages and CUDA"
pyout=$("$PY" -c "$PYCHECK" 2>&1)
echo "$pyout" | grep -E '^\[(OK|FAIL)\]' | sed 's/^/  /'
fail=$((fail + $(echo "$pyout" | grep -c '^\[FAIL\]')))

echo "3) LIBERO config (~/.libero/config.yaml is per environment; without it LIBERO asks interactively and dies)"
if [ -f "$HOME/.libero/config.yaml" ]; then
    ok "$HOME/.libero/config.yaml"
else
    bad "missing -> fix: echo n | $PY -c 'import libero.libero'"
fi

echo "4) EGL rendering (system libs are per environment)"
if MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY" -c 'from OpenGL import EGL; import mujoco.egl' >/dev/null 2>&1; then
    ok "EGL import"
else
    bad "EGL import failed -> libegl1 / libopengl0 are probably missing (apt)"
fi

echo "5) HuggingFace access (SmolVLM2 config/tokenizer are fetched from the Hub unless a local path is given)"
echo "  HF_HOME=${HF_HOME:-<unset>}  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
if [ -d "$ROOT/models/SmolVLM2-500M-Video-Instruct" ]; then
    ok "local VLM dir exists: pass rollout.model.smolvla.vlm_model_name=\$ROOT/models/SmolVLM2-500M-Video-Instruct"
else
    warn "no local models/SmolVLM2-500M-Video-Instruct: this environment needs network access to the Hub"
fi
if [ -d "$ROOT/models/smolvla_libero" ]; then ok "models/smolvla_libero"; else bad "models/smolvla_libero missing"; fi

echo "6) memory limit (Ray kills workers at 95% of the container limit; 5 envs peaked at ~22GB)"
lim=""
[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ] && lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
[ -z "$lim" ] && [ -r /sys/fs/cgroup/memory.max ] && lim=$(cat /sys/fs/cgroup/memory.max)
if [ -n "$lim" ] && [ "$lim" != "max" ] && [ "$lim" -lt 1000000000000 ] 2>/dev/null; then
    gb=$(( (lim + 500000000) / 1000000000 ))
    if [ "$gb" -ge "$MIN_MEM_GB" ]; then ok "limit ${gb} GB"; else bad "limit ${gb} GB < ${MIN_MEM_GB} GB -> lower env.eval.total_num_envs"; fi
else
    warn "no cgroup memory limit found ($(free -g | awk '/Mem:/{print $2}') GB host RAM)"
fi

echo "7) stale Ray / cluster env vars (a leftover local Ray would be attached by address=auto)"
if pgrep -f "raylet|gcs_server" >/dev/null 2>&1; then
    bad "Ray processes are running -> fix: ray stop --force   (only if they are yours)"
else
    ok "no Ray processes"
fi
for v in RAY_ADDRESS RLINF_NODE_RANK; do
    if [ -n "${!v:-}" ]; then bad "$v is set (${!v}); unset it for single-node collection"; else ok "$v unset"; fi
done

echo "8) RLinf working tree (shared, so it is identical everywhere; stacked patches cannot be checked one by one)"
if [ -f "$ROOT/RLinf/rlinf/models/embodiment/smolvla/smolvla_action_model.py" ]; then ok "smolvla adapter present"; else bad "smolvla adapter missing -> bash patches/apply_rlinf_patches.sh"; fi
if grep -q "_collect_all_episodes" "$ROOT/RLinf/rlinf/envs/sim/libero/libero_env.py" 2>/dev/null; then ok "collect-eval-pool patch present"; else bad "collect-eval-pool patch missing -> bash patches/apply_rlinf_patches.sh"; fi
echo "  tree fingerprint: $(git -C "$ROOT/RLinf" diff | sha1sum | cut -c1-12)   (must be equal on all environments)"

echo "9) shared filesystem"
df -h "$ROOT" | tail -1 | awk '{print "  fs: "$1"  use "$5}'
t="$ROOT/tmp/.preflight_$$"
if touch "$t" 2>/dev/null; then rm -f "$t"; ok "/workspace is writable"; else bad "/workspace is not writable"; fi

echo
if [ "$fail" -eq 0 ]; then echo "PRE-FLIGHT PASSED on $(hostname)"; else echo "PRE-FLIGHT FAILED: $fail problem(s) on $(hostname)"; fi
exit $((fail > 0))
