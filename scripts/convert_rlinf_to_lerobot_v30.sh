#!/usr/bin/env bash
# Make a LeRobot v3.0 copy of an RLinf RECAP dataset (LeRobot v2.0 / v2.1) for SmolVLA training.
#
# Usage: bash scripts/convert_rlinf_to_lerobot_v30.sh <src_dataset_dir> <dst_dataset_dir>
#
#   1. copy src -> dst (data/ and videos/ are hard-linked, meta/ is copied; src is not modified)
#   2. v2.0 only: v2.0 -> v2.1 with LeRobot 0.3.3 (scripts/convert_rlinf_v20_to_v21.py)
#   3. v2.1 -> v3.0 with LeRobot's converter (lerobot.scripts.convert_dataset_v21_to_v30)
# Episode and frame indices are preserved, so RECAP advantage files of src still apply.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <src_dataset_dir> <dst_dataset_dir>" >&2
    exit 1
fi
SRC="$(cd "$1" && pwd)"
DST="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -e "$DST" ]; then
    echo "$DST already exists" >&2
    exit 1
fi
mkdir -p "$(dirname "$DST")"
DST="$(cd "$(dirname "$DST")" && pwd)/$(basename "$DST")"

echo "[1/3] copy $SRC -> $DST"
mkdir -p "$DST"
if [ "$(uname)" = "Darwin" ]; then
    cp -Rc "$SRC"/data "$SRC"/videos "$DST"/   # APFS clone (copy-on-write)
else
    cp -al "$SRC"/data "$SRC"/videos "$DST"/   # hard links
fi
cp -R "$SRC"/meta "$DST"/

version="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$DST/meta/info.json")"
if [ "$version" = "v2.0" ]; then
    echo "[2/3] v2.0 -> v2.1"
    (cd "$ROOT" && pixi run -e lerobot-v21 python scripts/convert_rlinf_v20_to_v21.py --root "$DST")
else
    echo "[2/3] already $version"
fi

echo "[3/3] v2.1 -> v3.0"
(cd "$ROOT" && pixi run -e lerobot python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id "local/$(basename "$DST")" --root "$DST" --push-to-hub false)
rm -rf "${DST}_old"
echo "done: $DST"
