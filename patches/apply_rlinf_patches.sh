#!/usr/bin/env bash
# Apply (or revert) the libero_RL patches to the RLinf submodule.
#
# Usage:
#   bash patches/apply_rlinf_patches.sh           # apply patches/rlinf/*.patch
#   bash patches/apply_rlinf_patches.sh --revert  # undo them
#
# Patches were made against RLinf commit $BASE_COMMIT. Applying is idempotent:
# already-applied patches are skipped.

set -euo pipefail

BASE_COMMIT="d1bf37cdc73bba24218d523b08c1c88fcffd9deb"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rlinf"
RLINF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/RLinf"

mode="apply"
if [ "${1:-}" = "--revert" ]; then
    mode="revert"
fi

head_commit="$(git -C "$RLINF_DIR" rev-parse HEAD)"
if [ "$head_commit" != "$BASE_COMMIT" ]; then
    echo "warning: RLinf is at $head_commit, patches were made for $BASE_COMMIT" >&2
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [ "$mode" = "revert" ]; then
    # Revert in reverse order.
    for ((i = ${#patches[@]} - 1; i >= 0; i--)); do
        p="${patches[$i]}"
        if git -C "$RLINF_DIR" apply --reverse --check "$p" 2>/dev/null; then
            git -C "$RLINF_DIR" apply --reverse "$p"
            echo "reverted: $(basename "$p")"
        else
            echo "not applied, skipped: $(basename "$p")"
        fi
    done
    exit 0
fi

unclear=()
for p in "${patches[@]}"; do
    if git -C "$RLINF_DIR" apply --reverse --check "$p" 2>/dev/null; then
        echo "already applied, skipped: $(basename "$p")"
    elif git -C "$RLINF_DIR" apply --check "$p" 2>/dev/null; then
        git -C "$RLINF_DIR" apply "$p"
        echo "applied: $(basename "$p")"
    else
        # Neither direction applies: normally because a LATER patch in this directory changes the same
        # lines (stacked patches), i.e. this one is already part of the tree. On a fresh checkout every
        # patch applies in order, so this only happens when re-running on a patched tree.
        echo "skipped (superseded by a later patch, or conflict): $(basename "$p")"
        unclear+=("$(basename "$p")")
    fi
done
if [ "${#unclear[@]}" -gt 0 ]; then
    echo "note: ${#unclear[@]} patch(es) could be neither applied nor reverted: ${unclear[*]}" >&2
    echo "      expected for stacked patches on an already patched tree; on a fresh RLinf checkout this means a conflict." >&2
fi
