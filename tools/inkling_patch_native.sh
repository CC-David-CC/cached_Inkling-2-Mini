#!/usr/bin/env bash
# Prepare a pinned, clean Strata checkout for one Cached bridge profile.
set -euo pipefail
source_dir=${1:?Usage: inkling_patch_native.sh STRATA_CHECKOUT portable|host-kv|suffix}
profile=${2:?Select portable, host-kv or suffix}
patch_dir=$(cd "$(dirname "$0")/../experimental/branching_context/native/inkling" && pwd)
patches=(portable-state dspark-checkpoint layer-split-checkpoint)
case "$profile" in
    portable) ;;
    host-kv) patches+=(host-kv-checkpoint) ;;
    suffix) patches+=(parent-suffix) ;;
    *) echo "Unknown bridge profile: $profile" >&2; exit 2 ;;
esac
if [[ $(git -C "$source_dir" rev-parse HEAD) != b28ca95e3e6847f2d5e317994f253b66716deeaa ]]; then
    echo 'Expected the pinned Strata revision.' >&2
    exit 1
fi
files=()
for patch in "${patches[@]}"; do
    files+=("$patch_dir/$patch.patch")
done
# A single apply checks the complete patch sequence before changing the worktree.
git -C "$source_dir" apply "${files[@]}"
