#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cp -r "$here/source" "$work/source"
chmod 0664 "$work/source/check.js" "$work/source/retry.js"
status=0
baseline=$(node "$here/independent.cjs" "$work/source/retry.js") || status=$?
[[ $status == 1 && $baseline == '{"count":200,"failed":186,"passed":14}' ]]
branchwrite apply "$here/fix.bw" --source "$work/source" --destination "$work/repaired" \
    --sha256 c5b3cad074f0100fa34d422e29da20978c7ad9ec77c8468d9c0da908139fd12c
cmp "$work/source/check.js" "$work/repaired/check.js"
node "$work/repaired/check.js"
node "$here/independent.cjs" "$work/repaired/retry.js"
