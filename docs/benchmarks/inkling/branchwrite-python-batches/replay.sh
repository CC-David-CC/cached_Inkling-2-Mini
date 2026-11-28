#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cp -r "$here/source" "$work/source"
chmod 0664 "$work/source/batches.py" "$work/source/check.py"
baseline=0
before=$(python3 -B "$here/grade.py" "$work/source") || baseline=$?
[[ $baseline == 1 && $before == 63/171 ]]
branchwrite apply "$here/fix.bw" --source "$work/source" --destination "$work/repaired" \
    --sha256 1dfc8b1996179b9b0a57507581379b97fd1e0e0ba437bd21d31e79dcdb72622f
cmp "$work/source/check.py" "$work/repaired/check.py"
python3 -B "$work/repaired/check.py"
python3 -B "$here/grade.py" "$work/repaired"
