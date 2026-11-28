#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cp -r "$here/source" "$work/source"
chmod 0664 "$work/source/intervals.py" "$work/source/check.py"
status=0
baseline=$(python3 -B "$here/grade.py" "$work/source") || status=$?
[[ $status == 1 && $baseline == '{"correct": 118, "incorrect": 993}' ]]
branchwrite apply "$here/fix.bw" --source "$work/source" --destination "$work/repaired" \
    --sha256 8081a2393c3c28ab5f955ca59698b28f89d84421beba518e703690175965bd50
cmp "$work/source/check.py" "$work/repaired/check.py"
python3 -B "$work/repaired/check.py"
python3 -B "$here/grade.py" "$work/repaired"
