#!/usr/bin/env bash
# Usage: bash replay-feedback.sh PATH_TO_SSS PATH_TO_COMPLETED_BATCH_LIMIT_RUN
# Use a Python environment with BranchWrite installed for the real ledger check.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source_repo=$(realpath "${1:?supply the S³ repository path}")
ledger=$(realpath "${2:?supply the completed batch-limit run directory}")
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
file=sss_integration/bw_service_recovery.py
mkdir -p "$work/sss_integration"
git -C "$source_repo" show "4c3e12b:$file" > "$work/$file"
(cd "$work" && git apply "$here/feedback-round-3.patch")
python3 -B "$here/grade.py" "$work/$file" "$ledger"
python3 -B "$here/report_cases.py" "$work"
