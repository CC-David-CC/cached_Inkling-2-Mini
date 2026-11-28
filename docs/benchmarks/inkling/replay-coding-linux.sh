#!/usr/bin/env bash
# Replay the committed Inkling coding and service repairs on a Linux user session.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
base=docs/benchmarks/inkling
service="$base/branchwrite-systemd-repair"
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
git -C "$repo" archive HEAD "$base/branchwrite-python-intervals" "$base/branchwrite-python-batches" "$base/branchwrite-js-retry" "$service" | tar -x -C "$work"
cd "$work"
bash "$base/branchwrite-python-batches/replay.sh"
bash "$base/branchwrite-python-intervals/replay.sh"
bash "$base/branchwrite-js-retry/replay.sh"
bash "$service/replay.sh"
git apply "$service/timeout/fixture.patch"
bash "$service/replay.sh"
git apply "$service/timeout/transfer-1375.patch"
bash "$service/replay.sh"
