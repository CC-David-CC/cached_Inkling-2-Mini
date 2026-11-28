# Replay the timeout repair

Run from the root of a clean Cached checkout on Linux with a running user systemd manager and BranchWrite on PATH. The commands create a temporary fixture copy, reproduce the startup failure, apply the recorded repair, and probe the live service three times.

```bash
set -e
base=docs/benchmarks/inkling/branchwrite-systemd-repair
work=$(mktemp -d)
git archive HEAD "$base" | tar -x -C "$work"
(cd "$work" && git apply "$base/timeout/fixture.patch" && bash "$base/replay.sh")
```

The harness also checks rejection of a wrong archive digest and verifies that applying the valid archive twice preserves the systemd invocation ID. Replay uses the saved model output and requires no inference endpoint.
