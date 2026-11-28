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

## Verified client installation

The replay passed with CPython 3.14.4 and a fresh, non-editable BranchWrite installation from commit `9d1d3528d9d357d9d9f1d42decacad25881e9172`. To install that CLI in an isolated environment before running the replay commands above, use:

```bash
client=$(mktemp -d)
git clone https://github.com/CC-David-CC/BranchWrite.git "$client/source"
git -C "$client/source" checkout --detach 9d1d3528d9d357d9d9f1d42decacad25881e9172
uv venv --python 3.14 "$client/venv"
uv pip install --python "$client/venv/bin/python" "$client/source"
export PATH="$client/venv/bin:$PATH"
```
