# Repeat the 64k full-logit comparison

Use an installed SQ-10 native runtime with DSpark support and the matching Cached Python backend. The compressed fixture contains the exact first 65,536 little-endian uint32 input IDs from the station register. Each worker verifies its SHA-256 before prefill. Run from a fresh output directory with enough space for eight 800,232-byte score rows.

```bash
set -e
repo=/absolute/path/to/cached_Inkling-2-Mini
runtime=/absolute/path/to/prepared/runtime
bench="$repo/docs/benchmarks/inkling/frontier64"
export PYTHONPATH="$repo/experimental/branching_context/src"
export INKLING_EXPERT_VRAM_MIB=4096 STRATA_INK_EXACT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for trial in off1 off2 dspark1 dspark2; do
    python3 "$bench/worker.py" "$runtime" "$trial"
done
python3 "$bench/compare.py" > comparisons.csv
```

The worker measures prefill and generation separately, outside the score capture calls. The comparison checks every float32 bit pattern and reports the largest absolute score difference and the winning token at each frontier (as a pairwise equality flag). Generated-token equality compares hashes over all 512 token IDs. DSpark timing and score differences can vary between runs.
