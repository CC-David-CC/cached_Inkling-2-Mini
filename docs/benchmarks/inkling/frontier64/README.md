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

Run `python3 "$bench/ranking.py"` in the same output directory to compare the final next-token rankings and probabilities at temperature 1 with no sampling filters. All four runs assigned token 149884 more than 99.99999% probability. Comparing undrafted to drafted workers, seven of the top ten tokens kept the same rank, and nine or ten appeared in both top-ten lists. The largest total variation distance was 1.77 × 10⁻⁸. This measures one highly predictable position at the end of the register continuation. [Pairwise ranking results](../dspark-64k-ranking.csv)

For the short coding probe, run `python3 "$bench/coding.py" "$runtime" "$trial"` once per trial under the same environment, using a Python installation with the tokenizers package. The prompt asks for a stable topological sort. Each worker generates eight consecutive 16-token blocks and records the distribution after each block. At 96 generated tokens the winning token had 36.1% probability undrafted and 37.7% with DSpark. All four 128-token continuations were identical. The probe measures decoding agreement on partial code. [Coding frontier records](../dspark-code-frontiers.jsonl)
