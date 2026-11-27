# Cached Inkling 2 Mini

This local PCC fork targets Thinking Machines Lab's Inkling 2 Mini. The
initial profile uses one RTX PRO 6000 Blackwell, one request and an 8,192-token
context. GPU count is a runtime setting for later trials.

## Native inference baseline

Use the Linux x86-64 CUDA 13 `strata-inkling` 0.1.69 package from
[Strata 0.2.22](https://github.com/Niko1221/Strata/releases/tag/v0.2.22) and
`Inkling-2-Mini-SQ-20.strata` from the
[pinned checkpoint revision](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/tree/04996a1e2c45009b6a04f2af126cb6bdb2ba3587).
Extract the engine archive and place the checkpoint beside the executable. The
checkpoint uses about 18 GiB. On llm-60 both belong under
`/mnt/fast_cache/inkling`.

```bash
INKLING_DIR=/mnt/fast_cache/inkling tools/inkling_smoke.sh
```

The baseline uses FP16 conversation state and disables speculative decoding.

## PCC integration

The next integration step is to determine which native state operations
`strata-inkling` exposes and wire them into PCC. Inkling's loop layers keep
separate KV and convolution state for each pass. The first PCC milestone is
text continuation from a persisted checkpoint. Compare full next-token logits
against direct execution, verify strict restore in a fresh process with zero
prefix replay, and verify that writing to a sibling branch preserves its
parent's state. Physical prefix sharing needs its own evidence.

The [upstream README](README-PCC.md) records earlier PCC model results. Those
results apply to their documented profiles.
