# Follow-up framing repair

The candidate received the unmodified encoder file from Cached commit b42d5276a43f631f3ae7e475b5ea31983278a443 and a visible two-assertion Python check. It could edit only the encoder. The task was to close a text-stopped assistant frame before the next question while preserving native EOS and invalid-state behavior. Model inference ran on llm-60. The operator ran the historical fix from f74b335 separately as a reference.

The independent comparison executes the actual encoder class with a stub tokenizer and question suffix. It compares returned token sequences and exception types over 60 combinations of controller state, stop metadata, final tokens and input type. The two visible assertions exercise behaviors represented in the matrix. This is a method-level differential test. It does not exercise model generation, HTTP or KV restore.

To replay the compact candidate's visible test from the repository root in a disposable directory:

```sh
repo=$PWD
fixture=$repo/docs/benchmarks/inkling/branchwrite-harness/frame-repair
work=$(mktemp -d)
mkdir "$work/serve"
git show b42d527:serve/cow_inkling_prompt.py > "$work/serve/cow_inkling_prompt.py"
(cd "$work" && git apply "$fixture/compact.patch")
(cd "$work/serve" && python3 "$fixture/check.py")
```

| Controller | Workflow | Independent checks | Newly correct | Previously passing cases lost |
|---|---|---:|---:|---:|
| Original source | Before repair | 58/60 | — | — |
| Minimal | Step budget exhausted | 58/60 | 0 | 0 |
| Compact | Verified by visible test | 59/60 | 2 | 1 |
| Minimal with explicit file and command scope in task text | Step budget exhausted | 56/60 | 0 | 2 |

Each attempt had a 160-step interpreter budget, direct greedy generation, and a fresh workspace. The first two used identical task text. The compact candidate allowed a text stop with empty token history. The minimal candidate repeatedly searched the sandbox and left the source unchanged. The scoped candidate edited the function but failed the visible test. These are single attempts on a historical issue whose model-training exposure is unknown.
