# Clean portable CUDA build on llm-60

Cached revision `5a9c62d` and a fresh fetch of Strata revision `b28ca95e3e6847f2d5e317994f253b66716deeaa` passed the README's PCC CUDA checkpoint regression on one RTX PRO 6000. The test reused the installed CUDA 13.5 toolkit, SQ-20 weights and pinned tokenizer assets. It built a new portable bridge and created a fresh CPython 3.14.4 environment.

| Check | Observed result |
|---|---|
| Independent write process | RAM and sibling checks passed with a 4,097-token prefix. Disk payload was 306,656,744 bytes. |
| Independent read process | Exact full-logit frontier and continuation, rejection checks and zero prompt replay passed. Reported restore time was 0.379 seconds in one run. |
| Bridge SHA-256 | `59558230750f88b93e2493484988dba5d66a91354368de816641cf83871bf3e3` |
| HTTP startup | All 3,072 experts loaded on the GPU. The health endpoint returned OK and authenticated GET /cow/status returned worker and cache statistics. |

For HTTP startup after the build, install the server dependencies in the new environment and supply the pinned chat template. From the Cached checkout, with `runtime` set to the build directory and `template` set to the pinned chat template file, run:

```bash
uv pip install --python "$runtime/.venv/bin/python" -r requirements-cow-server.txt
mkdir -p "$runtime/tokenizer"
cp "$template" "$runtime/tokenizer/chat_template.jinja"
"$runtime/.venv/bin/python" tools/inkling_pcc_serve.py "$runtime" --require-gpu-experts --port 18938
```

This run exercised inference through the native checkpoint test. The HTTP checks covered startup and status. The host already had a working GPU driver and CUDA compiler.

### Cross-build logit comparison

Using the clean checkout's Python backend, load each native bridge in turn with SQ-20, one GPU, default capacity 8,192, and drafting off. Consume 4,097 token IDs given by `100 + i % 1000` starting at zero, capture the complete next-token logit row, then consume token 197 and capture the next row. Each row contains 200,058 float32 values. Comparing their little-endian bytes found zero differences at either position. [Measured row digests](cross-build-logits.json)

With DSpark enabled on both builds, the same 4,097-token prefix yielded identical initial logit digests, 64 identical generated token IDs, and identical final logit digests. Each build counted 68 draft nodes and 28 accepted drafts. These are cross-build measurements with drafting enabled on both sides. [Token sequences, counters and digests](cross-build-dspark.json)

On the clean build, an independent undrafted greedy run produced the same 64 token IDs as DSpark. The hash of the complete 200,058-logit row also matched before and after generation. Intermediate logit rows were not captured. [Undrafted control](dspark-vs-greedy-4097.json)
