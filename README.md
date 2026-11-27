# Cached Inkling 2 Mini

This local PCC fork targets Thinking Machines Lab's Inkling 2 Mini. The
initial profile uses one RTX PRO 6000 Blackwell, one request and an 8,192-token
context. GPU count is a runtime setting for later trials.

[Training, serving and cache qualification matrix](docs/benchmarks/inkling/qualification.md)

## Measured performance

### Tesla P4 independent streams and layer splitting

![Per-stream prefill and generation on three Tesla P4s](docs/benchmarks/inkling/p4-topology.png)

Each process used SQ-10, greedy native decoding, FP16 device KV and drafting off. The three one-card workers started concurrently. Each combined run served a single stream and placed every expert on the selected GPUs. Each bar represents one 512-token prompt and 128 generated tokens. [Download the rates](docs/benchmarks/inkling/p4-topology.csv) and [see the separate 64K capacity check](docs/benchmarks/inkling/p4-capacity.csv).

At 64K, three SQ-10 workers with all experts in host RAM decoded 32 steps each after a synchronized start from synthetic 65,504-token FP16 KV states. Across the common wall-clock interval the three cards produced **15.21 output tok/s combined**. The interval includes native decode, finite-logit checks and greedy selection. Real prompt prefill is outside this measurement. [Per-stream timing](docs/benchmarks/inkling/p4-64k-decode.csv)

### RTX 4090 and RTX PRO 6000 Blackwell

![Cached single-request prefill, generation, full HTTP throughput and first-text latency on the RTX 4090 and RTX PRO 6000](docs/benchmarks/inkling/gpu-comparison.png)

Each bar averages two single-request transcription trials with a fresh branch, 1,024 greedy output tokens, SQ-20 weights, FP16 KV, an 8,192-token capacity, 512-token prefill chunks and a 5 GiB RAM checkpoint pool. Both hosts kept all 3,072 experts on the GPU. Each drafting mode started a fresh server. The native prefill timer covers the 3,045 or 6,045 tokens consumed during branch creation. The client timer starts before branch creation and ends after the streamed completion receipt. Existing services also held VRAM on the RTX PRO 6000. These are single-request measurements on the two hosts.

[Download individual timings and generated-token hashes](docs/benchmarks/inkling/gpu-comparison.csv)

### 64K and 128K single requests

Each bar measures one cold request on one RTX PRO 6000 Blackwell, with SQ-20 weights, FP16 KV, a 512-token prefill chunk and a 1,024-token output. Each configuration starts a fresh server. Time to first visible token includes prompt processing. Peak GPU memory is sampled across the serving process and its children. The Cached client also times root creation and branch completion in the complete operation. All eight runs returned the same output text. Repeated questions on a shared parent require a separate warm-cache measurement.

![64K and 128K single-request GPU memory, time to first token, generation speed and complete request speed](docs/benchmarks/inkling/long-context.png)

[Download the underlying long-context measurements](docs/benchmarks/inkling/long-context.csv)

The client also timestamped first text, last text and the complete receipt. At 128K with DSpark the last text arrived 16.3 seconds before the receipt. The server separately measured 3.8 seconds inside its capture timer. A separate instrumented repeat measured 16.2 seconds from last text to receipt. Three successive server operations accounted for 14.3 seconds of that interval: checkpoint export (6.80 s), checkpoint preparation (3.76 s), and publication to the temporary RAM-backed filesystem (3.75 s). The 4.94 s native export is included in the export total. The 1.48 s payload write is included in publication. The parent restore counter showed zero imported bytes. After removing a full host-side payload copy, one matched repeat reduced native child export from 4.94 to 1.77 seconds and last-text-to-receipt time from 16.23 to 12.91 seconds. Both runs returned the same text.

![Elapsed time before, during and after visible output in cold Cached requests](docs/benchmarks/inkling/cached-request-phases.png)

[Download the phase and capture timings](docs/benchmarks/inkling/cached-request-phases.csv)

Stock Strata reported prompt processing speed and the number of DSpark draft tokens offered and accepted. Cached repeated the same workload after accepted-node accounting was added. Both engines produced identical output text at each length.

![Stock Strata prefill speed and DSpark acceptance for 64K and 128K inputs](docs/benchmarks/inkling/stock-dspark.png)

[Download the prefill and proposal counts](docs/benchmarks/inkling/stock-dspark.csv)

![Accepted and proposed DSpark nodes by engine and context length](docs/benchmarks/inkling/dspark-acceptance.png)

[Download the cross-engine proposal counts](docs/benchmarks/inkling/dspark-acceptance.csv)

### Three questions on one 128K document

BranchWrite loaded one station register, forked three siblings from the same parent and asked each a different lookup question. All three asks restored the resident parent and imported zero checkpoint bytes. Two answers were correct. The bars time parent loading and each ask through answer and receipt verification. Fork creation is outside the ask timer. Each bar is one observation.

![Parent loading and subsequent lookup times on one resident 128K document](docs/benchmarks/inkling/resident-128k.png)

[Download the shared-parent measurements](docs/benchmarks/inkling/resident-128k.csv)

An offline delta-storage experiment used the saved checkpoints from these same three children. Each 4.47 GB child reconstructed from the parent plus a 12.41 MB rsync delta, with its full SHA-256 verified. Applying each delta to a parent copy in RAM took 5.58–5.76 seconds, excluding the initial parent copy and final hash check. The live server still exports full child payloads.

![Cumulative checkpoint storage for one parent and zero to three branches](docs/benchmarks/inkling/shared-parent-storage.png)

[Download the cumulative payload sizes](docs/benchmarks/inkling/shared-parent-storage.csv)

### 3K and 6K single requests

One NVIDIA RTX PRO 6000 Blackwell, SQ-20 weights resident on the GPU, FP16 KV.
Each short-context HTTP throughput bar summarizes four single-request transcription trials
with 1,024 output tokens. The complete Cached request includes root creation
and the branch receipt. The generation panel uses each engine’s reported
generation time. Stock used a 2,048-token prefill chunk and Cached used 512.

![Stock and Cached HTTP throughput with and without DSpark on 3K and 6K prompts](docs/benchmarks/inkling/http-throughput.png)

The latency chart shows each observed request, the median and an interpolated p95. Its TPOT measure is engine generation time divided by 1,024 output tokens. Each group contains four trials, including the first. With four observations, the p95 is descriptive only and gives little evidence about production tail latency. Each of the 32 transcription requests produced 1,024 visible tokens and zero reasoning tokens. For Cached, decoding every recorded token ID reproduced the returned text byte for byte and no channel markers occurred. Stock reported zero reasoning tokens in its usage counters. [Per-request token counts](docs/benchmarks/inkling/output-channels.csv)

![Individual trials, median and sample p95 for first text, generation milliseconds per token and branch completion](docs/benchmarks/inkling/http-latencies.png)

[Download the latency summaries](docs/benchmarks/inkling/http-latencies.csv)

The following memory measurements sample the serving process and its children
during one request per configuration. Host checkpoint RAM is excluded.

![Sampled peak GPU memory on 3K and 6K transcription requests](docs/benchmarks/inkling/http-vram.png)

### Fresh-process checkpoint restore

Four fresh engine processes restored the same 4,097-token checkpoint with the OS filesystem cache retained. Median restore time was 475 ms. Reading the next-token frontier afterward took a median 155 ms. All 200,058 FP32 logits matched the saved reference byte for byte on every trial, with zero forward tokens. Timers exclude model loading. This measurement used SQ-20, FP16 KV and drafting off.

![Restore and frontier read times for four fresh processes](docs/benchmarks/inkling/checkpoint-restore.png)

[Download the four restore trials](docs/benchmarks/inkling/checkpoint-restore.csv)

### Concurrent requests

The native concurrency sweep used independent 4,097-token prompts and 256
output tokens per slot. A repeat at 38, 54, 56 and 64 slots found 1,093.2 aggregate tokens/s at 64 slots, compared with 643.5 on its first run. Both runs fit in GPU memory. The first sweep’s drop above 54 slots did not reproduce. The original 38-slot memory sample was below the model footprint and is omitted. Its repeat measured 35.44 GiB.

[Download both concurrency runs](docs/benchmarks/inkling/native-concurrency.csv)

![Native throughput, per-slot speed and GPU memory by concurrent slot count](docs/benchmarks/inkling/native-concurrency.png)

### Independent 128K contexts

Eight native slots each processed 131,072 prompt tokens and generated 256 output tokens. All 3,072 experts were GPU-resident with FP16 KV, KV offload disabled and drafting off. At eight slots, aggregate decode reached 196.5 tokens/s and sampled process memory peaked at 55.4 GiB. Each point is one run. The benchmark allocates an independent context per slot.

![Independent 128K slot counts, decode rates and process memory](docs/benchmarks/inkling/native-128k-concurrency.png)

[Download the 128K concurrency measurements](docs/benchmarks/inkling/native-128k-concurrency.csv)

A separate startup probe requested 131,328 tokens per independent slot from 8 through 54 slots. The chart shows the engine’s effective capacity after automatic sizing. Shorter capacities are startup observations. Generation was verified at the full requested length through eight slots. The CSV records GPU memory available before each load.

![Effective context per independent slot as concurrency increases](docs/benchmarks/inkling/native-context-capacity.png)

[Download the context capacity sweep](docs/benchmarks/inkling/native-context-capacity.csv)

### Training on the latest mistakes

A rank-4 AdamW output-head adapter trained for eight steps on three missed lookups, then took one step per round on the previous round’s misses. Each checkpoint answered all 54 questions against the same 128K document. Step 8 fixed two original misses and lost 39 previously correct answers. Step 11 recovered all 51 original successes and fixed two misses, reaching 53/54. On a fresh 24-question register the base scored 22 and the step-11 adapter scored 21. The base remains the serving default.

![Full-answer accuracy and per-round gains and losses](docs/benchmarks/inkling/training-retention.png)

[Download the aggregate training and retention counts](docs/benchmarks/inkling/training-retention.csv)

In a separate experiment, each optimizer trained only on the original three corrections. The second Muon round stayed at 8/54 while fixing six previously wrong answers and losing six previously correct ones. Starting from the first AdamW adapter, adding a second rank-4 update produced a rank-8 sum that scored 15/54. Compressing that sum to rank 4 changed eleven answer texts and lost one correct answer, leaving 14/54.

![Answer gains and losses between successive adapter checkpoints](docs/benchmarks/inkling/adapter-transitions.png)

[Download the paired adapter comparisons](docs/benchmarks/inkling/adapter-transitions.csv)

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

For a single-request HTTP endpoint, also place the pinned Strata checkout at
`strata-source` and the companion tokenizer files from the checkpoint revision
under `tokenizer`. Install `tokenizers`, `jinja2` and `regex` into the runtime environment.

```bash
python tools/inkling_serve.py /mnt/fast_cache/inkling
curl --fail-with-body http://127.0.0.1:18060/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"What is 2 + 2? Reply with just the number."}],"max_tokens":128,"temperature":0}'
```

The HTTP endpoint binds to loopback and serves one request at a time.
For image input, install Pillow and put `Inkling-2-Mini-Vision.strata` from
the same pinned checkpoint revision next to SQ-20. Then run:

```bash
python tools/inkling_serve.py /mnt/fast_cache/inkling --vision
# From a second shell on the GPU host, once the service reports ready:
python tools/test_inkling_http.py
```

The test sends the same question with two different PNGs, verifies the
answers, checks text inference after an invalid model request, and verifies
that disconnecting a stream cancels generation and releases the slot.

## Building the native engine

Build the pinned Strata source with CMake and a CUDA toolkit that supports
SM120. On llm-60 the build uses CUDA 13.5 and writes to the fast-cache
runtime directory beside the released executable.

```bash
git -C /mnt/fast_cache/inkling/strata-source sparse-checkout add engine
tools/inkling_build_native.sh /mnt/fast_cache/inkling
/mnt/fast_cache/inkling/native-build/strata-inkling --version
```

`INKLING_NVCC`, `INKLING_CUDA_ARCHS` and `INKLING_BUILD_JOBS` can override the
compiler path, GPU architecture and build parallelism.

## PCC integration

PCC's Inkling CUDA adapter stores per-visit full-attention and sliding-window
KV, ShortConv state, and a separately stored full-vocabulary frontier. The
pinned profile consumes canonical text token IDs at 8K context with FP16 KV,
512-token prefill chunks, and drafting disabled. The HTTP service defaults to an 8 GiB resident checkpoint budget at 8K context and scales it up for longer contexts. Use `--ram-cache-gib N` to set the budget explicitly. The worker cache also has a 32-entry limit. Disk checkpoints use PCC's existing
checksummed envelope and bind the model, tokenizer, execution profile and
token history. An optional vision profile uses the pinned vision tower and
absolute-position image embedding rows. Its histories also bind the original
image, encoder profile, and exact embedding bytes. The Strata chat baseline
and the PCC token service have separate launchers.

The [upstream README](README-PCC.md) records earlier PCC model results. Those
results apply to their documented profiles.

## Serving performance

Establish a cold and warm single-request baseline on llm-60 before tuning.
Measure prefill rate, time to first visible token, decode rate, peak VRAM and
restore latency across short and long prompts. Track visible and reasoning
tokens separately.

Use exact FP16-state continuation as the reference for RAM and disk cache
tests. Then measure native FP4 kernels and DSpark drafting with the same
weights, prompts, decoding settings and context sizes. Explore selective
layer and expert residency when the full checkpoint exceeds available VRAM.
Compare concurrent requests after the single-request results are reproducible.

### Native checkpoint regression

Apply [the CUDA state patch](experimental/branching_context/native/inkling/portable-state.patch)
to the pinned Strata checkout. Build the standalone regression with
`-DINKLING_PORTABLE_STATE_TEST_SOURCE=/absolute/path/to/check_portable_state.cpp`
in the native CMake build directory. Then run:

```bash
cmake --build native-build --target inkling_portable_state_test
native-build/inkling_portable_state_test Inkling-2-Mini-SQ-20.strata write portable-test.bin
native-build/inkling_portable_state_test Inkling-2-Mini-SQ-20.strata read portable-test.bin
```

On llm-60 the 4,097-token fixture exported 305,856,512 bytes of native state.
RAM restore after a sibling write and disk restore in a fresh process
matched all 200,058 continuation logits bit for bit with CUDA, FP16 KV,
8K context, 512-token prefill chunks and DSpark disabled. The native-only fixture stores unframed test bytes. The PCC regression below
uses the durable checkpoint format.

### PCC CUDA checkpoint regression

For a clean build on an RTX PRO 6000 Blackwell, run these commands from the Cached checkout. Set `assets` to the directory containing the SQ-20 model and the original pinned Hugging Face tokenizer saved as `tokenizer-source.json`. The commands create a new runtime and build directory, apply both patches in order and exercise restore in a second Python process.

```bash
set -e
assets=/mnt/fast_cache/inkling
runtime=$(mktemp -d "$assets/inkling-clean.XXXXXXXX")
bridge="$PWD/experimental/branching_context/native/inkling"
git init -q "$runtime/strata-source"
git -C "$runtime/strata-source" remote add origin https://github.com/Niko1221/Strata
git -C "$runtime/strata-source" fetch --depth=1 origin b28ca95e3e6847f2d5e317994f253b66716deeaa
git -C "$runtime/strata-source" checkout --detach FETCH_HEAD
for patch in portable-state dspark-checkpoint; do
    git -C "$runtime/strata-source" apply "$bridge/$patch.patch"
done
ln -s "$assets/Inkling-2-Mini-SQ-20.strata" "$runtime/"
ln -s "$assets/tokenizer-source.json" "$runtime/"
cmake -S "$runtime/strata-source/engine/src/inkling2" -B "$runtime/native-build" \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_INK_CUDA_ARCHS=120 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DINKLING_PCC_BRIDGE_SOURCE="$bridge/inkling_bridge.cpp"
cmake --build "$runtime/native-build" --target inkling_pcc_bridge --parallel 6
python3 -m venv "$runtime/.venv"
export PYTHONPATH="$PWD/experimental/branching_context/src"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test_dir=$(mktemp -d /dev/shm/inkling-check.XXXXXXXX)
"$runtime/.venv/bin/python" tools/test_inkling_pcc.py "$runtime" "$test_dir" write
"$runtime/.venv/bin/python" tools/test_inkling_pcc.py "$runtime" "$test_dir" read
```

The clean-checkout run at `0e77805` reused the host CUDA installation and model assets. Its fresh process restored the checkpoint in 0.518 seconds. The temporary test store is RAM-backed and can be removed after checking the results.

The two-process regression passed on llm-60 with the pinned Strata source
`b28ca95e3e6847f2d5e317994f253b66716deeaa` and CUDA 13.5. RAM and disk
restore matched all 200,058 frontier logits and all 200,058 logits after one
continuation token. Restore consumed zero prefix tokens. A sibling write
preserved the parent. The test rejected changed model, execution and input
identities with valid manifest checksums, and rejected a flipped payload byte.
The 306,656,744-byte disk payload restored in 0.530 seconds in that run.
PCC's G0 reference suite passed 58 tests. Coverage here is the single-GPU,
text-token, draft-off profile with owned copies of the native state.

For the image-aware profile, place the pinned vision tower alongside the
text checkpoint and run the vision regression in a separate workspace:

```bash
test_dir=$(mktemp -d "$runtime/pcc-vision.XXXXXXXX")
python tools/test_inkling_pcc_vision.py "$runtime" "$test_dir" write
python tools/test_inkling_pcc_vision.py "$runtime" "$test_dir" read
```

On llm-60, the two images used identical text and placeholder token IDs and
produced distinct history IDs and frontier logits. Each image matched all
200,058 frontier and continuation logits through RAM restore after sibling
writes and disk restore in a fresh process. The native work counter recorded
zero replay. The fresh-process test had no image encoding step.

### PCC HTTP token and chat service

After building the PCC bridge, start the authenticated localhost service
with a private, temporary store under Linux `/dev/shm`:

```bash
python tools/inkling_pcc_serve.py /mnt/fast_cache/inkling
```

The temporary store holds checkpoint files and branch metadata in tmpfs and is removed on normal shutdown. This still uses RAM for serialized checkpoints in addition to resident cache entries. For restart and disk durability testing, pass `--persistent` to reuse the profile’s runtime directory or `--store PATH` to select a directory. Use the same option on both launches.

The launcher creates a private `pcc-http.key` in the runtime directory.
The service listens on port 18061 by default. It offers `/cow/create` and
`/cow/generate` for token IDs, and `/cow/text/create` and
`/cow/text/generate` for chat messages. Chat uses the pinned Inkling
tokenizer and template, with greedy decoding and thinking disabled. Install
`jinja2` and `regex` in the runtime environment. The separate Strata baseline
serves OpenAI-compatible text and vision requests on port 18060.

Run `tools/test_inkling_pcc_http.py RUNTIME FIXTURE write` against the live
service, then restart it with the same store and run the test in `read` mode.
On llm-60 the restart returned the same five tokens and the same numerical
payload SHA-256. The native counter recorded five forward tokens for the
five generated tokens. The same regression with `--chat` returned the
visible answer `4` before and after restart with the same numerical payload
SHA-256. The counter recorded three generated tokens and two appended
assistant framing tokens.

### PCC HTTP vision service

Start a separate Inkling vision profile with Pillow installed and the pinned
vision tower in the runtime directory.

```bash
python tools/inkling_pcc_serve.py /mnt/fast_cache/inkling --vision --port 18062
```

The default vision store is `pcc-vision-http-store`. Use `/cow/namespace`,
`/cow/images`, `/cow/text/prepare`, and `/cow/text/answer` for a prepared
image question. The service binds image inputs to the normalized bytes,
encoder profile, and embedding digest. It accepts PNG and JPEG uploads and
uses the same private bearer token as the text service. New image questions
currently use a new prepared prompt.

Run `tools/test_inkling_pcc_vision_http.py RUNTIME FIXTURE write`, restart the
vision service with the same store, then run the test in `read` mode.

### Output-head LoRA

The SQ-20 CUDA bridge can capture the frozen model's final normalized
hidden rows and pre-softcap logits. Train rank-4 LoRA factors on these actual
model features with AdamW or factorwise Muon. The same bridge applies the
exported low-rank delta before softcapping logits during serving.

Use CUDA PyTorch and NumPy in a separate training environment. The
qualification run used PyTorch 2.14.1+cu130 and NumPy 2.5.5 on llm-60.

```bash
runtime=/mnt/fast_cache/inkling
python tools/inkling_train_head.py "$runtime" "$runtime/lora-adamw" --optimizer adamw
python tools/inkling_train_head.py "$runtime" "$runtime/lora-muon" --optimizer muon
export PYTHONPATH="$PWD/experimental/branching_context/src"
store=$(mktemp -d "$runtime/pcc-lora-test.XXXXXXXX")
python tools/test_inkling_head_lora.py "$runtime" "$store" \
    "$runtime/lora-adamw/adapter.iklora" "$runtime/lora-muon/adapter.iklora"
```

Resume a saved output-head optimizer checkpoint into a new output directory. Each resumed run writes the updated optimizer checkpoint, serving adapter and receipt. The optimizer is inferred from the checkpoint.

```bash
python tools/inkling_train_head.py "$runtime" "$runtime/lora-resumed" \
    --resume "$runtime/lora-muon/step-2.pt" --additional-steps 1
```

Use `--parent-adapter PATH` to train a fresh rank-4 output-head update over a frozen parent. Pass the same parent on resume. The checkpoint records its SHA-256, and the exported adapter combines both updates up to the native serving limit of rank 16.

Add `--collapse` to export the sum at rank 4 using truncated SVD. The step checkpoint retains the active optimizer and residual factors. Resume it with the same parent and `--collapse` to reproduce the compressed export. To start another round, use that export as the new parent with a fresh optimizer.

Launch PCC with `--adapter /path/to/adapter.iklora` to attach the adapter.
Each adapter gets a distinct model contract and default store. Omitting the
option selects the base model. Keep training outputs and optimizer checkpoints
in local runtime storage.

Both optimizers updated the real model's output-head adapter and matched
the uninterrupted third step bit for bit after reading the two-step
checkpoint into a new optimizer. Training loss across three updates went
from 8.749 to 6.415 for AdamW and from 8.749 to 0.325 for Muon on the two
synthetic examples. These numbers measure the training fixture. Broader
QLoRA targets, fixed-rank consolidation between rounds, and skill evaluation
are separate qualification work.
Both adapters passed the authenticated PCC HTTP launch, root creation and
greedy generation test on llm-60. The endpoint receipts carried the expected
adapter-specific model identity. The native regression also verified exact
base logits after disabling the adapter and refused cross-adapter restores.

### DFlash2 and Splash

[Splash](https://github.com/incoai/splash) is an Apple Silicon engine with
Inkling 2 Mini DFlash2 support. Use its speculative serving and memory-planning design as a reference for
Cached Inkling. Evaluate the Inkling-specific DFlash2 drafter on CUDA for
llm-60 as an opt-in profile. Compare against DSpark and draft-off under matched conditions.
Include draft acceptance, verifier cost, memory overhead, and post-restore
continuation. Qualify the drafter's state and rejection rollback before
enabling it with PCC checkpoints.

### Experimental Cached DSpark text serving

For the single-GPU unadapted greedy text profile, apply
`experimental/branching_context/native/inkling/dspark-checkpoint.patch` after
`portable-state.patch`, then rebuild the PCC bridge from this checkout.
Launch with `INKLING_EXACT_HEAD=1 python tools/inkling_pcc_serve.py RUNTIME --dspark`.
The selected output projection and drafting mode participate in checkpoint identity.
The server commits a full verifier frontier at the end of each speculative block.
The `draft_nodes` counter reports proposed tree nodes. Updated bridges also report `drafts_accepted` for committed proposals under the stock greedy tree-walk accounting rule. Use the complete HTTP
operation, including root creation and branch acknowledgment, to measure
end-to-end throughput.
