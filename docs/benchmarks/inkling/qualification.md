# Inkling 2 Mini qualification

Results cover one RTX PRO 6000 Blackwell on llm-60. A pass applies to the profile and workload described in its row. The [benchmark charts](../../../README.md#measured-performance) include sample counts and timings.

| Goal | Status | Verified result and remaining work |
|---|---|---|
| Reproducible install | Partial | A clean public checkout built the pinned CUDA bridge and passed text and vision checkpoint tests in a new virtual environment. Fresh-machine provisioning of CUDA and model assets is open. |
| Text serving | Pass for greedy single requests | PCC HTTP reproduced output tokens and checkpoint hashes after restart with zero saved-prefix replay. |
| Vision serving | Pass for the prepared image path | Red and blue uploads produced distinct answers. Restored image frontiers and one-token continuations matched all 200,058 logits. Broader image tasks remain unqualified. |
| Request lifecycle | Pass for single-worker text | Disconnecting a streaming request preserved its branch and allowed a registered second request to finish identically to its control. Invalid lengths and stale revisions preserved the branch. |
| State exactness and isolation | Pass for pinned FP16 KV | RAM and fresh-process restore matched full logits before and after continuation. Sibling writes preserved the parent. Invalid identities and corrupt payloads were rejected. |
| Physical prefix reuse | Partial | Three 128K sibling lookups reused a device-resident parent with zero imported checkpoint bytes. Each live child still exports a full checkpoint. Offline parent-relative deltas reconstructed all three exactly. Live incremental child export and multi-branch memory scaling remain open. |
| AdamW output-head LoRA | Pass for rank 4 | Training CLI resumes matched uninterrupted losses, serving bytes and step-four optimizer state on the two-example fixture. |
| Factorwise Muon output-head LoRA | Pass for rank 4 | The same fresh-process and chained-resume checks passed. The second eight-step station round gained six answers and lost six. |
| Adapter serving and cache identity | Pass for tested head adapters | Both exported adapters launched over HTTP and restored after restart. Backend checks rejected cross-adapter reuse and recovered the exact base frontier when disabled. |
| Transformer-layer LoRA and QLoRA | Open | Selected internal transformer modules still need training, exact optimizer resume, export, serving and held-out evaluation through a quantized backbone. |
| Fixed-rank rounds and retention | Partial | A fresh rank-4 residual combined with its rank-4 parent. Compressing the rank-8 sum to rank 4 lost one correct answer. Versioned multi-round resume and reliable retention across skills remain open. |
| Skill improvement | Promotion gate failed for the tested candidate | Failure-only practice reached 53/54 on the repeated register. On a fresh register the step-11 candidate scored 21/24 and the base scored 22/24. The base remains the default. |
| Stock Strata comparison | Measured for single cold requests | Matched 64K and 128K transcription runs cover generation speed, complete throughput, TTFT and GPU memory. Stock was faster end to end. Repeated long-context tail latency and shared-parent crossover remain open. |
| DSpark | Experimental | Acceptance, speed, VRAM and restart continuation were measured for greedy unadapted text. Verifier-only timing and dedicated rejected-path rollback coverage remain open. |
| DFlash2 and MTP | Open | Pin distinct compatible draft artifacts and verifier implementations, then qualify correctness, speed, memory and restore for each. |
| Parallel serving and GPU topology | Partial | Independent native slots were benchmarked, and queued PCC requests passed cancellation isolation. Concurrent decoding on a shared PCC parent, restart under sustained load and additional GPU topologies remain open. |
| BranchWrite integration | Partial | Separate branches asked questions of a shared 128K register and receipt recovery was exercised. Two guided attempts independently fixed a real launcher bug in scoped workspaces. A separate client resumed after the edit at a step-budget boundary and completed verification. Whole-checkout registration and recovery during an interrupted tool write remain open. |

[Checkpoint regression](../../../tools/test_inkling_pcc.py) · [Vision regression](../../../tools/test_inkling_pcc_vision.py) · [Cancellation isolation](../../../tools/test_inkling_cancel_isolation.py) · [Trainer restart](../../../tools/test_inkling_training_restart.py) · [Adapter isolation](../../../tools/test_inkling_head_lora.py)
