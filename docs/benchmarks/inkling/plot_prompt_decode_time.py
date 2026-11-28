"""Compare measured native prefill and decode durations for long fresh prompts."""
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sources = [
    (65536, "llm49-64k-frontier-runs.jsonl", "dspark-64k-frontier-runs.jsonl"),
    (131072, "llm49-128k-generation.jsonl", "dspark-128k-generation.jsonl"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, layout="constrained")
for ax, (length, *files) in zip(axes, sources):
    labels = []
    output_hashes = set()
    for host, filename in zip(("llm-49", "llm-60"), files):
        runs = [json.loads(line) for line in (root / filename).read_text().splitlines()]
        for mode in ("off", "dspark"):
            rows = [r for r in runs if r["mode"] == mode]
            for r in rows:
                if (r["prompt"], r["tokens"]) != (length, 512):
                    raise ValueError(f"Workload mismatch in {filename}")
                output_hashes.add(r["sha256"])
            read = median(r["prefill_s"] for r in rows)
            generate = median(r["generation_s"] for r in rows)
            index = len(labels)
            labels.append(f"{host} · {mode.upper() if mode == 'off' else 'DSpark'} (n={len(rows)})")
            ax.barh(index, read, color="#517797", label="Prompt prefill" if index == 0 else None)
            ax.barh(index, generate, left=read, color="#d08545", label="Generate 512 tokens" if index == 0 else None)
            ax.text(read + generate + 1, index, f"{read + generate:.1f}s", va="center")
    if len(output_hashes) != 1:
        raise ValueError(f"Output token IDs differ at {length} tokens")
    ax.set(yticks=range(len(labels)), yticklabels=labels, title=f"{length:,} input tokens",
           xlabel="Sum of median phase times (seconds)", xlim=(0, 110))
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
fig.suptitle("Cached SQ-10 · one fresh prompt and one answer · native execution")
fig.savefig(root / "prompt-decode-time.png", dpi=160)
