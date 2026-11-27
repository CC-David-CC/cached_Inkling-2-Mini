"""Plot pooled token rates from the three matched hybrid SQ-10 benchmarks."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
times = defaultdict(list)
hashes = set()
for filename in ("dspark-hybrid-control.csv", "dflash2-batch-control.csv",
                 "dflash2-3072-greedy-control.csv"):
    with (directory / filename).open() as source:
        for row in csv.DictReader(source):
            if (row["prompt_tokens"], row["output_tokens"]) != ("3072", "259"):
                raise ValueError(f"Workload mismatch in {filename}")
            times[row["mode"]].append(float(row["decode_seconds"]))
            hashes.add(row.get("output_sha256") or row["output_id_sha256"])
if len(hashes) != 1:
    raise ValueError("Generated token sequences differ")

modes = {"dspark": "DSpark", "plain": "Plain SQ-10", "batched_dflash2": "DFlash2 batched",
         "dflash2_serial": "DFlash2 serial"}
rates = [259 * len(times[key]) / sum(times[key]) for key in modes]
labels = [f"{label}  (n={len(times[key])})" for key, label in modes.items()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, values, title in zip(axes, (rates, [1000/r for r in rates]),
                             ("Decode tokens/s", "Mean decode ms/token")):
    bars = ax.barh(range(len(labels)), values, color=["#18755e", "#547390", "#bd6a42", "#bd6a42"])
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(values)*1.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].set_yticks(range(len(labels)), labels)
axes[0].invert_yaxis()
fig.suptitle("Single-request drafting on hybrid SQ-10", fontsize=17)
fig.text(.02, .04, "llm-60   •   3,072 input tokens   •   259 identical output IDs per run   •   Decode only")
fig.subplots_adjust(left=.19, right=.98, bottom=.21, top=.77, wspace=.13)
fig.savefig(directory / "matched-drafting.png", dpi=160)
