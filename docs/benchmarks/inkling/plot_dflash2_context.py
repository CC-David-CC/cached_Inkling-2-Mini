"""Chart warm proposal latency for the same growing SQ-10 log prompt."""
import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / "dflash2-context-proposal.csv").open() as stream:
    rows = list(csv.DictReader(stream))

fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
for budget, color, offset in [(1024, "#286480", -.17), (256, "#297455", .17)]:
    selected = [row for row in rows if int(row["expert_budget_mib"]) == budget]
    for index, length in enumerate([3072, 65536, 131072]):
        values = [float(row["proposal_ms"]) for row in selected if int(row["prefix_tokens"]) == length]
        if values:
            value = median(values)
            ax.bar(index + offset, value, width=.32, color=color,
                   label=f"{budget} MiB expert budget" if index == 0 else None)
            ax.text(index + offset, value + 1, f"{value:.1f}", ha="center")
ax.set_xticks(range(3), ["3,072", "65,536", "131,072"])
ax.set_xlabel("Tokens in target context")
ax.set_ylabel("Median ms per seven-token proposal")
ax.set_title("DFlash2 on llm-60 · resident output head · three warm trials")
ax.legend(frameon=False)
ax.margins(y=.16)
ax.spines[["right", "top"]].set_visible(False)
fig.savefig(folder / "dflash2-context-proposal.png", dpi=160)
