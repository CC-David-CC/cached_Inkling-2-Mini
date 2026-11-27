"""Plot the paired output-head placement experiments on llm-60."""
import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
with (directory / "dflash2-head-placement.csv").open() as source:
    rows = list(csv.DictReader(source))

prompts = ["python", "linux", "json"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for resident, shift, color, label in [
    ("0", -.18, "#9d6246", "Copy per block"),
    ("1", .18, "#26778a", "Keep on GPU"),
]:
    rates, latencies = [], []
    for prompt in prompts:
        blocks = [r for r in rows if r["prompt"] == prompt and r["resident"] == resident]
        rates.append(sum(int(r["committed"]) for r in blocks) /
                     sum(float(r["seconds"]) for r in blocks))
        latencies.append(median(float(r["proposal_seconds"]) * 1000 for r in blocks))
    for ax, values in zip(axes, [rates, latencies]):
        bars = ax.bar([i + shift for i in range(3)], values, .36, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
for ax, title in zip(axes, ["Decode throughput (tokens/s)", "Median proposal time (ms/block)"]):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xticks(range(3), ["Python", "Linux", "JSON"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=.17)
fig.suptitle("DFlash2 output-head placement on hybrid SQ-10", fontsize=16)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.91),
           ncol=2, frameon=False)
fig.text(.07,.035,"llm-60   •   Serial target verification   •   2 runs × 8 blocks per prompt and setting   •   Startup and prefill excluded",fontsize=9)
fig.subplots_adjust(top=.73,bottom=.17,left=.07,right=.97,wspace=.24)
fig.savefig(directory / "dflash2-head-placement.png", dpi=160)
