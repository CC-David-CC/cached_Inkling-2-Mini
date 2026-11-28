"""Plot single-request Inkling latency from the published per-trial rates."""
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
with (directory / "gpu-comparison.csv").open() as source:
    rows = list(csv.DictReader(source))

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
metrics = [
    ("First token (s)", lambda r: float(r["ttft_s"])),
    ("Decode (ms/token)", lambda r: 1000 / float(r["decode_tps"])),
    ("Full request (s)", lambda r: int(r["output_tokens"]) / float(r["full_tps"])),
]
groups = [(host, length) for host in ("llm-49", "llm-60") for length in (3045, 6045)]
for ax, (title, metric) in zip(axes, metrics):
    for index, (host, length) in enumerate(groups):
        for mode, offset, color in [("off", -0.17, "#57718d"), ("dspark", 0.17, "#007a68")]:
            values = [metric(r) for r in rows if r["host"] == host
                      and int(r["prefill_tokens"]) == length and r["draft"] == mode]
            if len(values) != 2:
                raise ValueError(f"Expected two trials for {host}, {length}, {mode}")
            y = index + offset
            ax.barh(y, median(values), height=0.29, color=color,
                    label=mode if index == 0 else None)
            ax.scatter(values, [y - 0.055, y + 0.055], s=13, facecolor="white",
                       edgecolor=color, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_yticks(range(4), [f"{h} · {n:,}" for h, n in groups] if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.35), ncol=2, frameon=False)
fig.suptitle("Cached Inkling 2 Mini  |  Single-request latency", x=0.13, ha="left",
             fontsize=18, fontweight="bold")
fig.text(0.13, 0.86, "SQ-20 transcription · 3,045 or 6,045 input tokens · 1,024 output tokens")
fig.text(0.45, 0.05, "Bars show medians. Dots show both trials. Lower is faster.", fontsize=10)
fig.subplots_adjust(left=0.13, right=0.98, top=0.73, bottom=0.22, wspace=0.17)
fig.savefig(directory / "gpu-latency.png", dpi=160)
