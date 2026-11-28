"""Plot server-side timing for the exact 1,024-token copy comparison."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

stem = Path(__file__).with_name("sq20-dspark-copy-timing")
with stem.with_suffix(".csv").open() as source:
    rows = list(csv.DictReader(source))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
fig.suptitle("SQ-20 on llm-60 · 1,024 copied tokens · one request per mode")
colors = ["#54748d", "#008174"]
for ax, title, values, unit in zip(
    axes, ["Generation rate", "Generate request including checkpoint save"],
    [[int(r["output_tokens"]) / float(r["generation_seconds"]) for r in rows],
     [float(r["request_seconds"]) for r in rows]], ["tokens/s", "seconds"]
):
    bars = ax.barh(["Undrafted", "DSpark"], values, color=colors, height=0.55)
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_xlim(0, max(values) * 1.19)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(unit)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.savefig(stem.with_suffix(".png"), dpi=160)
