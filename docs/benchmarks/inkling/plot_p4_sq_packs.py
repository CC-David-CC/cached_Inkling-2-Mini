"""Compare the three SQ packs on one P4 at the 128K boundary."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "p4-128k-sq-packs.csv").open() as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
for axis, field, title, unit in zip(axes,
        ("decode_tps", "host_expert_gb", "mapped_host_mib"),
        ("Decode speed", "Experts in host RAM", "Global KV in host RAM"),
        ("tokens/s", "GB", "MiB")):
    values = [float(row[field]) for row in rows]
    bars = axis.barh([r["pack"] for r in rows], values, color=["#146b67","#4c7285","#a96336"])
    axis.bar_label(bars, fmt="%.2f" if field!="mapped_host_mib" else "%.0f", padding=3)
    axis.invert_yaxis()
    axis.set(xlabel=unit, title=title, xlim=(0,max(values)*1.28))
    axis.spines[["right","top","left"]].set_visible(False)
fig.suptitle("Tesla P4 · 131,072 context · 32 native decode steps per pack", weight="bold")
fig.savefig(root/"p4-128k-sq-packs.png", dpi=180, bbox_inches="tight")
