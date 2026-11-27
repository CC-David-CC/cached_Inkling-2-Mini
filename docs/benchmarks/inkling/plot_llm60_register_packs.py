"""Plot per-pack prefill and fixed-step decode on the same serialized register."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "llm60-128k-sq-packs.csv").open() as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
for axis, field, label in zip(axes, ["prefill_tps", "decode_tps"],
                            ["Prefill (tokens/s)", "Decode, 32 steps (tokens/s)"]):
    values = [float(r[field]) for r in rows]
    bars = axis.barh([r["pack"] for r in rows], values, color=["#176d67", "#557589", "#aa663e"])
    axis.bar_label(bars, fmt="%.1f", padding=4)
    axis.set(xlabel=label, xlim=(0,max(values)*1.3))
    axis.invert_yaxis()
    axis.spines[["right","top","left"]].set_visible(False)
fig.suptitle("llm-60 · 131,040 input tokens · 6 GiB GPU expert budget per pack")
fig.savefig(root/"llm60-128k-sq-packs.png", dpi=180, bbox_inches="tight")
