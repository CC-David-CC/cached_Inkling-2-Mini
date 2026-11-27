"""Plot the two measured trials for each P4 KV placement."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "p4-kv-placement-decode.csv").open() as stream:
    rows = list(csv.DictReader(stream))
labels = ["64K · GPU", "128K · GPU + RAM", "128K · RAM"]
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, field, title in zip(axes, ["decode_tps", "tpot_ms", "final_free_mib"],
                           ["Decode (tokens/s)", "Time per token (ms)", "Free VRAM (MiB)"]):
    for i, mode in enumerate(["off", "auto", "on"]):
        values = [float(row[field]) for row in rows if row["kv_offload"] == mode]
        assert len(values) == 2
        avg = sum(values) / len(values)
        ax.barh(i, avg, height=.46, color=["#406079", "#17756b", "#ae7039"][i])
        ax.scatter(values, [i-.09, i+.09], color="#172932", s=13)
        ax.text(max(values)*1.015, i, f"{avg:.2f}" if field=="decode_tps" else f"{avg:.0f}", va="center")
    ax.set_yticks(range(3), labels if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.margins(x=.23)
    ax.spines[["top","right","left"]].set_visible(False)
fig.suptitle("SQ-10 on one Tesla P4", weight="bold")
fig.text(.02,.02,"32 native decode steps per trial after synthetic state import. Bars show means; dots show both trials.")
fig.subplots_adjust(left=.16, bottom=.17, top=.78, wspace=.23)
fig.savefig(root/"p4-kv-placement-decode.png", dpi=180, bbox_inches="tight")
