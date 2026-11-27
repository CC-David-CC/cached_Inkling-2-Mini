"""Plot per-document native timings using the exact serialized prompt as the join key."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
def read_results(filename):
    with (root / filename).open() as stream:
        return {row["prompt_sha256"]: row for row in csv.DictReader(stream)}

p4 = read_results("p4-111k-registers.csv")
control = read_results("llm60-111k-registers.csv")
assert len(p4) == 3 and p4.keys() == control.keys()
keys = sorted(p4, key=lambda k: float(p4[k]["depth_percent"]))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for axis, field, title in zip(axes, ("prefill_seconds", "decode_tps"), ("Time to read the document (minutes)", "Native decode (tokens/s)")):
    for offset, rows, label, color in [(-.19, p4, "P4 on r730", "#146b67"), (.19, control, "RTX PRO 6000 on llm-60", "#ad642e")]:
        values = [float(rows[k][field] or "nan") for k in keys]
        if field == "prefill_seconds":
            values = [value/60 for value in values]
        bars = axis.bar([i+offset for i in range(3)], values, .36, label=label, color=color)
        axis.bar_label(bars, labels=["OOM" if v != v else f"{v:.1f}" for v in values], fontsize=9)
        for idx, v in enumerate(values):
            if v != v:
                axis.text(idx+offset, 0, "OOM", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(3), [f"{p4[k]['depth_percent']}%" for k in keys])
    axis.set_xlabel("Location of answer in document")
    axis.set_title(title)
    axis.spines[["right","top"]].set_visible(False)
    axis.margins(y=.18)
fig.suptitle("SQ-10 retrieval from 111,328 input tokens", weight="bold")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
fig.subplots_adjust(bottom=.24, top=.78, wspace=.23)
fig.savefig(root/"register-111k.png", dpi=180, bbox_inches="tight")
