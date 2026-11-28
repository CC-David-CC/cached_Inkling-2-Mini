from pathlib import Path
import csv

import matplotlib.pyplot as plt

stem = Path(__file__).with_name("branchwrite-sq20-130k-first-visible")
with stem.with_suffix(".csv").open() as source:
    rows = list(csv.DictReader(source))

fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
left = [0.0] * len(rows)
for key, label, color in [
    ("first_visible_s", "Waiting for text", "#96b7d1"),
    ("last_visible_s", "Text streaming", "#167b74"),
    ("completion_s", "Until completion", "#efbb70"),
]:
    end = [float(row[key]) for row in rows]
    ax.barh(range(len(rows)), [b - a for a, b in zip(left, end)],
            left=left, label=label, color=color, height=0.62)
    left = end

labels = [f"{'Serial' if row['clients'] == '1' else 'Overlapping'} · station {row['station']}"
          for row in rows]
ax.set_yticks(range(len(rows)), labels)
ax.invert_yaxis()
ax.set_xlabel("Seconds since each client issued its ask")
ax.set_title("BranchWrite on llm-60 · warm 130k context · 353 tokens per answer")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False)
ax.spines[["right", "top", "left"]].set_visible(False)
fig.savefig(stem.with_suffix(".png"), dpi=160)
