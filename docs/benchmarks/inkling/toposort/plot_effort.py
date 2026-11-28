import csv
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
with (base / "repair-effort.csv").open() as stream:
    rows = list(csv.DictReader(stream))

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
labels = []
for i, row in enumerate(rows):
    passed = int(row["fixtures_passed"])
    elapsed = float(row["generation_s"])
    color = {8: "#167566", 7: "#536e96", 0: "#98603e"}[passed]
    ax.barh(i, elapsed, color=color, height=0.65)
    outcome = f"{passed}/8 passed" if int(row["fixtures_executed"]) else "no visible code"
    ax.text(elapsed + 0.6, i, f"{elapsed:.1f} s · {outcome}", va="center", fontsize=9)
    labels.append(f"{row['effort']}  ·  {int(row['budget_tokens']):,} tokens")
ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel="Generation time (seconds)",
       ylabel="Effort and output budget", xlim=(0,91),
       title="Repairing one Python function on llm-60 · SQ-10 · drafting off")
ax.invert_yaxis()
ax.spines[["top", "right", "left"]].set_visible(False)
fig.savefig(base / "repair-effort.png", dpi=160)
