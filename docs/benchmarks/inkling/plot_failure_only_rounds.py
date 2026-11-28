"""Plot paired changes and total correct under failure-only AdamW practice."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "failure-only-rounds.csv").open() as source:
    rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(source)]

steps = [row["optimizer_step"] for row in rows]
scores = [row["correct_of_54"] for row in rows]
previous = 51
for row in rows:
    assert previous + row["gained"] - row["lost"] == row["correct_of_54"]
    previous = row["correct_of_54"]

fig, (score, changes) = plt.subplots(2, 1, figsize=(8.8, 5.3), sharex=True,
                                    layout="constrained", height_ratios=[1.2, 1])
score.plot(steps, scores, color="#1d6285", marker="o")
score.axhline(51, color="#888888", linestyle="--", linewidth=1, label="Base · 51/54")
for step, value in zip(steps, scores):
    score.annotate(str(value), (step, value), xytext=(0, 6), textcoords="offset points",
                   ha="center", fontsize=9)
score.set(ylim=(0, 60), ylabel="Correct out of 54")
score.legend(loc="lower right", frameon=False)
changes.bar(steps, [r["gained"] for r in rows], color="#14786b", label="Newly correct")
changes.bar(steps, [-r["lost"] for r in rows], color="#ad5735", label="Previously correct, now wrong")
changes.axhline(0, linewidth=.8, color="#505050")
changes.set(xticks=steps, xlabel="Optimizer step", ylabel="Changes since previous score")
changes.legend(loc="lower right", frameon=False)
for ax in (score, changes):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Inkling SQ-20 · rank-4 AdamW on the preceding mistakes", fontsize=14)
fig.savefig(root / "failure-only-rounds.png", dpi=170)
