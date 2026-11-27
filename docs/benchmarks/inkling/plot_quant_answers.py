"""Plot short-answer grading from the matched stock Strata runs on llm-60."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = Path(__file__).resolve().parent
with (root / "quant-pilot.csv").open() as stream:
    rows = {(row["pack"], row["domain"]): row for row in csv.DictReader(stream)}

packs = ["SQ-10", "SQ-15", "SQ-20", "Q4_K_M"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
for ax, domain, title in zip(axes, ["coding", "linux"], ["Coding", "Linux administration"]):
    for i, pack in enumerate(packs):
        row = rows[pack, domain]
        exact = int(row["exact_format_correct"])
        correct = int(row["underlying_answer_correct"])
        total = int(row["questions"])
        assert 0 <= exact <= correct <= total == 4
        ax.barh(i, total, color="#e9edf0", height=.56)
        ax.barh(i, correct, color="#87c8bd", height=.56)
        ax.barh(i, exact, color="#17665f", height=.56)
    ax.set(xticks=range(5), yticks=range(4), yticklabels=packs, xlim=(0, 4), title=title)
    ax.spines[:].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle("Inkling 2 Mini  •  Four short questions per domain", weight="bold")
fig.legend(handles=[Patch(color=c, label=s) for c,s in [
    ("#17665f", "Correct and exact format"), ("#87c8bd", "Correct with format difference"),
    ("#e9edf0", "Incorrect")]], loc="lower center", ncol=3, frameon=False)
fig.subplots_adjust(bottom=.2, top=.78, wspace=.12)
fig.savefig(root / "quant-short-answers.png", dpi=180, bbox_inches="tight")
