"""Regenerate the llm-60 executable task comparison from the published scores."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
with (directory / "executable-quant-pilot.csv").open() as source:
    rows = list(csv.DictReader(source))

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True,
                         gridspec_kw={"width_ratios": [3, 4, 1.6]})
for ax, runtime in zip(axes, ["Cached", "stock Strata", "vLLM"]):
    scores = {(r["pack"], r["task"]): int(r["fixtures_passed"])
              for r in rows if r["runtime"] == runtime}
    packs = list(dict.fromkeys(pack for pack, task in scores))
    for task, label, offset, color in [
        ("latest", "SQLite latest row per host", -.18, "#236a97"),
        ("permissions", "Linux chmod operand", .18, "#d37938"),
    ]:
        bars = ax.bar([i + offset for i in range(len(packs))],
                      [scores[pack, task] for pack in packs], .36, color=color, label=label)
        ax.bar_label(bars, padding=3)
    ax.set_title(runtime)
    ax.set_xticks(range(len(packs)), packs, fontsize=9)
    ax.set_ylim(0, 3.6)
    ax.set_yticks(range(4))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Fixtures passed out of three")
fig.suptitle("Inkling 2 Mini on two executable tasks", fontsize=18)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(.5, .90), ncol=2, frameon=False)
fig.text(.075, .04, "llm-60  •  Each answer was tested on three fixture configurations.")
fig.subplots_adjust(top=.72, bottom=.18, left=.075, right=.98, wspace=.20)
fig.savefig(directory / "executable-quant-pilot.png", dpi=150)
