"""Plot measured service count errors on the llm-60 exit-log workload."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

folder = Path(__file__).resolve().parent
with (folder / "linux-rollup-per-service.csv").open() as stream:
    reader = csv.DictReader(stream)
    columns = reader.fieldnames[2:]
    rows = list(reader)

errors = np.array([[int(row[c]) - int(row["expected"]) for row in rows] for c in columns])
assert np.abs(errors).max() == 1
fig, ax = plt.subplots(figsize=(12, 4.7), layout="constrained")
ax.imshow(errors, cmap=ListedColormap(["#a3c8ee", "#edf3f0", "#ffc483"]), vmin=-1, vmax=1)
labels = ["Cached SQ-10", "Cached SQ-15", "Cached SQ-20", "Stock SQ-10",
          "Stock SQ-15", "Stock SQ-20", "Stock UD-Q4_K_M"]
ax.set_yticks(range(len(labels)), [f"{label}   {sum(diff == 0)}/18" for label, diff in zip(labels, errors)])
ax.set_xticks(range(len(rows)), [r["service"].split("-")[1] for r in rows])
ax.set_xlabel("Worker ID")
for (y, x), error in np.ndenumerate(errors):
    if error:
        ax.text(x, y, f"{error:+d}", ha="center", va="center", weight="bold")
ax.axhline(2.5, color="white", linewidth=3)
ax.set_title("Linux service exit counts on llm-60", loc="left", weight="bold", fontsize=17, pad=34)
ax.text(0, 1.055, "97 log records · 3 identical answers per configuration · signed cells show counting errors",
        transform=ax.transAxes, fontsize=10)
fig.savefig(folder / "linux-rollup-per-service.png", dpi=160)
