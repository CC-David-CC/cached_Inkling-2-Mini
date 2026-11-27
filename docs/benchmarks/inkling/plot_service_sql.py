"""Visualize fixture results for one generated service-status query per profile."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(9, 4))
index = 0
for filename in ("linux-service-sql", "linux-service-sql-references", "linux-service-sql-official-2048"):
    with (root / (filename + ".csv")).open() as source:
        for row in csv.DictReader(source):
            pack = row["pack"]
            official = pack.startswith("official")
            limit = 2048 if filename.endswith("2048") else 512
            label = f"Official mixed · {limit:,} cap" if official else pack
            passed = int(row["fixtures_passed"])
            total = int(row["fixtures_total"])
            ax.barh(index, total, color="#e8edf1", height=.58)
            ax.barh(index, passed, color="#98602f" if official else "#21736a", height=.58)
            stop = "hit cap" if row["native_stop"] == "False" else "stopped"
            ax.text(total + .15, index, f"{passed}/{total}   ·   {row['completion_tokens']} tokens, {stop}", va="center")
            ax.text(-.15, index, label, va="center", ha="right")
            index += 1
ax.invert_yaxis()
ax.set(xlim=(0, 10.5), xticks=range(8), yticks=[], xlabel="SQLite fixtures passed")
ax.spines[:].set_visible(False)
ax.set_title("Counting current failures by host", loc="left", weight="bold")
fig.text(.03, .01, "llm-60   ·   One greedy answer per profile   ·   SQ packs via Cached   ·   Q4 via stock Strata   ·   Official via vLLM")
fig.subplots_adjust(left=.27, right=.99, bottom=.19)
fig.savefig(root / "linux-service-sql.png", dpi=160, bbox_inches="tight")
