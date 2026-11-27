"""Render warm-request timings for matched SQ-10 P4 checkpoint runs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
rows = []
for filename, method in [
    ("p4-65k-state-save-http.csv", "Baseline"),
    ("p4-65k-owner-reuse-http.csv", "Digest reuse"),
    ("p4-65k-portable-save-http.csv", "Portable capture"),
    ("p4-65k-checksum-save-http.csv", "Single payload hash"),
]:
    with (folder / filename).open() as stream:
        rows.extend(dict(row, method=method) for row in csv.DictReader(stream))

fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
stages = [
    ("generation_s", "Generation", "#1b6486"),
    ("state_save_s", "State save", "#297660"),
    ("other_model_s", "Other model work", "#685693"),
    ("capture_s", "Checkpoint capture", "#965724"),
]
for i, row in enumerate(rows):
    elapsed = 0
    wall = float(row["wall_s"])
    for field, label, color in [*stages, (None, "Remaining wall time", "#52616e")]:
        duration = float(row[field]) if field else wall - elapsed
        assert duration >= 0
        ax.barh(i, duration, left=elapsed, height=.48, color=color, label=label if i == 0 else None)
        if duration >= 3:
            ax.text(elapsed + duration / 2, i, f"{duration:.1f}", ha="center", va="center", color="white")
        elapsed += duration
    ax.text(wall + .6, i, f"{wall:.1f} s", va="center")
ax.set_yticks(range(len(rows)), [f"{r['method']} · branch {r['trial']}" for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, max(float(r["wall_s"]) for r in rows) + 8)
ax.set_xlabel("Seconds per warm HTTP request")
ax.set_title("One Tesla P4 · 65,035-token resident parent · 159 visible tokens per answer", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=5, frameon=False, fontsize=9)
fig.savefig(folder / "p4-65k-state-save.png", dpi=160)
