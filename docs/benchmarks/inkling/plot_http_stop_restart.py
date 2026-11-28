"""Plot measured worker phases for one warm and one fresh-process 250k follow-up."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
rows = json.loads((base / "http-stop-250k-timed.json").read_text())
fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
labels = ["Open checkpoint", "Process follow-up", "Generate", "Save state", "Other worker time"]
colors = ["#28678a", "#59a391", "#d49a44", "#976789", "#c6cbd0"]
for y, row in enumerate(rows):
    answer = row["answer"]
    durations = [answer[key] for key in ("source_restore_seconds", "input_seconds",
                                          "generation_seconds", "state_save_seconds")]
    total = answer["request_seconds"]
    durations.append(total - sum(durations))
    assert min(durations) >= 0
    left = 0
    for label, color, seconds in zip(labels, colors, durations):
        ax.barh(y, seconds, left=left, color=color, height=.46, label=label if y == 0 else None)
        if seconds >= 2:
            ax.text(left + seconds / 2, y, f"{seconds:.1f} s", ha="center", va="center")
        left += seconds
    ax.annotate(f"{total:.2f} s", (total, y), xytext=(4, 0), textcoords="offset points", va="center")
ax.set_yticks([0, 1], ["Resident parent", "Fresh process"])
ax.invert_yaxis()
ax.set_xlim(0, max(row["answer"]["request_seconds"] for row in rows) * 1.16)
ax.set_xlabel("Worker request time (seconds)")
ax.legend(loc="upper left", bbox_to_anchor=(0, -.32), ncol=3, frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("250k-token document · one follow-up per condition", loc="left")
fig.savefig(base / "http-stop-250k-timed.png", dpi=170)
