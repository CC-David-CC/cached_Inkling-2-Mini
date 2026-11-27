"""Render client-observed timings from two synchronized three-request trials."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / "p4-three-warm-http.csv").open() as stream:
    rows = sorted(csv.DictReader(stream), key=lambda row: (int(row["trial"]), int(row["gpu"])))
assert len(rows) == 6
fig, (timing, rate) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
colors = ["#236583", "#228763"]
for i, row in enumerate(rows):
    trial = int(row["trial"])
    wall = float(row["wall_s"])
    timing.barh(i, wall, color=colors[trial - 1])
    if row["first_stream_text_s"]:
        timing.plot(float(row["first_stream_text_s"]), i, "D", color="#222222",
                    label="First visible text" if i == 0 else None)
    timing.text(wall + .5, i, f"{wall:.1f}")
timing.set_yticks(range(6), [f"Trial {r['trial']} · GPU {r['gpu']}" for r in rows])
timing.invert_yaxis()
timing.set_xlim(0, 1.18 * max(float(r["wall_s"]) for r in rows))
timing.set_xlabel("Seconds per warm HTTP request")
if timing.get_legend_handles_labels()[0]:
    timing.legend(frameon=False)
for i, row in enumerate(rows[::3]):
    aggregate = float(row["group_visible_tps"])
    rate.bar(i, aggregate, color=colors[i], width=.55)
    rate.text(i, aggregate, f"{aggregate:.2f}", ha="center", va="bottom")
rate.set_ylim(0, 1.18 * max(float(r["group_visible_tps"]) for r in rows))
rate.set_xticks([0, 1], ["Trial 1", "Trial 2"])
rate.set_ylabel("Aggregate visible tokens/s across three requests")
for ax in (timing, rate):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Cached SQ-10 · 65,035-token resident prompt on each Tesla P4")
fig.savefig(folder / "p4-three-warm-http.png", dpi=160)
