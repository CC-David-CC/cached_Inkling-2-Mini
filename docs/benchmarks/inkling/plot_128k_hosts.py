"""Plot matched native 128k timings, one fresh worker per host and mode."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
for i, (host, filename) in enumerate([
    ("llm-49 · RTX 4090", "llm49-128k-generation.jsonl"),
    ("llm-60 · RTX PRO 6000", "dspark-128k-generation.jsonl"),
]):
    rows = [json.loads(line) for line in (base / filename).read_text().splitlines()]
    assert [r["mode"] for r in rows] == ["off", "dspark"]
    for j, row in enumerate(rows):
        assert (row["prompt"], row["tokens"]) == (131072, 512)
        for ax, key in zip(axes, ["prefill_s", "generated_tps"]):
            value = row[key]
            y = i + (j - 0.5) * 0.32
            ax.barh(y, value, height=0.28, color=["#426689", "#187566"][j],
                    label=["Draft off", "DSpark"][j] if i == 0 else None)
            ax.text(value, y, f"  {value:.1f}", va="center", fontsize=9)
for ax, title in zip(axes, ["Read prompt (seconds)", "Generation (tokens/s)"]):
    ax.set(yticks=[0, 1], yticklabels=["llm-49 · 4090", "llm-60 · PRO 6000"], title=title)
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
fig.suptitle("Cached SQ-10 · 131,072 input / 512 output tokens · one run per bar")
fig.savefig(base / "cached-128k-hosts.png", dpi=165)
