"""Render per-request SQ-10 HTTP timings; trial zero includes first-use costs."""
from pathlib import Path
import json
from statistics import median

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
records = [json.loads(line) for line in (folder / 'draft-http-matched.jsonl').read_text().splitlines()]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout='constrained')
modes = [('off', 'Draft off', '#326b8b'), ('dspark', 'DSpark', '#85549b'), ('dflash2', 'DFlash2', '#267556')]
for ax, (metric, title) in zip(axes, [('end_to_end_tps', 'End-to-end tokens/s'),
                                      ('ttft_ms', 'First visible token (ms)'),
                                      ('tpot_ms', 'Generation ms/token')]):
    for i, (mode, name, color) in enumerate(modes):
        rows = [r for r in records if r['mode'] == mode and r['trial'] > 0]
        if not rows:
            ax.text(i, 0, 'n/a', ha='center', va='bottom')
            continue
        values = [1000/r['decode_tps'] if metric == 'tpot_ms' else r[metric] for r in rows]
        value = median(values)
        ax.bar(i, value, width=.58, color=color)
        ax.text(i, value, f'{value:.1f}', ha='center', va='bottom')
    ax.set_xticks(range(len(modes)), [mode[1] for mode in modes])
    ax.set_title(title, fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.margins(y=.18)
fig.suptitle('Cached SQ-10 on llm-60 · 300 inventory records · 256 output tokens\nMedian of two requests after one warm-up · 256 MiB GPU expert budget', fontsize=12)
fig.savefig(folder / 'draft-http-matched.png', dpi=160)
