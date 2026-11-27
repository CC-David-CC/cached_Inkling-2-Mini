"""Compare client-observed durable response times for the same two lookups."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (folder / 'branchwrite-128k-pair.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
for mode, offset, color in [('sequential', -.19, '#227184'), ('paired', .19, '#a45c32')]:
    values = sorted((r for r in rows if r['mode'] == mode), key=lambda r: r['idx'])
    positions = [i + offset for i in range(len(values))]
    bars = ax.barh(positions, [r['seconds'] for r in values], height=.34, color=color, label=mode.capitalize())
    ax.bar_label(bars, fmt='%.2f s', padding=4)
ax.set_yticks([0, 1], ['Station 02478', 'Station 08050'])
ax.invert_yaxis()
ax.set_xlim(0, max(r['seconds'] for r in rows) * 1.2)
ax.set_xlabel('Seconds from ask to durable answer')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(loc='lower right')
fig.suptitle('Two lookups on one 131,119-token parent\nBranchWrite · SQ-10 on llm-60 · one pair per mode')
fig.savefig(folder / 'branchwrite-128k-pair.png', dpi=170)
