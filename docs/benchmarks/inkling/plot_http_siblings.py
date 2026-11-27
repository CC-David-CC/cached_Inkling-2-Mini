"""Plot HTTP latency and unique checkpoint storage from the llm-60 receipts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
data = json.loads((folder / 'http-128k-siblings.json').read_text())
requests = data['requests']
sizes = data['payload_bytes']
child = requests[0]['payload']
parent, = set(sizes) - {child}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
for ax, title, labels, values, unit, color in (
    (axes[0], 'Time to durable HTTP response', ['First branch', 'Sibling'],
     [r['seconds'] for r in requests], 's', '#226c87'),
    (axes[1], 'Additional checkpoint storage', ['Parent', 'First branch', 'Identical sibling'],
     [sizes[parent] / 2**30, sizes[child] / 2**30, 0], 'GiB', '#29775e'),
):
    bars = ax.barh(labels, values, height=.5, color=color)
    ax.bar_label(bars, labels=[f'{v:.2f} {unit}' for v in values], padding=5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.45)
    ax.set_title(title, loc='left', fontsize=12)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(length=0)
fig.suptitle('131,072 shared tokens · SQ-10 on llm-60\nTwo sequential requests, two output tokens each, zero checkpoint bytes imported',
             fontsize=12)
fig.savefig(folder / 'http-128k-siblings.png', dpi=170)
