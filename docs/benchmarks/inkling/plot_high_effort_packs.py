"""Plot the measured SQ pack results from the high-effort chmod case."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / 'linux-special-mode-high-effort-packs.csv').open() as stream:
    rows = list(csv.DictReader(stream))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
fig.suptitle('One Linux permissions question on llm-60', fontsize=16, weight='bold')
for ax, title in zip(axes, ['Completion tokens per second', 'Generation time (seconds)']):
    ax.set_title(title, fontsize=11, loc='left')
    for i, row in enumerate(rows):
        tokens = int(row['completion_tokens'])
        seconds = float(row['generation_s'])
        speed = tokens / seconds
        value = speed if ax is axes[0] else seconds
        correct = row['correct'] == 'True'
        color = '#1f735b' if correct else '#596e84'
        ax.barh(i, value, color=color, height=.52)
        ax.text(value - 1, i, f'{value:.1f}', ha='right', va='center', color='white')
    ax.set_yticks(range(len(rows)), [row['pack'] for row in rows])
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
fig.supxlabel('Effort 0.99 · 2,048-token limit · one run per pack\nGreen returned 2755. Gray reached the limit with no visible answer.', fontsize=10)
fig.savefig(folder / 'linux-special-mode-high-effort-packs.png', dpi=170)
