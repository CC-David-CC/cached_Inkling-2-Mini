"""Plot measured request-to-receipt times for serial follow-up on llm-60."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

folder = Path(__file__).resolve().parent
with (folder / 'branchwrite-250k-serial.csv').open() as stream:
    lookups = list(csv.DictReader(stream))
with (folder / 'branchwrite-250k-sql.csv').open() as stream:
    sql = list(csv.DictReader(stream))

fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
labels = []
groups = [[row for row in lookups if row['position'] == pos] for pos in ('early', 'middle', 'late')]
for index, (label, rows) in enumerate(zip(['Early lookup', 'Middle lookup', 'Late lookup', 'SQLite query'], groups + [sql])):
    labels.append(f"{label}\n{rows[0].get('output_tokens') or rows[0]['generated']} output tokens")
    for offset, row in zip((-.18, .18), rows, strict=True):
        color = '#ad5830' if int(row['import_bytes']) else ('#256677' if offset < 0 else '#63a9a6')
        seconds = float(row['seconds'])
        ax.barh(index + offset, seconds, height=.29, color=color)
        ax.text(seconds + .3, index + offset, f'{seconds:.2f}', va='center', fontsize=9)
ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel='Seconds from request to durable answer', xlim=(0, 32))
ax.invert_yaxis()
ax.grid(axis='x', alpha=.16)
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(handles=[Patch(color=c, label=t) for c, t in [
    ('#256677', 'Original worker · resident'), ('#ad5830', 'Restart · parent imported'),
    ('#63a9a6', 'Restart · resident')]], loc='lower left', bbox_to_anchor=(0, 1), ncol=3, fontsize=9, frameon=False)
fig.suptitle('BranchWrite response times with a 250k-token parent\nInkling SQ-10 · llm-60 · one request per bar', fontsize=14)
fig.savefig(folder / 'branchwrite-250k-serving.png', dpi=180)
