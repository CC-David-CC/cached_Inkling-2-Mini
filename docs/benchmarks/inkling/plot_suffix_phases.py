"""Compare checkpoint verification paths on the 131,072-token GPU fixture."""
import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
with (folder / 'suffix-128k-phases.csv').open() as source:
    rows = list(csv.DictReader(source))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), layout='constrained')
for ax, key, title in zip(axes, ('parent_select_seconds', 'restore_seconds', 'suffix_export_seconds'),
                          ('Select parent', 'Import child suffix', 'Export child suffix')):
    for index, (mode, color) in enumerate([('Rehash', '#567496'), ('Verified owner', '#127562')]):
        samples = [float(r['ms']) for r in rows if r['phase'] == key and r['parent_verification'] == mode]
        value = median(samples)
        ax.bar(index, value, color=color, width=.55)
        ax.text(index, value, f'{value:,.1f}', ha='center', va='bottom')
    ax.set_xticks([0, 1], ['Rehash', 'Verified owner'])
    ax.set_title(title)
    ax.set_ylabel('Median milliseconds')
    ax.margins(y=.2)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('128K shared parent · two-token children · SQ-10 on llm-60\n'
             'One run per mode · 3 selections and imports, 2 exports per run', fontsize=13)
fig.savefig(folder / 'suffix-128k-phases.png', dpi=160)
