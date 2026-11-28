"""Plot measured restore and total lookup times for one resident 131k parent."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

directory = Path(__file__).resolve().parent
with (directory / 'llm60-131k-restore-hash.csv').open() as source:
    rows = list(csv.DictReader(source))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
colors = ['#26738a', '#e49a54', '#cfdbe0']
for ax, run, title in zip(axes, ['profile', 'scoped'], ['Rehash at each lookup', 'Reuse digest while owner is live']):
    group = [r for r in rows if r['run'] == run]
    assert len(group) == 3 and all(int(r['import_bytes']) == 0 for r in group)
    for y, row in enumerate(group):
        total, restore, digest = (float(row[k]) for k in ('lookup_ms', 'restore_ms', 'sha256_ms'))
        assert 0 <= digest <= restore <= total
        offset = 0
        for duration, color, label in zip((digest, restore-digest, total-restore), colors,
                                          ('SHA-256', 'Other restore work', 'Rest of lookup')):
            ax.barh(y, duration, left=offset, height=.52, color=color, label=label if y == 0 else None)
            offset += duration
        ax.text(total+35, y, f'{total:,.0f}', va='center', fontsize=9)
    ax.set(title=title, xlabel='Elapsed time (ms)', xlim=(0, 2240))
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.spines[['top','right','left']].set_visible(False)
axes[0].set_yticks(range(3), ['Lookup 1','Lookup 2','Lookup 3'])
axes[0].invert_yaxis()
fig.suptitle('Repeated questions on a shared 131,024-token parent', weight='bold')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncols=3, frameon=False,
           bbox_to_anchor=(.5,.095))
fig.text(.5,.045,'llm-60 · resident SQ-10 · one observation per bar · zero checkpoint bytes imported',
         ha='center', fontsize=9)
fig.text(.5,.002,'In the right panel, the checkpoint owner was released before lookup 3.',ha='center', fontsize=9)
fig.tight_layout(rect=(0,.20,1,.94))
fig.savefig(directory / 'llm60-131k-restore-hash.png', dpi=170, bbox_inches='tight')
