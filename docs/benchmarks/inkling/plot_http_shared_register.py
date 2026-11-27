"""Compare measured BranchWrite lookup time and time saving its generated child."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(8.5, 3.8))
y = 0
ticks = []
for size in ('32k', '131k'):
    with (root / f'llm60-http-owner-{size}.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['run'] == 'http-owner-traced':
                continue
            lookup = float(row['lookup_s'])
            save = float(row['save_s'])
            ax.barh(y, lookup, color='#cadae3', label='Other lookup work' if y == 0 else None)
            ax.barh(y, save, color='#176e6c', label='Child checkpoint save' if y == 0 else None)
            ax.text(lookup + .1, y, f'{lookup:.2f} s', va='center')
            ticks.append(f"{int(row['document_tokens']):,} tokens · station {int(row['station']):05}")
            y += 1
ax.set(yticks=range(y), yticklabels=ticks, xlabel='Elapsed seconds', xlim=(0,8))
ax.invert_yaxis()
ax.spines[['top','right','left']].set_visible(False)
ax.set_title('BranchWrite lookups on a shared register · SQ-10 on llm-60')
ax.legend(loc='upper center', bbox_to_anchor=(.5,-.27), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(root/'llm60-http-shared-register.png',dpi=180,bbox_inches='tight')
