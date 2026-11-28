"""Compare payload bytes for the three verified SQ-20 child reconstructions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
family = json.loads((folder / 'sq20-hybrid-128k-checkpoint-family.json').read_text())
parent = next(row for row in family if row['parent'] is None)
parent_bytes = parent['state_bytes'] + 4 * parent['logits_count']
children = [row for row in family if row['parent'] == parent['snapshot']]
full = sum(row['state_bytes'] + 4 * row['logits_count'] for row in children)
fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')
series = [('Full checkpoints', full)]
for label, filename in [('Same-offset 16 KiB blocks', 'sq20-128k-same-offset-blocks.json'),
                        ('Layer-aligned KV rows', 'sq20-128k-aligned-kv.json')]:
    rows = json.loads((folder / filename).read_text())
    assert {r.get('child', r.get('snapshot')) for r in rows} == {c['snapshot'] for c in children}
    series.append((label, sum(r['literal_bytes'] for r in rows)))
for y, (label, child_bytes) in enumerate(series):
    base, extra = parent_bytes / 1e9, child_bytes / 1e9
    ax.barh(y, base, color='#516d86', label='Shared document' if y == 0 else None)
    ax.barh(y, extra, left=base, color='#239483', label='Three children' if y == 0 else None)
    ax.text(base + extra + .2, y, f'{base + extra:.2f} GB', va='center')
ax.set_yticks(range(len(series)), [s[0] for s in series])
ax.invert_yaxis()
ax.set_xlim(0, 20)
ax.set_xlabel('Payload size in decimal GB, excluding metadata')
ax.set_title('SQ-20 on llm-60 · one 131,119-token document and three 18-token branches', loc='left')
ax.legend(loc='lower right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig(folder / 'sq20-shared-storage.png', dpi=160)
