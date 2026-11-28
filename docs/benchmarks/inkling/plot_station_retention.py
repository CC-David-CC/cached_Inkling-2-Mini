"""Group the 54 station cases by their complete recorded grading history."""
from collections import Counter
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

root = Path(__file__).resolve().parent
with (root / 'station-adapter-paired-answers.csv').open() as source:
    base = {r['station']: r['stations-v1-base'] == r['expected'] for r in csv.DictReader(source)}
with (root / 'failure-only-paired-answers.csv').open() as source:
    rows = list(csv.DictReader(source))
if len(rows) != 54 or {r['station'] for r in rows} != base.keys():
    raise ValueError('Expected the same 54 station cases in both reports')
steps = list(range(8, 17))
patterns = Counter((base[r['station']], *(r[f'step_{s}'] == r['expected'] for s in steps))
                   for r in rows)
ordered = sorted(patterns, key=lambda p: (p[0], patterns[p], p), reverse=True)
fig, ax = plt.subplots(figsize=(8.2, max(4, len(ordered) * .35)), layout='constrained')
ax.imshow(ordered, aspect='auto', interpolation='nearest',
          cmap=ListedColormap(['#f0c59d', '#146958']), vmin=0, vmax=1)
for y, pattern in enumerate(ordered):
    for x, passed in enumerate(pattern):
        ax.text(x, y, '+' if passed else '−', ha='center', va='center',
                color='white' if passed else '#633b22')
ax.set_xticks(range(len(steps) + 1), ['Base', *steps])
ax.set_yticks(range(len(ordered)), [str(patterns[p]) for p in ordered])
ax.set_ylabel('Stations with this history')
ax.set_xlabel('Optimizer step     ·     + correct     − incorrect')
ax.set_title('SQ-20 failure-only training · 54 station answers')
fig.savefig(root / 'station-retention-history.png', dpi=170)
print('Always correct:', sum(n for p, n in patterns.items() if all(p)))
print('Base successes lost then recovered by step 16:',
      sum(n for p, n in patterns.items() if p[0] and p[-1] and not all(p)))
