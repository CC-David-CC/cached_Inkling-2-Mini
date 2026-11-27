"""Plot measured input-processing speed as one Inkling context fills."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

folder = Path(__file__).resolve().parent
with (folder / 'llm60-250k-prefill.csv').open() as stream:
    rows = list(csv.DictReader(stream))

tokens = [0] + [int(row['tokens_processed']) for row in rows]
elapsed = [0.0] + [float(row['elapsed_s']) for row in rows]
rates = [(end - start) / (after - before)
         for start, end, before, after in zip(tokens, tokens[1:], elapsed, elapsed[1:])]
fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
ax.stairs(rates, tokens, baseline=None, color='#156c77', linewidth=2)
ax.set(xlabel='Input tokens processed', ylabel='Prefill tokens per second',
       xlim=(0, tokens[-1]), ylim=(0, max(rates) * 1.15))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(axis='y', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Reading a 250,000-token station register\nSQ-10 · llm-60 · 4 GiB GPU expert pool · one run')
fig.savefig(folder / 'llm60-250k-prefill.png', dpi=170)
