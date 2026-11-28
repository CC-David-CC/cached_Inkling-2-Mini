"""Plot generated token costs from the three outcome-only service repairs."""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
runs = {
    'outcome-incident': 'Original tool description',
    'journal-card': 'Journal path visible',
    'command-card': 'Journal path + exact-check rule',
}
costs = {run: Counter() for run in runs}
with (folder / 'journal-discovery-tokens.csv').open() as source:
    for row in csv.DictReader(source):
        costs[row['run']][row['tool']] += int(row['output_tokens'])

fig, ax = plt.subplots(figsize=(9, 3.4), layout='constrained')
left = [0] * len(runs)
for tool, label, color in [
    ('read_file', 'Read files', '#365B79'), ('run_shell', 'Shell commands', '#137B78'),
    ('replace', 'Edit config', '#925C24'), ('finish', 'Finish requests', '#715296')
]:
    values = [costs[run][tool] for run in runs]
    bars = ax.barh(range(len(runs)), values, left=left, color=color, label=label)
    ax.bar_label(bars, label_type='center', color='white')
    left = [a + b for a, b in zip(left, values)]
for index, total in enumerate(left):
    ax.text(total + 5, index, str(total), va='center', weight='bold')
ax.set_yticks(range(len(runs)), runs.values())
ax.invert_yaxis()
ax.set_xlim(0, max(left) + 40)
ax.set_xlabel('Generated output tokens')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False)
ax.set_title('BranchWrite timeout repair on llm-60\nInkling SQ-20 · same user task · one run per description', loc='left')
fig.savefig(folder / 'decision-cost.png', dpi=160)
