"""Plot the six paired DFlash2 hidden-buffer restore measurements."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (directory / 'dflash2-250k-restore.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for checked, offset, color, label in [
        (False, -0.18, '#45677a', 'Copy'), (True, 0.18, '#008473', 'Verify SHA-256 and copy')]:
    group = sorted((r for r in rows if r['verify_sha256'] == checked), key=lambda r: r['trial'])
    assert [r['trial'] for r in group] == [0, 1, 2]
    bars = ax.bar([r['trial'] + offset for r in group], [r['restore_s'] for r in group],
                  width=.34, color=color, label=label)
    ax.bar_label(bars, fmt='%.2f', padding=2)
ax.set_xticks([0, 1, 2], ['Pair 1', 'Pair 2', 'Pair 3'])
ax.set_ylabel('Restore time (seconds)')
ax.margins(y=.22)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Restoring 250,000 positions of DFlash2 hidden state', loc='left', weight='bold')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncols=2, frameon=False)
fig.text(.5, .105, 'llm-60 · synthetic BF16 taps · warm tmpfs · preallocated GPU destination', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .16, 1, 1))
fig.savefig(directory / 'dflash2-250k-restore.png', dpi=165, bbox_inches='tight')
