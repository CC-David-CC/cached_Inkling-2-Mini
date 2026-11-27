"""Compare observed native decode rates and remaining VRAM on three P4 workers."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
for name, free, shift, label, color in [
    ('p4-allhost-real-64k.csv', 'free_mib', -.18, '0 MiB', '#286480'),
    ('p4-synchronized-64k.csv', 'final_free_mib', .18, '768 MiB', '#ce823c'),
]:
    with (folder / name).open() as stream:
        rows = list(csv.DictReader(stream))
    for ax, metric in zip(axes, ['decode_tps' if shift < 0 else 'decode_tok_s', free]):
        bars = ax.bar([int(r['gpu']) + shift for r in rows], [float(r[metric]) for r in rows],
                      width=.34, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f' if ax is axes[0] else '%.0f', fontsize=9)
for ax, title in zip(axes, ['Decode tokens/s per stream', 'Free VRAM after decode (MiB)']):
    ax.set_title(title)
    ax.set_xticks(range(3), ['P4 0', 'P4 1', 'P4 2'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(y=.15)
axes[0].legend(title='GPU expert budget', loc='lower right')
fig.suptitle('SQ-10 · 65,504 real prompt tokens + 32 synchronized decode steps per GPU')
fig.savefig(folder / 'p4-real-64k-budget.png', dpi=160)
