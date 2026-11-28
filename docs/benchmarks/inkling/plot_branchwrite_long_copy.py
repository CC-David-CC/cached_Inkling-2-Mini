"""Plot client latency and backend generation rates for the paired copy tasks."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (folder / 'branchwrite-sq20-130k-long-copy.jsonl').read_text().splitlines()]
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
fig.subplots_adjust(top=.75, bottom=.19, wspace=.43)
fig.suptitle('Two questions about one 130,000-token register', fontsize=17, x=.06, ha='left')
fig.text(.06, .83, 'BranchWrite + Cached SQ-20  ·  RTX PRO 6000  ·  353 generated tokens per answer', fontsize=10)
labels = ['Client ask (seconds)', 'Generation (tokens/s)', 'Generation (ms/token)']
for index, (mode, color, label) in enumerate([(1, '#296b80', 'Serial'), (2, '#b55a32', 'Overlap')]):
    group = sorted((r for r in rows if r['topology'] == f'sq20-long-copy-{mode}-clients'), key=lambda r: r['station'])
    assert len(group) == 2
    for station, r in enumerate(group):
        a = r['answer']
        rate = len(a['tokens']) / a['generation_seconds']
        for ax, value in zip(axes, (r['end'] - r['start'], rate, 1000 / rate)):
            pos = index + (station - .5) * .29
            ax.bar(pos, value, width=.25, color=color, alpha=1 if station == 0 else .68)
            ax.text(pos, value, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
for ax, title in zip(axes, labels):
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 1], ['Serial', 'Overlap'])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.13)
    ax.spines[['top', 'right']].set_visible(False)
fig.text(.06, .045, 'One pair per mode. Bars in each pair show stations 00017 and 05712. Client times exclude branch creation.', fontsize=9)
fig.savefig(folder / 'branchwrite-sq20-130k-long-copy.png', dpi=180)
