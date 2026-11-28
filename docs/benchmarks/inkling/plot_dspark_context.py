"""Plot native single-request decode speed and prefill time across prompt lengths."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
for mode, label, color in [('off', 'Draft off', '#315c83'), ('dspark', 'DSpark', '#14786b')]:
    rows = []
    for length in ('64k', '128k', '250k'):
        source = base / f'dspark-{length}-generation.jsonl'
        records = [json.loads(line) for line in source.read_text().splitlines()]
        matches = [record for record in records if record['mode'] == mode]
        if not matches:
            continue
        row, = matches
        assert row['tokens'] == 512
        rows.append(row)
    for ax, key in zip(axes, ('prefill_s', 'generated_tps')):
        ax.plot([r['prompt']/1000 for r in rows], [r[key] for r in rows], marker='o',
                color=color, label=label)
for filename, label, color in [
    ('off-250k-2gib.json', 'Draft off, 2 GiB', '#315c83'),
    ('dspark-250k-2gib.json', 'DSpark, 2 GiB', '#14786b'),
]:
    row = json.loads((base / filename).read_text())
    for ax, metric in zip(axes, ('prefill_s', 'generated_tps')):
        ax.scatter(row['prompt'] / 1000, row[metric], marker='D',
                   facecolors='none', edgecolors=color, label=label, zorder=3)
for ax, title in zip(axes, ('Time to read the prompt (s)', 'Generation (output tokens/s)')):
    ax.set(title=title, xlabel='Input tokens (thousands)', xticks=[65.536,131.072,250],
           xticklabels=['65.5','131.1','250'], ylim=(0,None))
    ax.spines[['right','top']].set_visible(False)
fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
axes[1].text(.98, .46, '250k DSpark at 4 GiB: out of VRAM', transform=axes[1].transAxes, ha='right', fontsize=9)
fig.suptitle('llm-60 · SQ-10 · 4 GiB pool unless labeled · 512 output tokens per run')
fig.savefig(base/'dspark-context.png',dpi=160)
