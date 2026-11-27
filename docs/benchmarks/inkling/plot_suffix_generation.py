"""Plot the timed greedy request after the fixed 128k suffix restart control."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
run = json.loads((folder / 'suffix-128k-generation.json').read_text())
tokens = run['generated_tokens']
decode = run['generation_seconds']
save = run['state_save_seconds']
total = run['request_seconds']
segments = [('Decode', decode, '#21738c'), ('Capture output state', save, '#457c56'),
            ('Remaining request time', total - decode - save, '#95622d')]
fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
bars = ax.barh([label for label, _, _ in segments], [seconds for _, seconds, _ in segments],
               color=[color for _, _, color in segments], height=.52)
ax.bar_label(bars, labels=[f'{seconds:.2f} s' for _, seconds, _ in segments], padding=5)
ax.set_xlim(0, max(seconds for _, seconds, _ in segments) * 1.18)
ax.set_xlabel('Seconds per model request')
ax.invert_yaxis()
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(length=0)
ax.set_title(f'{tokens / decode:.1f} decode tok/s  ·  {1000 * decode / tokens:.1f} ms per output token'
             f'  ·  {total:.2f} s total', fontsize=11, loc='left')
fig.suptitle('512-token continuation from a 131,072-token parent\n'
             'llm-60 · SQ-10 · greedy · drafting off · one timed request', fontsize=13)
fig.savefig(folder / 'suffix-128k-generation.png', dpi=170)
