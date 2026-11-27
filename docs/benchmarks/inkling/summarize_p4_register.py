"""Summarize a synchronized three-P4 65,504-token register run from its native logs."""
import argparse
import csv
from pathlib import Path
import re
import sys


def summarize(directory, budget_mib):
    workers = []
    for gpu in range(3):
        log = (directory / f'{gpu}.log').read_text()
        if 'final_position=65535 finite_logits=200058' not in log:
            raise ValueError(f'GPU {gpu} did not finish its 64K context')
        tokens = re.findall(r'^output_id=(\d+)$', log, re.M)
        if len(tokens) != 32:
            raise ValueError(f'GPU {gpu} emitted {len(tokens)} tokens')
        start, end = map(int, re.search(r'start_ns=(\d+) end_ns=(\d+)', log).groups())
        if end <= start:
            raise ValueError(f'Invalid decode interval on GPU {gpu}')
        workers.append(dict(gpu=gpu, budget_mib=budget_mib, prompt_tokens=65504,
                            prefill_s=float(re.search(r'prefill_seconds=([\d.]+)', log)[1]),
                            output_tokens=len(tokens), start_ns=start, end_ns=end,
                            free_mib=int(re.search(r'finite_logits=200058 remaining_mib=(\d+)', log)[1])))
    start = min(w['start_ns'] for w in workers)
    end = max(w['end_ns'] for w in workers)
    overlap = min(w['end_ns'] for w in workers) - max(w['start_ns'] for w in workers)
    if overlap <= 0:
        raise ValueError('No common three-stream decode interval')
    for w in workers:
        w['decode_tps'] = w['output_tokens'] * 1e9 / (w['end_ns'] - w['start_ns'])
        w['aggregate_tps'] = sum(row['output_tokens'] for row in workers) * 1e9 / (end - start)
        w['overlap_fraction'] = overlap / (end - start)
    return workers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path)
    parser.add_argument('budget_mib', type=int)
    args = parser.parse_args()
    rows = summarize(args.logs, args.budget_mib)
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
