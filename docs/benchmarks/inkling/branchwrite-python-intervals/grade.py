from itertools import combinations_with_replacement, product
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from intervals import merge

# Half-unit probes distinguish adjacent integer endpoints from touching intervals.
def oracle(intervals):
    covered = {point for a, b in intervals for point in range(2*a, 2*b+1)}
    return list(zip(sorted(p//2 for p in covered if p-1 not in covered),
                    sorted(p//2 for p in covered if p+1 not in covered)))

cases = list(combinations_with_replacement(range(-1, 3), 2))
counts = {'correct': 0, 'incorrect': 0}
for length in range(4):
    for case in product(cases, repeat=length):
        supplied = list(case)
        expected = oracle(case)
        actual = merge(supplied)
        counts['correct' if actual == expected and supplied == list(case) else 'incorrect'] += 1
print(json.dumps(counts))
sys.exit(bool(counts['incorrect']))
