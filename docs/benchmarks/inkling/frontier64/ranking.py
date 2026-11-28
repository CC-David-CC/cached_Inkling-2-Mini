"""Compare the final next-token distributions at temperature 1, before filtering."""
import csv
import itertools
import math
import struct
import sys
from pathlib import Path


def distribution(case):
    raw = Path(f"frontier64-{case}-final.f32").read_bytes()
    logits = struct.unpack("<200058f", raw)
    if not all(map(math.isfinite, logits)):
        raise ValueError(f"Nonfinite scores for {case}")
    order = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    peak = logits[order[0]]
    weights = [math.exp(score - peak) for score in logits]
    total = math.fsum(weights)
    return order, [weight / total for weight in weights]


cases = {case: distribution(case) for case in ("off1", "off2", "dspark1", "dspark2")}
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["left", "right", "winning_id_left", "winning_id_right", "top10_shared",
                 "top10_same_rank", "winner_probability_left", "winner_probability_right",
                 "total_variation"])
for left, right in itertools.combinations(cases, 2):
    a, p = cases[left]
    b, q = cases[right]
    variation = math.fsum(abs(x - y) for x, y in zip(p, q)) / 2
    writer.writerow([left, right, a[0], b[0], len(set(a[:10]) & set(b[:10])),
                     sum(x == y for x, y in zip(a[:10], b[:10])),
                     p[a[0]], q[b[0]], variation])
