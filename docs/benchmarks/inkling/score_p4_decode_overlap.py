"""Summarize three native P4 decode logs and compare their tokens with a reference run."""

import argparse
import json
from pathlib import Path
import re


def read_worker(path):
    log = path.read_text()
    times = re.search(r"^start_ns=(\d+) end_ns=(\d+)$", log, re.MULTILINE)
    final = re.search(r"^final_position=65535 finite_logits=200058 remaining_mib=(\d+)$", log, re.MULTILINE)
    tokens = list(map(int, re.findall(r"^output_id=(\d+)$", log, re.MULTILINE)))
    if not times or not final or len(tokens) != 32:
        raise ValueError(f"Incomplete 64k decode in {path}")
    start, end = map(int, times.groups())
    if end <= start:
        raise ValueError(f"Invalid decode timestamps in {path}")
    return start, end, tokens, int(final[1])


def summarize(run, reference):
    workers = []
    starts, ends = [], []
    for gpu in range(3):
        name = f"{gpu}.log"
        start, end, tokens, free = read_worker(run / name)
        starts.append(start)
        ends.append(end)
        workers.append(dict(gpu=gpu, tokens=len(tokens), final_free_mib=free,
                            decode_tok_s=len(tokens) * 1e9 / (end - start),
                            matches_reference=tokens == read_worker(reference / name)[2]))
    overlap = max(0, min(ends) - max(starts))
    elapsed = max(ends) - min(starts)
    return dict(workers=workers, three_way_overlap_s=overlap / 1e9,
                elapsed_s=elapsed / 1e9, overlap_fraction=overlap / elapsed,
                aggregate_tok_s=sum(w["tokens"] for w in workers) * 1e9 / elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.reference), indent=2))
