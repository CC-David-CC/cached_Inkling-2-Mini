"""Compare resident parent/child transitions to independently imported GPU states."""
import argparse
import json
from pathlib import Path
from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import runtime_profile


def main(runtime, prefix_tokens):
    profile = runtime_profile(runtime, capacity=max(16384, prefix_tokens + 512))
    backend = load_backend(profile)
    parent = tuple(100 + pos % 3 for pos in range(prefix_tokens))
    children = []
    try:
        with backend.lease() as lease:
            backend.consume(lease, parent)
            blob = backend.export(lease)
            frontier = backend.logits(lease)
        for token in (104, 105):
            with backend.lease() as lease:
                backend.restore(lease, blob, history=parent, frontier=frontier)
                backend.consume(lease, [token])
                child = backend.export(lease)
                row = backend.logits(lease)
            with backend.lease() as lease:
                backend.restore(lease, child, history=parent+(token,), frontier=row)
                backend.consume(lease, [106])
                children.append((token, child, row, backend.logits(lease)))
        hot = backend.work_counters()
        assert hot['resident_parent_restores'] == hot['resident_child_restores'] == 2
        assert hot['checkpoint_import_bytes'] == 0
    finally:
        backend.close()
    # Independent cold imports check the entire next-token frontier after each child restore.
    for token, child, row, expected in children:
        cold = load_backend(profile)
        try:
            with cold.lease() as lease:
                cold.restore(lease, child, history=parent+(token,), frontier=row)
                cold.consume(lease, [106])
                assert cold.logits(lease) == expected
                assert cold.work_counters()['checkpoint_import_bytes'] == len(child)
        finally:
            cold.close()
    print(json.dumps(dict(prefix_tokens=len(parent), logits_per_child=len(expected), counters=hot)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--prefix-tokens', type=int, default=9000)
    args = parser.parse_args()
    main(args.runtime.resolve(), args.prefix_tokens)
