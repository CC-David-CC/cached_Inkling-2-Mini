"""Compare direct sibling suffix restores to uninterrupted Inkling continuations."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from context_time_travel.backend_registry import load_backend
from context_time_travel.digest_attestation import checkpoint_export_scope, restore_payload_digest
from context_time_travel.inkling_backend import MAX_CONTEXT_TOKENS, runtime_profile
from context_time_travel.real import BranchingModel, Checkpoint
from test_inkling_pcc import exact_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--pack', choices=('SQ-10', 'SQ-15', 'SQ-20'), default='SQ-10')
    parser.add_argument('--parent-tokens', type=int, default=1017)
    parser.add_argument('--child-tokens', type=int, default=2)
    args = parser.parse_args()
    capacity = args.parent_tokens + args.child_tokens + 1
    if min(args.parent_tokens, args.child_tokens) < 1 or capacity > MAX_CONTEXT_TOKENS:
        parser.error('positive parent and child lengths must leave room for one continuation token')
    history = tuple(100 + pos % 1000 for pos in range(args.parent_tokens))
    profile = runtime_profile(args.runtime.resolve(), pack=args.pack, capacity=max(16384, capacity))
    backend = load_backend(profile)
    try:
        with backend.lease() as lease:
            backend.consume(lease, history)
            with checkpoint_export_scope():
                parent = backend.export(lease)
                parent_row = backend.logits(lease)
                owner = Checkpoint.create(profile['model_contract_id'], profile['execution_profile_id'],
                                          len(history), parent, parent_row, BranchingModel._input_identity(history))
            digest = restore_payload_digest(owner.blob)
            children = []
            export_times = []
            for first in (110, 150):
                tokens = tuple(first + pos % 1000 for pos in range(args.child_tokens))
                backend.restore(lease, parent, history=history, frontier=parent_row)
                backend.consume(lease, tokens)
                started = perf_counter()
                suffix = backend.export_pinned_suffix(lease, len(history), digest)
                export_times.append(perf_counter() - started)
                assert suffix, 'suffix export unavailable'
                frontier = backend.logits(lease)
                backend.consume(lease, (501,))
                children.append((suffix, frontier, exact_row(backend.logits(lease))))
            assert children[0][2] != children[1][2]
            before = backend.work_counters()
            timings = []
            select_times = []
            for index in (1, 0, 1):
                suffix, row, expected = children[index]
                started = perf_counter()
                backend.restore(lease, parent, history=history, frontier=parent_row)
                select_times.append(perf_counter() - started)
                child_length = len(history) + args.child_tokens
                assert not backend.restore_pinned_suffix(lease, len(history), bytes(32), child_length, suffix, row)
                start = perf_counter()
                assert backend.restore_pinned_suffix(lease, len(history), digest, child_length, suffix, row)
                timings.append(perf_counter() - start)
                backend.consume(lease, (501,))
                assert exact_row(backend.logits(lease)) == expected
            after = backend.work_counters()
            transferred = after['checkpoint_import_bytes'] - before['checkpoint_import_bytes']
            assert transferred == sum(len(children[i][0]) for i in (1, 0, 1))
            assert after['consumed_tokens'] - before['consumed_tokens'] == 3
            print(json.dumps(dict(pack=args.pack, parent_tokens=len(history), child_tokens=args.child_tokens, parent_bytes=len(parent),
                                  import_bytes=transferred, restores=3, restore_seconds=timings,
                                  parent_select_seconds=select_times, suffix_export_seconds=export_times,
                                  exact_logits=backend.vocab_size)))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
