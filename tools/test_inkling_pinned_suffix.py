"""Compare direct sibling suffix restores to uninterrupted Inkling continuations."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from context_time_travel.backend_registry import load_backend
from context_time_travel.digest_attestation import checkpoint_export_scope, restore_payload_digest
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.real import BranchingModel, Checkpoint
from test_inkling_pcc import exact_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--parent-tokens', type=int, default=1017)
    args = parser.parse_args()
    if not 1 <= args.parent_tokens <= 133117:
        parser.error('parent must leave room for three continuation tokens')
    history = tuple(100 + pos % 1000 for pos in range(args.parent_tokens))
    profile = runtime_profile(args.runtime.resolve(), pack='SQ-10', capacity=max(16384, len(history) + 3))
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
            for tokens in [(110, 111), (150, 151)]:
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
                child_length = len(history) + 2
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
            print(json.dumps(dict(parent_tokens=len(history), parent_bytes=len(parent),
                                  import_bytes=transferred, restores=3, restore_seconds=timings,
                                  parent_select_seconds=select_times, suffix_export_seconds=export_times,
                                  exact_logits=backend.vocab_size)))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
