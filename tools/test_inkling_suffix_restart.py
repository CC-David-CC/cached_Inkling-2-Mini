"""Check a persisted native suffix against its canonical parent in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, _json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('store', type=Path)
    parser.add_argument('phase', choices=('write', 'read'))
    parser.add_argument('--parent-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=1)
    args = parser.parse_args()
    capacity = args.parent_tokens + 2 + args.output_tokens
    if min(args.parent_tokens, args.output_tokens) < 1 or capacity > 133120:
        parser.error('Positive parent and output lengths plus the two-token child must fit in 133120 tokens')
    profile = runtime_profile(args.runtime.resolve(), pack='SQ-10', capacity=max(16384, capacity))
    backend = load_backend(profile)
    try:
        model = BranchingModel(args.store, backend, profile)
        fixture = args.store / 'suffix-restart.json'
        if args.phase == 'write':
            parent = model.create(tuple(100 + index % 1000 for index in range(args.parent_tokens)))
            model.publish_prepared_checkpoint(model.prepare_checkpoint(parent))
            child = model.feed(parent, (110, 111))
        else:
            saved = json.loads(fixture.read_bytes())
            child = saved['child']
        parent = model.store.get_snapshot(child)['parent']
        parent_history = model.history(parent)
        row_format = f'<{backend.vocab_size}f'
        if args.phase == 'write':
            prepared = model.prepare_suffix_checkpoint(child)
            first = model.publish_suffix_checkpoint(prepared)
            assert model.publish_suffix_checkpoint(prepared) == first
            with model.materialize(child) as (lease, _):
                backend.consume(lease, (501,))
                expected = hashlib.sha256(struct.pack(row_format, *backend.logits(lease))).hexdigest()
            frontier_hash = hashlib.sha256(struct.pack(row_format, *prepared.checkpoint.frontier)).hexdigest()
            answer = model.run_greedy_request(child, (), args.output_tokens, stream='suffix-restart')
            result_hash = model.cache[answer['snapshot']].checksum
            fixture.write_text(json.dumps(dict(child=child, expected=expected, frontier_hash=frontier_hash,
                                               greedy_tokens=answer['tokens'], result_hash=result_hash)))
            return

        envelope = json.loads((model.checkpoint_directory / (child + '.json')).read_bytes())
        altered = dict(envelope['manifest'], parent_payload_sha256='0' * 64)
        before = backend.work_counters()
        try:
            with model.materialize_suffix(child, dict(manifest=altered, checksum=hashlib.sha256(_json_bytes(altered)).hexdigest())):
                raise AssertionError('Invalid suffix reached a live lease')
        except IntegrityError:
            pass
        else:
            raise AssertionError('Rechecksummed child accepted the wrong parent payload')
        assert before == backend.work_counters()
        frontier = model.next_logits(child)
        assert hashlib.sha256(struct.pack(row_format, *frontier)).hexdigest() == saved['frontier_hash']
        assert backend.work_counters() == before
        assert child not in model.cache and parent not in model.cache
        suffix_bytes = envelope['manifest']['state_bytes']
        with model.materialize(child) as (lease, _):
            backend.consume(lease, (501,))
            assert hashlib.sha256(struct.pack(row_format, *backend.logits(lease))).hexdigest() == saved['expected']
        assert backend.work_counters()['consumed_tokens'] == 1
        del model.cache[child]
        before_generate = backend.work_counters()
        answer = model.run_greedy_request(child, (), args.output_tokens, stream='suffix-restart')
        assert child not in model.cache
        assert answer['tokens'] == saved['greedy_tokens']
        assert model.cache[answer['snapshot']].checksum == saved['result_hash']
        assert 0 < len(answer['tokens']) <= args.output_tokens
        assert backend.work_counters()['consumed_tokens'] - before_generate['consumed_tokens'] == len(answer['tokens'])
        print(json.dumps(dict(parent_tokens=len(parent_history), suffix_bytes=suffix_bytes,
                              exact_continuation_logits=backend.vocab_size, saved_tokens_replayed=0,
                              requested_tokens=args.output_tokens, generated_tokens=len(answer['tokens']),
                              final_payload_sha256=saved['result_hash'],
                              generation_seconds=answer['generation_seconds'],
                              state_save_seconds=answer['state_save_seconds'], request_seconds=answer['request_seconds'],
                              wrong_parent_rejected=True, frontier_lookup_import_bytes=0, greedy_source_cached=False)))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
