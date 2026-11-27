"""Check a persisted native suffix against its canonical parent in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from context_time_travel.backend_registry import load_backend
from context_time_travel.digest_attestation import restore_payload_digest
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, _json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('store', type=Path)
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    profile = runtime_profile(args.runtime.resolve(), pack='SQ-10', capacity=16384)
    backend = load_backend(profile)
    try:
        model = BranchingModel(args.store, backend, profile)
        fixture = args.store / 'suffix-restart.json'
        if args.phase == 'write':
            parent = model.create(tuple(range(100, 228)))
            model.publish_prepared_checkpoint(model.prepare_checkpoint(parent))
            child = model.feed(parent, (110, 111))
        else:
            saved = json.loads(fixture.read_bytes())
            child = saved['child']
        parent = model.store.get_snapshot(child)['parent']
        history = model.history(child)
        parent_history = model.history(parent)
        required = dict(schema=5, snapshot=child, consumed=len(history),
                        input_identity=model._input_identity(history), positions='absolute-zero-based',
                        logits_count=backend.vocab_size, model_contract_id=profile['model_contract_id'],
                        execution_profile_id=profile['execution_profile_id'], parent_snapshot=parent,
                        parent_consumed=len(parent_history),
                        parent_payload_sha256=model.checkpoint_manifest(parent)['payload_sha256'])
        row_format = f'<{backend.vocab_size}f'
        if args.phase == 'write':
            with model.materialize(child) as (lease, row):
                digest = restore_payload_digest(model.cache[parent].blob)
                suffix = backend.export_pinned_suffix(lease, len(parent_history), digest)
                assert suffix
                data = suffix + struct.pack(row_format, *row)
                manifest = dict(required, state_bytes=len(suffix), payload_sha256=hashlib.sha256(data).hexdigest())
                envelope = dict(manifest=manifest, checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
                model._publish_checkpoint_file(model.checkpoint_directory / (manifest['payload_sha256'] + '.state'), data)
                backend.consume(lease, (501,))
                expected = hashlib.sha256(struct.pack(row_format, *backend.logits(lease))).hexdigest()
            fixture.write_text(json.dumps(dict(child=child, envelope=envelope, expected=expected)))
            return

        envelope = saved['envelope']
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
        suffix_bytes = envelope['manifest']['state_bytes']
        with model.materialize_suffix(child, envelope) as (lease, _):
            backend.consume(lease, (501,))
            assert hashlib.sha256(struct.pack(row_format, *backend.logits(lease))).hexdigest() == saved['expected']
        assert backend.work_counters()['consumed_tokens'] == 1
        print(json.dumps(dict(parent_tokens=len(parent_history), suffix_bytes=suffix_bytes,
                              exact_continuation_logits=backend.vocab_size, saved_tokens_replayed=0,
                              wrong_parent_rejected=True)))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
