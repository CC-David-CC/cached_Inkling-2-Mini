"""Exercise PCC checkpoint persistence across two Inkling CUDA processes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import runtime_profile as profile
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, _json_bytes


def exact_row(logits):
    return struct.pack(f'<{len(logits)}f', *logits)


def assert_rejected(call, reason):
    try:
        call()
    except IntegrityError as error:
        assert reason in str(error), error
        return
    raise AssertionError('Invalid checkpoint was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('mode', choices=('write', 'read'))
    args = parser.parse_args()
    workspace = args.workspace
    workspace.mkdir(exist_ok=True)
    config = profile(args.runtime.resolve())
    backend = load_backend(config)
    try:
        model = BranchingModel(workspace / 'pcc', backend, config)
        prefix = tuple(100 + pos % 1000 for pos in range(4097))
        token = 100 + len(prefix) % 1000
        if args.mode == 'write':
            with backend.lease() as lease:
                backend.consume(lease, prefix)
                frontier = exact_row(backend.logits(lease))
                backend.consume(lease, (token,))
                expected = exact_row(backend.logits(lease))
            root = model.create(prefix)
            before = backend.work_counters()
            imports = backend.stats['checkpoint_import_bytes']
            assert exact_row(model.next_logits(root)) == frontier
            assert backend.stats['checkpoint_import_bytes'] == imports
            assert backend.work_counters() == before
            model.feed(root, (291, 292))
            branch = model.feed(root, (token,))
            assert exact_row(model.next_logits(branch)) == expected
            assert exact_row(model.next_logits(root)) == frontier
            assert backend.work_counters()['consumed_tokens'] == before['consumed_tokens'] + 3
            files = model.checkpoint(root)
            (workspace / 'root').write_text(root)
            (workspace / 'frontier').write_bytes(frontier)
            (workspace / 'continuation').write_bytes(expected)
            print(f'RAM and sibling checks passed; disk payload {files["payload_path"].stat().st_size} bytes')
        else:
            root = (workspace / 'root').read_text()
            manifest = model.checkpoint_manifest(root)
            for field in ('execution_profile_id', 'model_contract_id', 'input_identity'):
                incompatible = dict(manifest, **{field: '0' * 64})
                envelope = dict(manifest=incompatible,
                                checksum=hashlib.sha256(_json_bytes(incompatible)).hexdigest())
                assert_rejected(lambda: model.validate_checkpoint_envelope(root, envelope),
                                'Checkpoint manifest incompatible')
            payload = model.checkpoint_directory / (manifest['payload_sha256'] + '.state')
            with payload.open('r+b') as stream:
                original = stream.read(1)
                stream.seek(0)
                stream.write(bytes([original[0] ^ 1]))
            try:
                assert_rejected(lambda: model.restore_checkpoint(root), 'Checkpoint payload corruption')
            finally:
                with payload.open('r+b') as stream:
                    stream.write(original)
            assert backend.work_counters()['consumed_tokens'] == 0
            receipt = model.restore_checkpoint(root)
            assert receipt['forward_tokens'] == 0
            assert exact_row(model.next_logits(root)) == (workspace / 'frontier').read_bytes()
            branch = model.feed(root, (token,))
            assert exact_row(model.next_logits(branch)) == (workspace / 'continuation').read_bytes()
            assert backend.work_counters()['consumed_tokens'] == 1
            print(f'Fresh restore and rejection checks passed; restore {receipt["seconds"]:.3f}s')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
