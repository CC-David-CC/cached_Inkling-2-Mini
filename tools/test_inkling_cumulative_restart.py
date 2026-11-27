"""Check a cumulative suffix against a stored full second-turn checkpoint in a new worker."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from context_time_travel.backend_registry import load_backend
from context_time_travel.digest_attestation import restore_payload_digest
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.real import BranchingModel


def check(args, scratch, phase):
    profile = runtime_profile(args.runtime.resolve(), capacity=args.capacity, pack=args.pack, serial_followup=True)
    backend = load_backend(profile)
    try:
        model = BranchingModel(args.store, backend, profile)
        full = model.checkpoint_manifest(args.snapshot)
        record = model.store.get_snapshot(args.snapshot)
        child = record['parent']
        root = model.store.get_snapshot(child)['parent']
        row_format = f'<{backend.vocab_size}f'
        suffix_file = scratch / 'suffix.bin'
        row_file = scratch / 'frontier.bin'
        assert full['schema'] == 1 and model.suffix_restore_bytes(child) > 0
        if phase == 'capture':
            with model.materialize_suffix(child) as (lease, _):
                parent = model.cache[root]
                for token in record['inputs']:
                    backend.consume(lease, (token,))
                assert backend.position(lease) + 1 == full['consumed']
                suffix = backend.export_pinned_suffix(lease, parent.consumed, restore_payload_digest(parent.blob))
                assert suffix
                suffix_file.write_bytes(suffix)
                row_file.write_bytes(struct.pack(row_format, *backend.logits(lease)))
            return

        with model.materialize(root) as (lease, _):
            parent = model.cache[root]
            suffix = suffix_file.read_bytes()
            row = struct.unpack(row_format, row_file.read_bytes())
            assert backend.restore_pinned_suffix(lease, parent.consumed, restore_payload_digest(parent.blob),
                                                 full['consumed'], suffix, row)
            size = backend.lib.ctt_state_size(lease)
            assert size == full['state_bytes']
            with tempfile.TemporaryFile(dir=scratch) as stream:
                stream.truncate(size)
                with mmap.mmap(stream.fileno(), size) as buf:
                    with memoryview(buf) as view:
                        backend._export_into(lease, view)
                    digest = hashlib.sha256(buf)
            digest.update(struct.pack(row_format, *backend.logits(lease)))
            assert digest.hexdigest() == full['payload_sha256']
            assert backend.work_counters()['forward_tokens'] == 0
            print(json.dumps(dict(full_bytes=size, suffix_bytes=len(suffix), sha256=digest.hexdigest(),
                                  restored_tokens=full['consumed'], replay_tokens=0)))
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('store', type=Path)
    parser.add_argument('snapshot', help='full second-turn snapshot whose parent is a suffix')
    parser.add_argument('--capacity', type=int, default=262144)
    parser.add_argument('--pack', default='SQ-10')
    parser.add_argument('--scratch-parent', type=Path, help='filesystem with space for one full export')
    parser.add_argument('--worker', nargs=2, metavar=('PHASE', 'DIRECTORY'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        phase, directory = args.worker
        check(args, Path(directory), phase)
    else:
        with tempfile.TemporaryDirectory(dir=args.scratch_parent) as work:
            for phase in ('capture', 'restore'):
                subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', phase, work], check=True)


if __name__ == '__main__':
    main()
