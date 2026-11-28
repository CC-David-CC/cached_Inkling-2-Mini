"""Measure repeated one-token decode after identical native checkpoint imports.

Run in a separate process per STRATA_INK_EXACT setting. This diagnostic uses
an operator-selected native checkpoint from a compatible build and does not
exercise the serving layer's checkpoint admission or resident reuse.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct

from context_time_travel.backend_registry import load_backend
from context_time_travel.checkpoint_cache import validate_manifest
from context_time_travel.inkling_backend import runtime_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--token', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=16)
    args = parser.parse_args()
    lane = os.environ.get('STRATA_INK_EXACT')
    if lane not in ('0', '1') or args.iterations < 2:
        parser.error('select STRATA_INK_EXACT=0 or 1 and at least two iterations')
    profile = runtime_profile(args.runtime.resolve(), capacity=args.capacity, pack='SQ-20')
    envelope = json.loads(args.manifest.read_bytes())
    manifest = validate_manifest(envelope, dict(schema=1, model_contract_id=profile['model_contract_id']))
    payload = args.manifest.with_name(manifest['payload_sha256'] + '.state').read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['payload_sha256']:
        raise ValueError('Checkpoint digest mismatch')
    size = manifest['state_bytes']
    count = manifest['logits_count']
    if len(payload) != size + 4 * count or not 0 <= args.token < count:
        raise ValueError('Invalid checkpoint geometry or continuation token')
    state = payload[:size]
    row = (ctypes.c_float * count).from_buffer_copy(payload[size:])
    backend = load_backend(profile)
    try:
        with backend.lease() as lease:
            for i in range(args.iterations):
                imported = backend.lib.ctt_import_checkpoint(
                    lease, ctypes.c_char_p(state), size, manifest['consumed'], row, count)
                if imported != size:
                    raise RuntimeError('Native checkpoint import failed')
                backend.consume(lease, (args.token,))
                scores = struct.pack(f'<{count}f', *backend.logits(lease))
                print(json.dumps(dict(iteration=i, exact=lane,
                                      sha256=hashlib.sha256(scores).hexdigest())), flush=True)
    finally:
        backend.close()


if __name__ == '__main__':
    main()
