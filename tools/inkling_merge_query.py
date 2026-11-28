"""Build the pinned SQ-20 layer-zero Muon serving pack from its two inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental/branching_context/src'))
from context_time_travel.inkling_backend import MODEL_SHA256, QUERY_MUON_PACK, SERVING_PACK_HASHES
from context_time_travel.real import digest_file


def build(base, adapter, destination):
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if digest_file(base) != MODEL_SHA256:
        raise ValueError('Expected the pinned SQ-20 base')
    raw = adapter.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '124e3ec997fedf1fd3ec37b218ffcffca602ead7a25cb6d0c064dfec79a247b1':
        raise ValueError('Expected the qualified rank-4 Muon layer-zero safetensors export')
    header_size, = struct.unpack_from('<Q', raw)
    tensors = json.loads(raw[8:8 + header_size])

    def factor(side):
        key = f'base_model.model.model.llm.layers.0.attn.wq_du.lora_{side}.weight'
        entry = tensors[key]
        start, end = entry['data_offsets']
        data = memoryview(raw)[8 + header_size + start:8 + header_size + end]
        return np.frombuffer(data, dtype='<f4').reshape(entry['shape'])

    # Offsets and dimensions are bound by the complete SQ-20 base fingerprint.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.strata') as out:
        with base.open('rb') as inp:
            shutil.copyfileobj(inp, out, 8 << 20)
        out.seek(80)
        trunk, length = struct.unpack('<QQ', out.read(16))
        projection = trunk + 960207360
        out.seek(projection)
        weights = np.frombuffer(out.read(3072 * 2304 * 2), dtype='<u2')
        fp32 = (weights.astype('<u4') << 16).view('<f4').reshape(3072, 2304)
        bits = (fp32 + factor('B') @ factor('A')).view('<u4')
        bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')
        out.seek(projection)
        out.write(bf16.tobytes())
        out.seek(trunk)
        digest = hashlib.sha256()
        while length:
            chunk = out.read(min(length, 8 << 20))
            if not chunk:
                raise ValueError('Truncated trunk section')
            digest.update(chunk)
            length -= len(chunk)
        out.seek(96)
        out.write(digest.digest())
        out.flush()
        if digest_file(out.name) != SERVING_PACK_HASHES[QUERY_MUON_PACK]:
            raise ValueError('Merged pack fingerprint differs from the qualified build')
        os.link(out.name, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('adapter', type=Path)
    args = parser.parse_args()
    build(args.runtime / 'Inkling-2-Mini-SQ-20.strata', args.adapter,
          args.runtime / f'Inkling-2-Mini-{QUERY_MUON_PACK}.strata')
