"""Read and compare the stored FP32 frontiers for a recovery receipt."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

def compare(directory, receipt):
    hashes = receipt['uninterrupted']['control_hashes'] + [receipt['recovery']['surviving_payload']]
    manifests = {}
    for path in directory.glob('*.json'):
        row = json.loads(path.read_bytes()).get('manifest', {})
        if row.get('payload_sha256') in hashes:
            manifests[row['payload_sha256']] = row
    rows = []
    for digest in hashes:
        m = manifests[digest]
        payload = (directory / (digest + '.state')).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        state = payload[:m['state_bytes']]
        frontier = payload[len(state):]
        assert len(frontier) == 4 * m['logits_count']
        scores = struct.unpack(f"<{m['logits_count']}f", frontier)
        assert all(map(math.isfinite, scores))
        if not rows:
            first, reference, identity = frontier, scores, m
        assert all(m[k] == identity[k] for k in ('input_identity','execution_profile_id','model_contract_id','consumed','logits_count'))
        rows.append(dict(payload=digest, state_sha256=hashlib.sha256(state).hexdigest(),
                         frontier_sha256=hashlib.sha256(frontier).hexdigest(),
                         logits=len(scores), argmax=max(range(len(scores)), key=scores.__getitem__),
                         bitwise_different_logits=sum(x != y for x,y in zip(struct.iter_unpack('<I',first),struct.iter_unpack('<I',frontier))),
                         max_absolute_delta=max(abs(x-y) for x,y in zip(reference,scores))))
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.checkpoints, json.loads(args.receipt.read_bytes())), indent=2))
