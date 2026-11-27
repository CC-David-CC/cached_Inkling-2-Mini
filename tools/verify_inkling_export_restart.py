"""Verify a native Inkling export checksum and continuation in a fresh process."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental/branching_context/src"))
from context_time_travel.backend_registry import load_backend
from context_time_travel.digest_attestation import checkpoint_export_scope
from context_time_travel.inkling_backend import runtime_profile
from context_time_travel.real import Checkpoint

PREFIX = [19, 97948, 19]
CONTINUATION = [97948, 19]


def row_bytes(row):
    return struct.pack(f"<{len(row)}f", *row)


def continuation(backend, lease):
    digests = []
    for token in CONTINUATION:
        backend.consume_generated(lease, [token])
        digests.append(hashlib.sha256(row_bytes(backend.logits(lease))).hexdigest())
    return digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--pack", choices=["SQ-10", "SQ-15", "SQ-20"], default="SQ-10")
    parser.add_argument("--read", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    profile = runtime_profile(args.runtime.resolve(), pack=args.pack, capacity=16384)
    if args.read:
        folder = args.read
        blob = (folder / "state").read_bytes()
        raw = (folder / "frontier").read_bytes()
        expected = json.loads((folder / "expected.json").read_text())
        frontier = struct.unpack(f"<{len(raw) // 4}f", raw)
        if Checkpoint.digest(blob, frontier) != expected["checksum"]:
            raise ValueError("Checkpoint checksum differs after restart")
        with closing(load_backend(profile)) as backend, backend.lease() as lease:
            backend.restore(lease, blob, history=PREFIX, frontier=frontier)
            if row_bytes(backend.logits(lease)) != raw:
                raise ValueError("Restored frontier differs")
            if continuation(backend, lease) != expected["continuation"]:
                raise ValueError("Continuation frontier differs")
        print(json.dumps(dict(pack=args.pack, state_bytes=len(blob), logits=len(frontier),
                              exact=True, **expected)))
        return

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        with closing(load_backend(profile)) as backend, backend.lease() as lease:
            backend.consume(lease, PREFIX)
            frontier = backend.logits(lease)
            with checkpoint_export_scope():
                blob = backend.export(lease)
                checkpoint = Checkpoint.create(profile["model_contract_id"],
                                               profile["execution_profile_id"],
                                               len(PREFIX), blob, frontier)
            (folder / "state").write_bytes(blob)
            (folder / "frontier").write_bytes(row_bytes(frontier))
            (folder / "expected.json").write_text(json.dumps(dict(
                checksum=checkpoint.checksum, continuation=continuation(backend, lease))))
        subprocess.run([sys.executable, __file__, str(args.runtime), "--pack", args.pack,
                        "--read", directory], check=True)


if __name__ == "__main__":
    main()
