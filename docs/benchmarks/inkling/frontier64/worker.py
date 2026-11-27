import argparse
import gzip
import hashlib
import json
import struct
import time
from pathlib import Path

from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import runtime_profile

parser = argparse.ArgumentParser(
    description="Capture both full score rows for one native worker."
)
parser.add_argument("runtime", type=Path)
parser.add_argument("case", choices=("off1", "off2", "dspark1", "dspark2"))
args = parser.parse_args()
raw = gzip.decompress(Path(__file__).with_name("register.u32le.gz").read_bytes())
if (
    hashlib.sha256(raw).hexdigest()
    != "d8b1c465356d27abeeb9a97d14b337a3c5020b078c5e1fcfd3565fb1dd910e41"
):
    raise ValueError("Register fixture digest mismatch")
ids = struct.unpack("<65536I", raw)
mode = "dspark" if args.case.startswith("dspark") else "off"
profile = runtime_profile(
    args.runtime.resolve(), capacity=66560, pack="SQ-10", draft=mode == "dspark"
)
backend = load_backend(profile)


def capture(lease, stage):
    row = struct.pack("<200058f", *backend.logits(lease))
    Path(f"frontier64-{args.case}-{stage}.f32").write_bytes(row)


try:
    with backend.lease() as lease:
        start = time.monotonic()
        for i in range(0, len(ids), 8192):
            backend.consume(lease, ids[i : i + 8192])
        prefill = time.monotonic() - start
        capture(lease, "prefill")
        before = backend.work_counters()
        start = time.monotonic()
        with backend.generate_request(lease, 512) as stream:
            output = list(stream)
        elapsed = time.monotonic() - start
        capture(lease, "final")
        if len(output) != 512:
            raise ValueError(f"Expected 512 tokens, got {len(output)}")
        result = dict(
            mode=mode,
            prompt=len(ids),
            tokens=len(output),
            prefill_s=prefill,
            generation_s=elapsed,
            generated_tps=len(output) / elapsed,
            prompt_tps=len(ids) / prefill,
            sha256=hashlib.sha256(struct.pack("<512I", *output)).hexdigest(),
            work={k: v - before[k] for k, v in backend.work_counters().items()},
        )
        Path(f"frontier64-{args.case}.json").write_text(json.dumps(result) + "\n")
finally:
    backend.close()
