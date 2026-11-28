import argparse
import hashlib
import heapq
import json
import math
import struct
from pathlib import Path
from tokenizers import Tokenizer
from context_time_travel.backend_registry import load_backend
from context_time_travel.inkling_backend import runtime_profile

parser = argparse.ArgumentParser(description="Capture coding frontiers every 16 generated tokens.")
parser.add_argument("runtime", type=Path)
parser.add_argument("case", choices=("off1", "off2", "dspark1", "dspark2"))
parser.add_argument("--tokens", type=int, choices=(128, 768), default=128)
args = parser.parse_args()
runtime = args.runtime.resolve()
case = args.case
prompt = """# Python 3. Implement a stable topological sort. Nodes whose dependencies are satisfied should appear in input order. Raise ValueError on a cycle.
def stable_toposort(dependencies):
"""
tok = Tokenizer.from_file(str(runtime / "tokenizer-source.json"))
ids = tok.encode(prompt, add_special_tokens=False).ids
backend = load_backend(
    runtime_profile(
        runtime, capacity=8192, pack="SQ-10", draft=case.startswith("dspark")
    )
)
output = []
try:
    with backend.lease() as lease:
        backend.consume(lease, ids)
        for _ in range(args.tokens // 16):
            with backend.generate_request(lease, 16) as stream:
                output.extend(stream)
            scores = backend.logits(lease)
            first, second = heapq.nlargest(
                2, range(len(scores)), key=scores.__getitem__
            )
            print(
                json.dumps(
                    dict(
                        case=case,
                        output_length=len(output),
                        output_sha256=hashlib.sha256(
                            struct.pack(f"<{len(output)}I", *output)
                        ).hexdigest(),
                        top_id=first,
                        second_id=second,
                        margin=scores[first] - scores[second],
                        top_probability=1
                        / math.fsum(math.exp(s - scores[first]) for s in scores),
                    )
                ),
                flush=True,
            )
        Path(case + ".txt").write_text(tok.decode(output))
finally:
    backend.close()
