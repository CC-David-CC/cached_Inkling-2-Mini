import csv, itertools, json, math, struct, sys
from pathlib import Path

cases = ["off1", "off2", "dspark1", "dspark2"]
data = {
    case: json.loads(Path(f"frontier64-{case}.json").read_bytes()) for case in cases
}
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(
    [
        "stage",
        "left",
        "right",
        "logits",
        "unequal_float_bits",
        "max_abs_diff",
        "same_argmax",
        "same_generated_tokens",
    ]
)
for stage in ("prefill", "final"):
    rows = {case: Path(f"frontier64-{case}-{stage}.f32").read_bytes() for case in cases}
    for left, right in itertools.combinations(cases, 2):
        a, b = rows[left], rows[right]
        av, bv = [struct.unpack("<200058f", v) for v in [a, b]]
        if not all(map(math.isfinite, av)) or not all(map(math.isfinite, bv)):
            raise ValueError(
                f"nonfinite score in {stage} comparison of {left} and {right}"
            )
        writer.writerow(
            [
                stage,
                left,
                right,
                len(av),
                sum(
                    x != y
                    for x, y in zip(memoryview(a).cast("I"), memoryview(b).cast("I"))
                ),
                max(abs(x - y) for x, y in zip(av, bv)),
                av.index(max(av)) == bv.index(max(bv)),
                data[left]["sha256"] == data[right]["sha256"],
            ]
        )
