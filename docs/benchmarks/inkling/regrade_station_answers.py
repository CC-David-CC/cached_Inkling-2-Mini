"""Regrade every published station answer against the original register and base."""
import csv
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
root = directory.parents[2]
fixture = root / "training/wrong_but_corrected/shared-document-128k"
retention = root / "evaluation/retention/shared-document-128k/cases.jsonl"
expected = {}
for path, field in [(fixture / "examples.jsonl", "chosen"), (retention, "expected")]:
    for line in path.read_text().splitlines():
        case = json.loads(line)
        if case["id"] in expected:
            raise ValueError(f"Duplicate fixture ID {case['id']}")
        expected[case["id"]] = case[field]

register = set((fixture / "system-prompt.txt").read_text().splitlines())
with (directory / "station-adapter-paired-answers.csv").open() as source:
    reader = csv.DictReader(source)
    runs = reader.fieldnames[2:]
    records = list(reader)
if len(records) != 54 or len(expected) != 54 or {r["station"] for r in records} != expected.keys():
    raise ValueError("Expected the same 54 unique stations in the fixtures and answers")

for row in records:
    station = row["station"]
    evidence = f"Station {station.removeprefix('station-')} has label {expected[station]}."
    if row["expected"] != expected[station] or evidence not in register:
        raise ValueError(f"Ground truth mismatch for {station}")

print("run,correct,gains_vs_base,losses_vs_base")
for run in runs:
    correct = gains = losses = 0
    for row in records:
        label = expected[row["station"]]
        base = row["stations-v1-base"].strip().lower().strip(" .") == label
        passed = row[run].strip().lower().strip(" .") == label
        correct += passed
        gains += passed and not base
        losses += base and not passed
    print(f"{run},{correct},{gains},{losses}")
