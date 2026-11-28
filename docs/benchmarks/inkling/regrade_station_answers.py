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

with (directory / "failure-only-paired-answers.csv").open() as source:
    practice = list(csv.DictReader(source))
if len(practice) != len(expected) or {row["station"] for row in practice} != expected.keys():
    raise ValueError("Failure-only evaluation must cover each original station exactly once")
if any(row["expected"] != expected[row["station"]] for row in practice):
    raise ValueError("Failure-only ground truth differs from the original register")

base_correct = {row["station"] for row in records
                if row["stations-v1-base"].strip().lower().strip(" .") == expected[row["station"]]}
previous = base_correct
print("optimizer_step,correct,original_successes_retained,gains_since_previous,losses_since_previous")
with (directory / "failure-only-rounds.csv").open() as source:
    summaries = list(csv.DictReader(source))
steps = [int(summary["optimizer_step"]) for summary in summaries]
answer_steps = sorted(int(key.removeprefix("step_")) for key in practice[0] if key.startswith("step_"))
if steps != answer_steps:
    raise ValueError("Optimizer steps differ between answer and summary files")
for step, summary in zip(steps, summaries):
    correct_now = {row["station"] for row in practice
                   if row[f"step_{step}"].strip().lower().strip(" .") == expected[row["station"]]}
    gains, losses = len(correct_now - previous), len(previous - correct_now)
    measured = (len(correct_now), len(correct_now - base_correct), gains, losses)
    columns = ("correct_of_54", "original_misses_fixed", "gained", "lost")
    if measured != tuple(int(summary[column]) for column in columns):
        raise ValueError(f"Step {step} summary differs from paired answers")
    print(f"{step},{len(correct_now)},{len(correct_now & base_correct)},{gains},{losses}")
    previous = correct_now
