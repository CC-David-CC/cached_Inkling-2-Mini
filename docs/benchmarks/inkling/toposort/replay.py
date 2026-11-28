"""Execute the saved model functions against the fixture in fresh Python processes."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


def grade():
    from dspark1 import stable_toposort as drafted
    from off1 import stable_toposort as undrafted

    from repair_effort01 import stable_toposort as repaired

    results = []
    for fixture in json.loads((DIRECTORY / "fixtures.json").read_text()):
        for arm, function in (("off1", undrafted), ("dspark1", drafted), ("repair_effort01", repaired)):
            try:
                answer = function(fixture["input"])
            except Exception as exc:
                answer = type(exc).__name__
            expected = fixture.get("expected", fixture.get("error"))
            results.append([arm, fixture["id"], json.dumps(answer), answer == expected])
    return results


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(grade()))
    else:
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(["seed", "arm", "fixture", "answer", "passed"])
        incomplete = DIRECTORY / "dspark2-incomplete.txt"
        source = "def stable_toposort(dependencies):\n" + incomplete.read_text()
        try:
            compile(source, incomplete.name, "exec")
        except SyntaxError as exc:
            writer.writerow(["", "dspark2", "compile", json.dumps(type(exc).__name__), False])
        else:
            writer.writerow(["", "dspark2", "compile", json.dumps("compiled"), True])
        for seed in range(32):
            output = subprocess.check_output(
                [sys.executable, __file__, "--worker"],
                env=dict(os.environ, PYTHONHASHSEED=str(seed)), text=True
            )
            for result in json.loads(output):
                writer.writerow([seed, *result])
