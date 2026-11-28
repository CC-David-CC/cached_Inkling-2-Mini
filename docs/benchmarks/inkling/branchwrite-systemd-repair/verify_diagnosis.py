"""Check the live diagnostic action against its recorded stdout and the original config."""
import json
import re
from pathlib import Path
import sys

from branchwrite.kernel import FrameKernel
from branchwrite.records import Artifact, RuntimeLog, decode


def reads_journal(command):
    # Match simple reads of the mounted file. Shell expressions can supply unrelated output.
    return isinstance(command, str) and re.fullmatch(
        r"(?:cat|(?:head|tail)(?:[ \t]+-n[ \t]+[1-9][0-9]*)?)"
        r"[ \t]+(?:--[ \t]+)?/run/branchwrite/unit-journal", command.strip()
    ) is not None


def verify(work, unit):
    with RuntimeLog(work / "state/ledger", read_only=True) as log:
        for frame in FrameKernel(log).frames():
            if frame["contract"] != ("ToolOutcome", "record") or frame["state"] != "Closed":
                continue
            result = decode(log.read_artifact(Artifact(*frame["result"])))[2]
            if result["tool"] != "run_shell" or result["exit_code"] != 0:
                continue
            plan = decode(log.read_artifact(Artifact(*frame["inputs"])))
            if not reads_journal(plan[4].get("command")):
                continue
            output = log.read_artifact(Artifact(*result["stdout"]))
            if unit.encode() in output and b"ValueError: unsupported mode" in output:
                break
        else:
            raise ValueError("Missing successful journal read for the failed unit")
    expected = json.loads((work / "project/service.json").read_bytes())
    expected["mode"] = "serve"
    actual = json.loads((work / "state/world/workspace/service.json").read_bytes())
    if actual != expected:
        raise ValueError("Candidate configuration differs from the requested repair")


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2])
