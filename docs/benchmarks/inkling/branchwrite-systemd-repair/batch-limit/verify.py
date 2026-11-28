"""Verify the completed run against independent service observations."""
import json
import re
import sys
from pathlib import Path

from branchwrite.kernel import FrameKernel
from branchwrite.records import Artifact, RuntimeLog, decode

work = Path(sys.argv[1])
result = json.loads((work / "model.log").read_text().splitlines()[-1])
if result.get("verified") is not True or result["status"] != "complete":
    raise ValueError("BranchWrite did not verify the repair")
journal = (work / "service.log").read_text().splitlines()
expected = json.loads((work / "project/service.json").read_bytes())
expected["batch_limit"] = int(expected["batch_limit"])
if json.loads((work / "published/service.json").read_bytes()) != expected:
    raise ValueError("Published configuration differs from the requested repair")
with RuntimeLog(work / "state/ledger", read_only=True) as log:
    observations = []
    for frame in FrameKernel(log).frames():
        if frame["contract"] != ("ToolOutcome", "record") or frame["state"] != "Closed":
            continue
        args = decode(log.read_artifact(Artifact(*frame["inputs"])))[4]
        outcome = decode(log.read_artifact(Artifact(*frame["result"])))[2]
        observations.append((outcome["tool"], args, outcome))
    answers = [args["answer"] for tool, args, _ in observations if tool == "finish"]
    greetings = [log.read_artifact(Artifact(*out["stdout"])).decode().strip()
                 for tool, args, out in observations if tool == "run_shell"
                 and args.get("command") == "cat /run/branchwrite/health" and out["exit_code"] == 0]
    if not any(re.fullmatch(r"ready-[0-9a-f]{16}", greeting) and greeting in journal
               and greeting in answer for greeting in greetings for answer in answers):
        raise ValueError("Final answer lacks an observed greeting from the repaired worker")
print(json.dumps({"result_artifact": result["result_artifact"], "greetings": greetings, "verified": True}))
