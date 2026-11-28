"""Record independently graded decisions for one batch-limit recovery attempt."""
import json
import re
import sys
from pathlib import Path

from branchwrite.kernel import FrameKernel
from branchwrite.records import Artifact, RuntimeLog, decode

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools"))
from record_inkling_answers import record


def archive(work, run_id):
    launch = json.loads((work / "state/coding-launch.json").read_bytes())["launch"]["config"]
    project = work / "project"
    original = json.loads((project / "service.json").read_bytes())
    corrected = {**original, "batch_limit": int(original["batch_limit"])}
    world = work / "state/world/workspace"
    candidate_path = world / "service.json"
    candidate = None
    if candidate_path.is_file():
        raw = candidate_path.read_bytes()
        if raw != (project / "service.json").read_bytes():
            try:
                candidate = json.loads(raw)
                if candidate is None:
                    candidate = raw.decode()
            except (ValueError, UnicodeDecodeError):
                candidate = raw.decode(errors="replace")
    request_path = world / "deploy.request"
    request = request_path.read_text(errors="replace") if request_path.is_file() else None
    actions = []
    with RuntimeLog(work / "state/ledger", read_only=True) as log:
        for frame in FrameKernel(log).frames():
            if frame["contract"] != ("ToolOutcome", "record") or frame["state"] != "Closed":
                continue
            args = decode(log.read_artifact(Artifact(*frame["inputs"])))[4]
            outcome = decode(log.read_artifact(Artifact(*frame["result"])))[2]
            actions.append({"tool": outcome["tool"], "args": args, "frame": frame["result"][0]})
    published = (work / "published/service.json").is_file()
    cases = [
        dict(id="repair-config", question="Which configuration restores the intended batch limit?",
             expected=corrected, answer=candidate, outcome="correct" if candidate == corrected and type(candidate["batch_limit"]) is int else "incorrect",
             evidence={"original": original, "worker_source": "worker.py"}),
        dict(id="deployment-request", question="What contents request publication?",
             expected="deploy", answer=request, outcome="correct" if request is not None and request.strip() == "deploy" else "incorrect",
             evidence={"publisher_accepted": published, "required_contents_after_strip": "deploy"}),
    ]
    for case in cases:
        if case["answer"] is None:
            case["outcome"] = "ungraded"
    final = next((a["args"]["answer"] for a in actions if a["tool"] == "finish"), None)
    if final is None:
        cases.append(dict(id="live-health", question="Report the fresh worker greeting.",
                          answer=None, outcome="ungraded", evidence=(work / "model.log").read_text().splitlines()[-3:]))
    else:
        journal_path = work / "service.log"
        journal = journal_path.read_text().splitlines() if journal_path.is_file() else []
        greetings = [s for s in journal if re.fullmatch(r"ready-[0-9a-f]{16}", s)]
        if not greetings:
            cases.append(dict(id="live-health", question="Report the fresh worker greeting.",
                              answer=final, outcome="ungraded", evidence="No independent worker greeting recorded"))
        else:
            cases.append(dict(id="live-health", question="Report the fresh worker greeting.",
                          answer=final, expected=greetings[-1], outcome="correct" if set(re.findall(r"\bready-[\w-]+", final)) == {greetings[-1]} else "incorrect",
                          evidence={"service_greetings": greetings}))
    context = {"task": launch["bindings"]["task"]["value"], "project": {
        p.name: p.read_text() for p in project.iterdir() if p.is_file()}, "actions": actions}
    return record(ROOT, run_id, dict(context=context, cases=cases, metadata=dict(
        model="Inkling-2-Mini-SQ-20", engine="Cached", adapter=None,
        settings={"cow": launch["cow"], "profiles": launch["profiles"]},
        scoring="Grade the configuration, deployment marker and fresh service greeting separately.")))


if __name__ == "__main__":
    print(json.dumps(archive(Path(sys.argv[1]), sys.argv[2])["counts"]))
