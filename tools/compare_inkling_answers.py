"""Compare two archived Inkling answer runs on the same scored questions."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_run(root, run):
    directory = root / "evaluation/runs" / run
    manifest = json.loads((directory / "manifest.json").read_bytes())
    cases = {}
    for outcome in ("correct", "incorrect", "ungraded"):
        path = root / manifest["files"][outcome]
        records = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(records) != manifest["counts"][outcome]:
            raise ValueError(f"{run}: count mismatch for {outcome}")
        for case in records:
            if case["id"] in cases or case["outcome"] != outcome:
                raise ValueError(f"{run}: invalid outcome or duplicate case {case['id']}")
            cases[case["id"]] = case
    return manifest, json.loads((directory / "context.json").read_bytes()), cases


def compare(root, before, after):
    old, old_context, left = read_run(root, before)
    new, new_context, right = read_run(root, after)
    if old_context != new_context or old["metadata"]["scoring"] != new["metadata"]["scoring"]:
        raise ValueError("Context or scoring rule differs between runs")
    if left.keys() != right.keys():
        raise ValueError("Case IDs differ between runs")
    transitions = Counter()
    changed = 0
    for key, a in left.items():
        b = right[key]
        if a["question"] != b["question"] or a.get("expected") != b.get("expected"):
            raise ValueError(f"Question or expected answer differs for {key}")
        transitions[f"{a['outcome']}_to_{b['outcome']}"] += 1
        changed += a["answer"] != b["answer"]
    return dict(before=before, after=after, cases=len(left), before_counts=old["counts"],
                after_counts=new["counts"], transitions=dict(sorted(transitions.items())),
                changed_answers=changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(compare(args.root, args.before, args.after), indent=2))
