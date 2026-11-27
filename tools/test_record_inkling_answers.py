"""Exercise process interruption at each publication boundary of an answer run."""

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import record_inkling_answers as recorder


def batch(answer):
    return {
        "metadata": dict(model="fixture", engine="fixture", adapter=None,
                         settings={}, scoring="exact"),
        "context": "The answer is blue.",
        "cases": [dict(id="color", question="Which color?", expected="blue",
                       answer=answer, outcome="correct" if answer == "blue" else "incorrect",
                       evidence="The answer is blue.")],
    }


def interrupt_after(root, boundary):
    publish = recorder.publish
    writes = 0

    def interrupted(path, data):
        nonlocal writes
        publish(path, data)
        writes += 1
        if writes == boundary:
            os._exit(81)

    with patch.object(recorder, "publish", interrupted):
        recorder.record(root, "trial", batch("blue"))


class InterruptedRecordTest(unittest.TestCase):
    def test_resume_each_boundary(self):
        for boundary in range(1, 7):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                child = multiprocessing.get_context("spawn").Process(
                    target=interrupt_after, args=(root, boundary))
                child.start()
                child.join(15)
                if child.is_alive():
                    child.kill()
                    child.join()
                    self.fail("Recorder did not reach the selected publication boundary")
                self.assertEqual(child.exitcode, 81)
                manifest_path = root / "evaluation/runs/trial/manifest.json"
                self.assertEqual(manifest_path.exists(), boundary == 6)
                with self.assertRaises(ValueError):
                    recorder.record(root, "trial", batch("red"))
                manifest = recorder.record(root, "trial", batch("blue"))
                self.assertEqual(json.loads(manifest_path.read_bytes()), manifest)
                self.assertEqual(manifest["counts"], dict(correct=1, incorrect=0, ungraded=0))
                for outcome, path in manifest["files"].items():
                    cases = list(map(json.loads, (root / path).read_bytes().splitlines()))
                    expected = batch("blue")["cases"] if outcome == "correct" else []
                    self.assertEqual(cases, expected)


if __name__ == "__main__":
    unittest.main()
