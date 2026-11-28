import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("candidate", Path(sys.argv[1]) / "batches.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
passed = total = 0
for length in range(21):
    for size in range(1, 9):
        items = list(range(length))
        actual = candidate.batches(items.copy(), size)
        expected = []
        for index, item in enumerate(items):
            if index % size == 0:
                expected.append([])
            expected[-1].append(item)
        passed += actual == expected
        total += 1
for size in [0, -1, -5]:
    try:
        candidate.batches([1], size)
    except ValueError:
        passed += 1
    total += 1
print(f"{passed}/{total}")
sys.exit(passed != total)
