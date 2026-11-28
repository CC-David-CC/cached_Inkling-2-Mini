"""Replay the published BranchWrite permission repair in a temporary workspace."""
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
receipt = json.loads(here.with_suffix('.json').read_text())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / 'source'
    shutil.copytree(here / 'source', source)
    # Git tracks only the executable bit. Recreate the run's recorded source modes.
    for file in source.iterdir():
        file.chmod(0o664)
    before = subprocess.run([sys.executable, 'check.py'], cwd=source, capture_output=True)
    if before.returncode == 0 or b'AssertionError' not in before.stderr:
        raise RuntimeError('Source did not reproduce the expected failing test')
    fixed = root / 'fixed'
    subprocess.run(['branchwrite', 'apply', str(here / 'fix.bw'), '--source', str(source),
                    '--destination', str(fixed), '--sha256', receipt['archive_sha256']], check=True)
    subprocess.run([sys.executable, 'check.py'], cwd=fixed, check=True)
    payload = bytes(range(256))
    for case in receipt['external_checks']:
        file = root / case['path']
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(payload)
        file.chmod(int(case['initial_mode'], 8))
        subprocess.run(['sh', str(fixed / 'secure_report.sh'), case['path']], cwd=root, check=True)
        if stat.S_IMODE(file.stat().st_mode) != int(case['final_mode'], 8) or file.read_bytes() != payload:
            raise RuntimeError(f"Filesystem check failed for {case['path']}")
    print('Source failed, applied patch passed, all three independent filesystem checks passed.')
