import os,stat,subprocess,tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'report sample'
 p.write_text('unchanged')
 subprocess.run(['sh','secure_report.sh',str(p)],check=True)
 assert stat.S_IMODE(p.stat().st_mode)==0o640
 assert p.read_text()=='unchanged'
