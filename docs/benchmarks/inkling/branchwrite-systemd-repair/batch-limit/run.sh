#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
connection=$(realpath "${1:?Pass a BranchWrite connection profile}")
d=$(mktemp -d "${TMPDIR:-/tmp}/bw-batch-limit-XXXXXXXX")
echo "Evidence directory: $d"
cp -r "$here/source" "$d/project"
cd "$d"
unit=bw-${d##*/}.service
export BW_USER_JOURNAL_UNIT="$unit" BW_HEALTH_UNIX_SOCKET="/tmp/${unit}.sock"
systemd-run --user --unit="${unit%.service}" --path-property="PathExists=$d/published" --setenv="HEALTH_ENDPOINT=$BW_HEALTH_UNIX_SOCKET" /bin/sh -c 'cd "$1/published" 2>/dev/null || cd "$1/project"; exec /usr/bin/python3 worker.py' sh "$d"
trap 'systemctl --user stop "${unit%.service}.path" "$unit" || true; rm -f "$BW_HEALTH_UNIX_SOCKET"; if [[ -n ${watcher:-} ]]; then kill "$watcher" 2>/dev/null || true; fi' EXIT
systemctl --user start "$unit"
for i in {1..40}; do systemctl --user is-failed --quiet "$unit" && break; sleep .1; done
systemctl --user is-failed --quiet "$unit"
python3 - <<'PY' > deploy.log 2>&1 &
import json,time,shutil
from pathlib import Path
world=Path('state/world/workspace')
request=world/'deploy.request'
while True:
    if request.is_file() and request.read_text().strip()=='deploy':
        data=(world/'service.json').read_bytes()
        try:
            config = json.loads(data)
        except ValueError:
            config = None
        if config == {'mode':'serve','name':'report-worker','batch_limit':32} and type(config['batch_limit']) is int:
            shutil.copytree('project','staged')
            Path('staged/service.json').write_bytes(data)
            Path('staged').rename('published')
            print('Published verified configuration on explicit request',flush=True)
            break
    time.sleep(.05)
PY
watcher=$!
model_status=0
branchwrite code project --state state --connection "$connection" --thinking off --instructions compact-finish-request --read OPERATIONS.md --read worker.py --edit service.json --create deploy.request --test-command 'cat /run/branchwrite/health' --task 'Restore the report worker using its operations guide. Return the exact greeting from the recovered instance.' --max-steps "${BW_FIXTURE_MAX_STEPS:-320}" > model.log 2>&1 || model_status=$?
if [[ -n ${BW_FIXTURE_RESUME_STEPS:-} ]] && python3 - model.log <<'PYRESUME'
import json
import sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text().splitlines()
try:
    status = json.loads(lines[-1]).get("status")
except (IndexError, ValueError, AttributeError):
    status = None
sys.exit(status != "step_budget_exhausted")
PYRESUME
then
    cp model.log before-resume.log
    python3 "$here/record_run.py" "$d" "${d##*/}-before-resume"
    if [[ -d "$d/published" ]]; then
        systemctl --user is-active --quiet "$unit"
        systemctl --user show "$unit" --property=InvocationID --value > invocation-before-resume
        stat -c '%d:%i' "$d/published" > publication-before-resume
    fi
    model_status=0
    branchwrite resume state --max-steps "$BW_FIXTURE_RESUME_STEPS" >> model.log 2>&1 || model_status=$?
fi
journalctl --user -u "$unit" -o cat --no-pager > service.log
python3 "$here/record_run.py" "$d" "${d##*/}"
if (( model_status )); then exit "$model_status"; fi
python3 "$here/verify.py" "$d"
if [[ -f invocation-before-resume ]]; then
    systemctl --user is-active --quiet "$unit"
    systemctl --user show "$unit" --property=InvocationID --value > invocation-after-resume
    stat -c '%d:%i' "$d/published" > publication-after-resume
    cmp invocation-before-resume invocation-after-resume
    cmp publication-before-resume publication-after-resume
fi
