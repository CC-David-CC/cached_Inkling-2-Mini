#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
live=false
if [[ $# == 3 && $1 == --live ]]; then
    live=true
    connection=$(realpath "$2")
    mkdir -m 700 -- "$3"
    work=$(realpath "$3")
elif [[ $# == 0 ]]; then
    work=$(mktemp -d)
else
    echo "Usage: $0 [--live CONNECTION NEW_EVIDENCE_DIRECTORY]" >&2
    exit 2
fi
unit=bw-report-${work##*/}.service
cleanup() {
    systemctl --user stop "${unit%.service}.path" 2>/dev/null || true
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    if ! "$live"; then rm -rf -- "$work"; fi
}
trap cleanup EXIT
cp -r "$here/source" "$work/project"
if "$live"; then rm "$work/project/incident.log"; fi
chmod 0664 "$work/project/"*
# Publishing the complete repaired directory activates the same service definition.
systemd-run --user --unit="${unit%.service}" --path-property="PathExists=$work/repaired" \
    /bin/sh -c 'cd "$1/repaired" 2>/dev/null || cd "$1/project" || exit 1; exec /usr/bin/python3 worker.py' sh "$work"
systemctl --user start "$unit"
for ((i=0; i<50; i++)); do
    systemctl --user is-failed --quiet "$unit" && break
    sleep 0.1
done
systemctl --user is-failed --quiet "$unit"
journalctl --user -u "$unit" -o cat --no-pager | grep -F 'ValueError: unsupported mode'
archive="$here/fix.bw"
digest=4937e463599a5f94cd430131ad07032bfb10442d6a5c21c49ddf376a0503a4c4
if "$live"; then
    export BW_USER_JOURNAL_UNIT="$unit"
    branchwrite code "$work/project" --connection "$connection" --state "$work/state" \
        --thinking off --instructions compact-finish-request --max-steps 256 \
        --read worker.py --edit service.json --test-command 'python3 worker.py --check' \
        --task 'Inspect the failed report worker using cat /run/branchwrite/unit-journal. Fix its config to serve requests, preserve its name, run python3 worker.py --check and finish.' \
        | tee "$work/model.log"
    python3 - "$work/model.log" <<'PYRESULT'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
if result.get("status") != "complete" or result.get("verified") is not True:
    raise SystemExit("Model repair incomplete")
PYRESULT
    python3 "$here/verify_diagnosis.py" "$work" "$unit"
    archive="$work/fix.bw"
    branchwrite export "$work/state" "$archive" > "$work/export.json"
    digest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$work/export.json")
fi
apply=(branchwrite apply "$archive" --source "$work/project" --destination "$work/repaired")
if "${apply[@]}" --sha256 "$(printf '%064d' 0)" >"$work/rejection" 2>&1; then
    exit 1
fi
grep -F 'archive hash mismatch' "$work/rejection"
[[ ! -e "$work/repaired" ]]
systemctl --user is-failed --quiet "$unit"
"${apply[@]}" --sha256 "$digest"
# The path unit now starts the repaired generation without a second deploy command.
cmp "$here/source/service.json" "$work/project/service.json"
for ((i=0; i<50; i++)); do
    port=$(journalctl --user -u "$unit" -o cat --no-pager | awk '/^health_port [0-9]+$/ {print $2}')
    [[ -n "$port" ]] && break
    sleep 0.1
done
python3 - "$port" <<'PY'
import socket
import sys
for _ in range(3):
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2) as client:
        with client.makefile("rb") as response:
            if response.readline() != b"serve\n":
                raise SystemExit("Unexpected health reply")
PY
systemctl --user is-active "$unit"
first=$(systemctl --user show "$unit" --property=InvocationID --value)
[[ -n "$first" ]]
"${apply[@]}" --sha256 "$digest"
[[ "$(systemctl --user show "$unit" --property=InvocationID --value)" == "$first" ]]
systemctl --user is-active --quiet "$unit"
echo 'Three live health probes passed. Duplicate apply preserved the running invocation.'
