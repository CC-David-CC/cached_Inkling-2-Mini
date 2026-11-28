#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
unit=bw-report-${work##*/}.service
cleanup() {
    systemctl --user stop "${unit%.service}.path" 2>/dev/null || true
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    rm -rf -- "$work"
}
trap cleanup EXIT
cp -r "$here/source" "$work/project"
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
apply=(branchwrite apply "$here/fix.bw" --source "$work/project" --destination "$work/repaired")
if "${apply[@]}" --sha256 "$(printf '%064d' 0)" >"$work/rejection" 2>&1; then
    exit 1
fi
grep -F 'archive hash mismatch' "$work/rejection"
[[ ! -e "$work/repaired" ]]
systemctl --user is-failed --quiet "$unit"
digest=4937e463599a5f94cd430131ad07032bfb10442d6a5c21c49ddf376a0503a4c4
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
