#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
unit=bw-report-${work##*/}.service
cleanup() {
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    rm -rf -- "$work"
}
trap cleanup EXIT
cp -r "$here/source" "$work/project"
chmod 0664 "$work/project/"*
systemd-run --user --unit="$unit" /usr/bin/python3 "$work/project/worker.py"
for ((i=0; i<50; i++)); do
    systemctl --user is-failed --quiet "$unit" && break
    sleep 0.1
done
systemctl --user is-failed --quiet "$unit"
journalctl --user -u "$unit" -o cat --no-pager | grep -F 'ValueError: unsupported mode'
branchwrite apply "$here/fix.bw" --source "$work/project" --destination "$work/repaired" \
    --sha256 4937e463599a5f94cd430131ad07032bfb10442d6a5c21c49ddf376a0503a4c4
install -m 0664 "$work/repaired/service.json" "$work/project/service.json"
systemctl --user restart "$unit"
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
echo 'Three live health probes passed.'
