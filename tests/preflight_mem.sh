#!/usr/bin/env bash
# preflight_mem.sh — fail fast if any fleet node has leaked (unreleased) wired
# memory before a test run. Abnormal Metal terminations (warmup-wedge SIGKILL,
# GPU-timeout abort) leak wired memory that only a reboot reclaims; a poisoned
# node silently causes false placement 400s and decode GPU-timeouts that get
# mis-blamed on code bugs — it cost a multi-hour gpt-oss misdiagnosis where a
# memory-starved node's GPU timeout read as a "distributed ring bug" that did
# not exist (Skulk#236).
#
# Signal: a node with NO live inference runners should sit near its idle
# baseline (~2GB wired on the M4 kites). Wired above the threshold with zero
# live runners == leaked memory; reboot to reclaim.
#
# The gate FAILS CLOSED: an unreachable node, an unreadable field, or a
# parse failure flags the node rather than passing it — a gate that cannot
# verify a node must not green-light it.
#
# Usage: tests/preflight_mem.sh [node1 node2 ...]   (default: kite1 kite2 kite3)
# Workstation-side: it SSHes each node (the kites can't ssh each other).
set -uo pipefail
NODES=("$@"); (( ${#NODES[@]} )) || NODES=(kite1 kite2 kite3)
THRESHOLD_GB=5                     # idle baseline ~2GB; poisoned observed 13.2GB
THRESHOLD_BYTES=$(( THRESHOLD_GB * 1024 * 1024 * 1024 ))
API_PORT=52415
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
flagged=0
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

for n in "${NODES[@]}"; do
  # One round-trip: page size (don't assume 16K), wired pages, and the local
  # node's LIVE-runner count from its own API (authoritative — avoids pgrep -f
  # matching this probe's shell, and skips retained dead supervisors).
  out=$(ssh "${SSH_OPTS[@]}" "$n" '
    ps=$(sysctl -n hw.pagesize)
    wp=$(vm_stat | awk "/wired/{gsub(/[^0-9]/,\"\",\$NF); print \$NF}")
    runners=$(curl -s --max-time 5 http://localhost:'"$API_PORT"'/v1/diagnostics/node \
      | python3 -c "import json,sys
try:
  rs=json.load(sys.stdin).get(\"supervisorRunners\",[])
  print(sum(1 for r in rs if r.get(\"processAlive\")))
except Exception: print(0)" 2>/dev/null || echo 0)
    echo "$ps $wp $runners"
  ' 2>/dev/null) || { echo "FAIL     $n: unreachable — cannot verify, failing preflight"; flagged=1; continue; }

  read -r pagesize wpages runners <<<"$out"
  if ! is_uint "${pagesize:-}" || ! is_uint "${wpages:-}" || ! is_uint "${runners:-}"; then
    echo "FAIL     $n: unreadable memory/runner state ('$out') — failing preflight"; flagged=1; continue
  fi
  wired_bytes=$(( wpages * pagesize ))
  wired_gb=$(awk "BEGIN{printf \"%.1f\", $wired_bytes/2^30}")
  if [ "$runners" = "0" ] && (( wired_bytes > THRESHOLD_BYTES )); then
    echo "POISONED $n: wired=${wired_gb}GB, 0 live runners (> ${THRESHOLD_GB}GB) — leaked; REBOOT before testing"
    flagged=1
  else
    echo "OK       $n: wired=${wired_gb}GB live_runners=$runners"
  fi
done

if [ "$flagged" = 1 ]; then
  echo "PREFLIGHT FAIL: reboot flagged node(s) (osascript -e 'tell app \"System Events\" to restart') and re-run."
  exit 1
fi
echo "PREFLIGHT OK: no leaked memory detected."
