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
  # The remote emits the live-runner count, or the literal "NA" when the
  # diagnostics endpoint can't be fetched/parsed — so we never silently
  # conflate "couldn't check" with "verified zero". (A diagnostics fetch
  # that connects and parses but holds no live runners is a real 0.)
  out=$(ssh "${SSH_OPTS[@]}" "$n" '
    ps=$(sysctl -n hw.pagesize)
    wp=$(vm_stat | awk "/wired/{gsub(/[^0-9]/,\"\",\$NF); print \$NF}")
    diag=$(curl -s --max-time 5 http://localhost:'"$API_PORT"'/v1/diagnostics/node)
    runners=$(printf "%s" "$diag" | python3 -c "import json,sys
try:
  rs=json.load(sys.stdin).get(\"supervisorRunners\",[])
  print(sum(1 for r in rs if r.get(\"processAlive\")))
except Exception: print(\"NA\")" 2>/dev/null || echo NA)
    echo "$ps $wp $runners"
  ' 2>/dev/null) || { echo "FAIL     $n: unreachable — cannot verify, failing preflight"; flagged=1; continue; }

  read -r pagesize wpages runners <<<"$out"
  if ! is_uint "${pagesize:-}" || ! is_uint "${wpages:-}"; then
    echo "FAIL     $n: unreadable memory state ('$out') — failing preflight"; flagged=1; continue
  fi
  wired_bytes=$(( wpages * pagesize ))
  wired_gb=$(awk "BEGIN{printf \"%.1f\", $wired_bytes/2^30}")
  over=$(( wired_bytes > THRESHOLD_BYTES ? 1 : 0 ))

  if ! is_uint "${runners:-}"; then
    # Runner state unknown (diagnostics unreachable/unparseable). Fail closed
    # ONLY when it matters — i.e. wired is already over the threshold and we
    # can't rule out a leak. Low wired is safe regardless of runner state.
    if (( over )); then
      echo "FAIL     $n: wired=${wired_gb}GB (> ${THRESHOLD_GB}GB) but runner state UNVERIFIABLE (diagnostics unreachable) — failing closed; check/reboot"
      flagged=1
    else
      echo "OK       $n: wired=${wired_gb}GB (runner state unverifiable, but wired is low)"
    fi
    continue
  fi

  if [ "$runners" = "0" ] && (( over )); then
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
