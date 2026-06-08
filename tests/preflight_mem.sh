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
# Fail-closed policy: an unreachable node or unreadable wired memory always
# flags. When wired is OVER the threshold, anything that prevents confirming
# the node is actively serving (diagnostics unreachable / not a NodeDiagnostics
# payload) also flags — a high-wired node we can't clear must not be
# green-lit. Wired BELOW the threshold is safe regardless of runner state.
#
# Usage: tests/preflight_mem.sh [node1 node2 ...]   (default: kite1 kite2 kite3)
# Node args are ssh targets: bare aliases (kite1) resolve per-node users via
# ~/.ssh/config; pass user@host explicitly if you have no alias. Runs on the
# workstation and SSHes each node (the kites can't ssh each other).
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
  d=json.load(sys.stdin)
  # Must be a real NodeDiagnostics payload; an error JSON (e.g. FastAPI
  # {\"detail\":...} on 404) lacks supervisorRunners and is NOT a verified 0.
  if not isinstance(d, dict) or \"supervisorRunners\" not in d: print(\"NA\")
  else: print(sum(1 for r in d[\"supervisorRunners\"] if r.get(\"processAlive\")))
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
  echo "PREFLIGHT FAIL: reboot each flagged node, then re-run. From this"
  echo "workstation: ssh <node> 'osascript -e \"tell app \\\"System Events\\\" to restart\"'"
  echo "(run remotely per node — NOT locally, which would reboot this workstation)."
  exit 1
fi
echo "PREFLIGHT OK: no leaked memory detected."
