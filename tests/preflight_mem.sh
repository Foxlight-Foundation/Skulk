#!/bin/zsh
# preflight_mem.sh — fail fast if any fleet node has leaked (unreleased) wired
# memory before a test run. Abnormal Metal terminations (warmup-wedge SIGKILL,
# GPU-timeout abort) leak wired memory that only a reboot reclaims; a poisoned
# node silently causes false placement 400s and decode GPU-timeouts that get
# mis-blamed on code bugs — it cost a multi-hour gpt-oss misdiagnosis where a
# memory-starved node's GPU timeout read as a "distributed ring bug" that did
# not exist (Skulk#236).
#
# Signal: a node with NO active inference runners should sit near its idle
# baseline (~2GB wired on the M4 kites). Wired above the threshold with zero
# active runners == leaked memory; reboot to reclaim.
#
# Usage: tests/preflight_mem.sh [node1 node2 ...]   (default: kite1 kite2 kite3)
# Workstation-side: it SSHes each node (the kites can't ssh each other).
set -u
NODES=("$@"); (( ${#NODES} )) || NODES=(kite1 kite2 kite3)
THRESHOLD_GB=5.0   # idle baseline ~2GB; poisoned observed 13.2GB
API_PORT=52415
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
flagged=0

for n in "${NODES[@]}"; do
  # One round-trip: page size (don't assume 16K), wired pages, and the local
  # node's active-runner count from its own API (authoritative — avoids
  # pgrep -f matching this very probe's shell). API down => 0 runners, the
  # correct assumption for a leak check.
  out=$(ssh "${SSH_OPTS[@]}" "$n" '
    ps=$(sysctl -n hw.pagesize)
    wp=$(vm_stat | awk "/wired/{gsub(/[^0-9]/,\"\",\$NF); print \$NF}")
    runners=$(curl -s --max-time 5 http://localhost:'"$API_PORT"'/v1/diagnostics/node \
      | python3 -c "import json,sys
try: print(len(json.load(sys.stdin).get(\"supervisorRunners\",[])))
except Exception: print(0)" 2>/dev/null || echo 0)
    echo "$ps $wp $runners"
  ' 2>/dev/null) || { echo "FAIL $n: unreachable — cannot verify, failing preflight"; flagged=1; continue; }

  pagesize=${out%% *}; rest=${out#* }; wpages=${rest%% *}; runners=${rest##* }
  if [ -z "$pagesize" ] || [ -z "$wpages" ]; then
    echo "FAIL $n: could not read memory state — failing preflight"; flagged=1; continue
  fi
  wired_gb=$(echo "scale=1; $wpages * $pagesize / 2^30" | bc -l)
  over=$(echo "$wired_gb > $THRESHOLD_GB" | bc -l)
  if [ "$runners" = "0" ] && [ "$over" = "1" ]; then
    echo "POISONED $n: wired=${wired_gb}GB, 0 active runners (> ${THRESHOLD_GB}GB) — leaked; REBOOT before testing"
    flagged=1
  else
    echo "OK       $n: wired=${wired_gb}GB active_runners=$runners"
  fi
done

if [ "$flagged" = 1 ]; then
  echo "PREFLIGHT FAIL: reboot flagged node(s) (osascript -e 'tell app \"System Events\" to restart') and re-run."
  exit 1
fi
echo "PREFLIGHT OK: no leaked memory detected."
