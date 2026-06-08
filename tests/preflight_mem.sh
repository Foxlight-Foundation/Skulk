#!/bin/zsh
# preflight_mem.sh — fail fast if any fleet node has leaked (unreleased) wired
# memory before a test run. Abnormal Metal terminations (warmup-wedge SIGKILL,
# GPU-timeout abort) leak wired memory that only a reboot reclaims; a poisoned
# node silently causes false 400s and GPU-timeouts that get mis-blamed on
# code bugs (see gpt-oss / Skulk#236). Run this BEFORE any benchmark/torture.
#
# Signal: a node with ZERO active runner processes should sit near its idle
# baseline (~2GB wired). Wired above the threshold with no runners == leak.
# Usage: preflight_mem.sh [node1 node2 ...]   (default: kite1 kite2 kite3)
set -u
NODES=("$@"); (( ${#NODES} )) || NODES=(kite1 kite2 kite3)
THRESHOLD_GB=5.0   # idle baseline ~2GB; poisoned observed 13.2GB
poisoned=0

for n in $NODES; do
  read -r wired runners <<<"$(ssh -o ConnectTimeout=8 "$n" '
    w=$(vm_stat | awk "/wired/{gsub(/\./,\"\",\$4); print \$4}")
    r=$(pgrep -fc "entrypoint|[b]in/skulk.*runner" 2>/dev/null || echo 0)
    echo "$(echo "$w*16384/2^30" | bc -l) $r" ' 2>/dev/null)"
  if [ -z "$wired" ]; then echo "WARN  $n: unreachable"; continue; fi
  wired_fmt=$(printf "%.1f" "$wired")
  over=$(echo "$wired > $THRESHOLD_GB" | bc -l)
  if [ "$runners" = "0" ] && [ "$over" = "1" ]; then
    echo "POISONED $n: wired=${wired_fmt}GB with 0 runner procs (> ${THRESHOLD_GB}GB) — leaked memory; REBOOT before testing"
    poisoned=1
  else
    echo "OK       $n: wired=${wired_fmt}GB runner_procs=$runners"
  fi
done

if [ "$poisoned" = 1 ]; then
  echo "PREFLIGHT FAIL: reboot poisoned node(s) (osascript -e 'tell app \"System Events\" to restart') before running tests."
  exit 1
fi
echo "PREFLIGHT OK: no leaked memory detected."
