#!/usr/bin/env bash
# Pod entrypoint for the prebaked CUDA image (RunPod and similar).
#
# RunPod injects the operator's SSH public key as the PUBLIC_KEY environment
# variable when the pod is created with it (API-created pods MUST pass it:
# without a key sshd is unreachable and the pod is only usable through the
# provider's interactive web terminal). This installs the key, starts sshd,
# and idles as pid 1 so the pod stays alive between SSH sessions.
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  printf '%s\n' "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

echo "[pod-start] sshd up; bootstrap a session with /opt/skulk/pod-bootstrap.sh"
exec sleep infinity
