#!/usr/bin/env bash
# Pod entrypoint for the prebaked CUDA image (RunPod and similar).
#
# RunPod injects the operator's SSH public key as the PUBLIC_KEY environment
# variable when the pod is created with it (API-created pods MUST pass it:
# without a key sshd is unreachable and the pod is only usable through the
# provider's interactive web terminal). This installs the key and then runs
# sshd in the foreground as pid 1 for the pod's lifetime.
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  # Overwrite, not append: container restarts re-run this entrypoint and
  # appending would duplicate entries. Multiple newline-separated keys are a
  # legitimate provider convention, so the content is written as-is.
  printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# A Hugging Face token can be injected as the HF_TOKEN environment variable at
# pod-create time (same pattern as PUBLIC_KEY). Persist it to the Hugging Face
# token file so skulk's downloader and huggingface_hub authenticate every
# download: without a token, model fetches from gated or Xet-backed repos fail
# and a `--ensure-store-downloads` run stalls waiting on a download that never
# starts. skulk's get_hf_token reads `$HF_HOME/token`, so honor a custom HF_HOME
# (a pod may point it at the mounted model volume). The token is written
# verbatim as data, never sourced as a script.
if [ -n "${HF_TOKEN:-}" ]; then
  hf_home="${HF_HOME:-/root/.cache/huggingface}"
  mkdir -p "${hf_home}"
  printf '%s' "${HF_TOKEN}" > "${hf_home}/token"
  chmod 600 "${hf_home}/token"
fi

# Host keys are generated per pod: the public image ships without any (they
# are deleted at build time) because baked-in keys would be shared across
# every pod and readable by anyone who pulls the image.
ssh-keygen -A

echo "[pod-start] starting sshd; bootstrap a session with /opt/skulk/pod-bootstrap.sh"
# Foreground as pid 1: container stop signals reach sshd directly, and an
# sshd death ends the pod instead of leaving it alive but unreachable.
exec /usr/sbin/sshd -D -e
