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
# pod-create time (same pattern as PUBLIC_KEY). Persist it to the canonical
# Hugging Face token file so skulk's downloader and huggingface_hub authenticate
# every download: without a token, model fetches from gated or Xet-backed repos
# fail and a `--ensure-store-downloads` run stalls waiting on a download that
# never starts. Written to the token file (which skulk's get_hf_token reads via
# HF_HOME) and exported for the interactive bootstrap shell.
if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p /root/.cache/huggingface
  printf '%s' "${HF_TOKEN}" > /root/.cache/huggingface/token
  chmod 600 /root/.cache/huggingface/token
  printf 'export HF_TOKEN=%s\n' "${HF_TOKEN}" > /etc/profile.d/skulk-hf-token.sh
  chmod 600 /etc/profile.d/skulk-hf-token.sh
fi

# Host keys are generated per pod: the public image ships without any (they
# are deleted at build time) because baked-in keys would be shared across
# every pod and readable by anyone who pulls the image.
ssh-keygen -A

echo "[pod-start] starting sshd; bootstrap a session with /opt/skulk/pod-bootstrap.sh"
# Foreground as pid 1: container stop signals reach sshd directly, and an
# sshd death ends the pod instead of leaving it alive but unreachable.
exec /usr/sbin/sshd -D -e
