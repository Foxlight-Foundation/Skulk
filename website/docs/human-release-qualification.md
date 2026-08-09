---
id: human-release-qualification
title: Human Release Qualification
sidebar_position: 5
---

<!-- Copyright 2025 Foxlight Foundation -->

Human acceptance is the final usability check on a Skulk release candidate. It
starts only after the exact proposed `dev` commit has passed the automated
fresh-install matrix on Apple Silicon, AMD Linux, and a clean ephemeral NVIDIA
environment. It supplements that matrix; it never replaces or repairs a failed
automated qualification.

The purpose is deliberately simple: use the candidate as a new user would and
decide whether the first experience is understandable, truthful, and useful.

## Candidate contract

Record the full 40-character candidate commit before testing. Every machine in
the human test must begin from an empty Skulk home and use the public installer
with that exact commit:

```bash
CANDIDATE_COMMIT=<40-character-dev-commit>
curl -fsSL \
  "https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/${CANDIDATE_COMMIT}/install.sh" \
  | bash -s -- --ref "${CANDIDATE_COMMIT}"
```

Both the installer script and the checkout therefore come from the same exact
candidate. Fetching the script from `main` would leave candidate installer
changes untested; the literal `main/install.sh | bash` command belongs to the
harness shipping profile (`skulk-harness fresh-install qualify --profile
shipping`), an optional post-promotion sanity check that the public installer
resolves the promoted commit. It is not a release gate: every release gate
runs before promotion.

Do not add `SKULK_*` overrides, edit the generated `skulk.yaml`, substitute an
engine, reuse an existing Skulk home, or use a private setup wrapper. Those
changes may help diagnose a problem, but they no longer describe the product
we are considering for release.

Before opening the dashboard, retain the installer output and run:

```bash
cd ~/skulk
uv run skulk doctor
uv run skulk
```

Use the launch command printed by the installer if it differs. Confirm the
dashboard URL is printed, the page loads, and the topology represents the
machines that were freshly installed.

## Human journeys

Exercise these in order. Judge both whether the action works and whether the
interface tells the truth while it is working.

1. **Install and first launch.** Read the installer and doctor output as a new
   user would. Warnings must explain their consequence and next action. No
   undocumented intervention should be necessary.
2. **Topology and Settings.** Open Settings, review the generated defaults,
   save without changing them, return to topology, and refresh the page. The
   dashboard must stay usable, show the expected nodes and health state, and
   produce no uncaught browser-console exception.
3. **Find, download, and launch.** Discover a model through the Model Store,
   download it, launch it through the dashboard, and wait for a clear ready
   state. Progress, placement failure, and recovery messages must be actionable.
   Begin without a preseeded Hugging Face token. If the provider requires one,
   the dashboard must explain why and accept it through Settings rather than
   relying on a hidden environment credential.
4. **Text chat.** Select the launched model, send several ordinary prompts,
   and confirm streaming, completion, reasoning display, and cancellation feel
   coherent. Retry one deliberately interrupted request without reloading the
   dashboard.
5. **Vision on Apple Silicon.** For every dashboard model presented as vision
   capable, attach a newly chosen image whose contents are not stated in the
   prompt. Confirm the thumbnail appears before sending, remains on the sent
   message, and the answer identifies details that can only come from the
   image. A text-only model must not offer a vision action it cannot execute.
6. **Speech.** When qualified TTS and STT cards are available, use **Speak
   draft** and listen to the result, then use the microphone path and verify the
   recognized words appear in the composer. Check permissions and errors are
   explained rather than leaving the UI stuck.
7. **Conversation persistence.** Reload the page and reopen the conversation.
   Text turns and image attachments must remain associated with the correct
   messages.
8. **Restart and recovery.** Stop Skulk normally, start it with the same printed
   command, and confirm the dashboard, topology, model state, and a new text
   request recover without configuration edits.

## Platform focus

| Platform | Human emphasis |
| --- | --- |
| Apple Silicon | Installer, MLX detection, complete dashboard journey, text, vision, and configured speech journeys |
| AMD Linux | Installer, GPU-backed llama.cpp detection, placement, concurrent text serving, and honest text-only UI |
| NVIDIA Linux | Installer, CUDA detection, placement, and text serving; the automated gate must still use a clean ephemeral RunPod target and prove deletion |

Human testing on available NVIDIA hardware is useful, but it does not remove
the mandatory clean RunPod leg from automated candidate or shipping
qualification.

## Reporting a failure

Stop at the first release-blocking failure and retain enough evidence for a
short reproduction:

- the exact candidate commit and public hardware/platform class;
- the shortest steps that reproduce the behavior and the expected result;
- the model id and the stage that failed: install, download, placement, load,
  dashboard, request, or restart;
- a screenshot or short recording, browser-console output when relevant,
  `skulk doctor` output, and a bounded section of the Skulk log.

Remove secrets, tokens, private hostnames, addresses, node identifiers, and
unrelated local paths before posting evidence publicly.

If the fix changes Skulk code, the installer, shipped defaults, dashboard, or a
model card, the release owner judges whether the change is material to the
first-install experience. A material change creates a new candidate: merge the
fix to `dev`, record the new full commit, repeat automated fresh-install
qualification, and only then repeat the affected human journey. A change judged
immaterial (or a documentation-only clarification that does not change the
commands or runtime contract) does not invalidate the candidate; record the
judgment and the commits it covers in the sign-off.

## Sign-off

Record the candidate commit, automated qualification report, tester, date,
platforms exercised, journeys completed, any post-qualification materiality
judgments, and links to any issues. Qualification is complete before
promotion: human acceptance on top of the passed automated matrix is the final
release gate, and the `dev` to `main` promotion publishes the already-qualified
release. The harness shipping profile may be run after promotion as a
non-gating sanity check that the literal public `main` installer resolves the
promoted commit.
