---
id: install
title: Install Skulk
sidebar_position: 2
description: Install the signed Skulk desktop app on macOS or the packaged app and runtime on Ubuntu and Debian.
---

<!-- Copyright 2025 Foxlight Foundation -->

The packaged Skulk apps are the recommended way to install and operate Skulk.
They give the runtime a stable application identity, include the matching
dashboard and native components, and let you start, stop, inspect, and open a
node without managing a source checkout or Python environment.

Choose the path that matches the machine:

| Machine | Recommended install |
| --- | --- |
| Apple Silicon Mac running macOS 15 or newer | Signed and notarized Skulk menu-bar app through a DMG or Homebrew |
| Ubuntu or Debian desktop, `amd64` or `arm64` | Skulk app and runtime through the Foxlight APT repository |
| Headless Ubuntu or Debian, `amd64` or `arm64` | Runtime-only APT package |
| Contributor workstation, development branch, or another Linux distribution | [Source installer](#source-and-development-installs) |

## macOS

Download the pinned Skulk 1.5.1 signed and notarized app directly:

**[Download Skulk 1.5.1 for Apple Silicon (.dmg)](https://releases.foxlight.ai/desktop/macos/1.5.1/3/Skulk-1.5.1-3-macOS-arm64.dmg)**

Open the DMG, drag **Skulk** to **Applications**, eject the DMG, and open Skulk
from Applications. macOS verifies the Developer ID signature and stapled
notarization ticket before launch.

Homebrew is an equally supported install and update channel:

```bash
brew install --cask Foxlight-Foundation/skulk/skulk
```

Skulk is a menu-bar app: it does not keep a window or Dock icon open. After
launch, click the Skulk fox in the macOS menu bar. The menu shows the node state
and provides **Start Skulk**, **Stop Skulk**, **Open Dashboard**, and
**Reveal Runtime Log**. The app includes the exact Skulk runtime and dashboard
built for its release, so you do not need to install or approve `uv`, Python,
Node.js, or a source tree.

### First run on macOS

1. Open **Skulk** from Applications. The menu initially reports **Stopped**;
   that is expected because Skulk waits for you to choose to start.
2. Click the Skulk menu-bar fox and choose **Start Skulk**.
3. Approve macOS **Local Network** access. Skulk needs it to discover and
   communicate with nodes on your LAN. Skulk does **not** require Screen &
   System Audio Recording; deny that permission and [report the app
   version](https://github.com/Foxlight-Foundation/Skulk/issues) if macOS ever
   presents it.
4. Wait for the menu to report **Ready**, then choose **Open Dashboard**.
5. Confirm that your Mac appears in the topology. A single Mac is a complete
   one-node cluster, so you can launch a model and chat before adding more
   machines.

### Update or remove the Mac app

Choose **Check for Updates…** from the Skulk menu. When a newer version is
available, **Download _version_…** opens its signed DMG. Install it through the
same Applications-folder flow. Skulk also checks at app startup and never
silently replaces itself.

If you use Homebrew, update with `brew upgrade --cask
Foxlight-Foundation/skulk/skulk`. To remove a DMG installation, quit Skulk and
move **Skulk** from Applications to the Trash. Homebrew users can instead run
`brew uninstall --cask skulk`.

## Ubuntu and Debian

Install the Foxlight repository keyring, refresh APT, and install Skulk:

```bash
curl -fLO https://apt.foxlight.ai/foxlight-archive-keyring.deb
sudo apt install ./foxlight-archive-keyring.deb
sudo apt update
sudo apt install skulk
```

When `sudo` asks for your Linux login password, type it and press Enter. The
terminal deliberately shows no dots or other characters while you type.

The public keyring package contains only the repository's public signing key
and APT source definition. The repository's private signing key is never
distributed.

The `skulk` package installs both the desktop controller and the matching
runtime. Open **Skulk** from the application menu and select **Start Skulk**.
The app enables and starts the packaged `skulk.service` user unit, then gives
you controls for the dashboard, logs, node lifecycle, and cluster namespace.
Installing the package alone does not start the service without your action.

### First run on Linux desktop

1. Open **Skulk** from the application menu.
2. Choose **Start Skulk** and wait for the status to report **Ready**.
3. Choose **Open Dashboard**.
4. Confirm that the local machine appears. One machine is a valid one-node
   cluster; add more machines only after this first node works.

The runtime package contains the exact reviewed Skulk source, locked Python
environment, built dashboard, native bindings, launcher, and user service unit
for that release. Linux inference engines are selected from the hardware Skulk
detects; see the [NVIDIA](nvidia-cuda-nodes) and
[AMD](amd-strix-halo-nodes) guides for hardware-specific preparation.

To update later:

```bash
sudo apt update
sudo apt install skulk
```

To uninstall the app and runtime:

```bash
sudo apt remove skulk skulk-desktop skulk-runtime
```

### Headless Ubuntu or Debian

On a machine with no graphical session, install only the runtime after adding
the repository above:

```bash
sudo apt install skulk-runtime
systemctl --user daemon-reload
systemctl --user enable --now skulk
```

A systemd user service normally runs only while that user has a session. For an
unattended node that must survive logout, an administrator can enable lingering
for the service account:

```bash
sudo loginctl enable-linger "$USER"
```

## Form a cluster

1. Get one node to **Ready** and confirm it in the dashboard.
2. Install the same Skulk version on every additional machine.
3. Start Skulk on each machine from the app or service manager.
4. Open the dashboard from any running node and confirm the machines appear in
   the topology.
5. Pick and launch a model. Skulk places it on compatible hardware and begins
   serving when the placement is ready.

Nodes on the same network use Skulk's shared default namespace and discover one
another automatically. You do not need to invent an identifier for an ordinary
cluster. If multiple independent Skulk clusters share a network, set a custom
namespace in the app on **every** node that should belong together. Nodes with
different namespaces cannot form one cluster.

## What success looks like

The dashboard first shows every connected node and its available memory. Start
with one node, then add machines one at a time so a discovery or permission
problem is easy to isolate.

*These captures show the actual dashboard connected to a live cluster.*

![Skulk dashboard topology showing connected nodes and ready model placements](./imgs/dash-1.png)

After a model reports ready, open Chat and send a short prompt. The response is
generated by your own Skulk node or cluster.

![Skulk dashboard chat ready for a new conversation with a placed model](./imgs/dash-3.png)

## Configure Skulk from the dashboard

Choose **Open Dashboard** from the desktop app, then open **Settings**. Expand
the section you need and select **Save changes** when finished. You do not need
to edit a configuration file for ordinary setup. The
[Settings guide](configuration) covers model storage, downloads, inference,
Hugging Face access, logging, Intelligent Fabric, telemetry, and pairing.

### Add a Hugging Face token

Public repositories normally need no token. Gated or private repositories require
an account with access; accept any model terms using that same account first.

1. Create a token with model-read access in your
   [Hugging Face account](https://huggingface.co/settings/tokens).
2. Open dashboard **Settings → HuggingFace** on any node.
3. Paste it into **API Token** and select **Save changes**.
4. Reopen the section to confirm **Token is configured**, then retry the download.

The token synchronizes across the cluster, including the model-store host and
nodes that join later. A dashboard-supplied token takes effect without restarting
Skulk. Leaving the token field blank retains the saved value; it does not erase
another node's token.

<details>
<summary>Advanced: headless token setup and launch overrides</summary>

An explicit `HF_TOKEN` supplied when a node starts overrides the dashboard's
token on that node. If your administrator configured such an override, changing
the dashboard token will not replace it. The administrator must change or remove
the launch override and restart that node.

Headless operators can use `hf auth login` to populate the Hugging Face token
file when no environment or configured token takes precedence. That file is read
at download time. Run `skulk doctor` on the downloading node to inspect the token
source without printing the secret. The [runtime guide](build-and-runtime)
covers launch environments and service configuration.

</details>

## First-run troubleshooting

| What you see | What to do |
| --- | --- |
| Skulk opened but no window appeared on macOS | Click the Skulk fox in the menu bar. A persistent Dock window is intentionally not shown. |
| The menu remains at **Stopped** on first launch | Choose **Start Skulk**. The initial startup is deliberately user-triggered. |
| The runtime does not reach **Ready** | Choose **Reveal Runtime Log** on macOS or **Open Runtime Log** in the Linux app. Headless operators can inspect `journalctl --user -u skulk`. |
| A second local node never appears | Confirm every node uses the same version and namespace. On macOS, enable **System Settings → Privacy & Security → Local Network → Skulk**, then stop and start Skulk. |
| macOS asks for Screen & System Audio Recording | Deny it. That permission is not part of the Skulk desktop contract. Include the app version when reporting the prompt. |
| The dashboard opens but shows only one node | That is a working one-node cluster. Start the other nodes, then troubleshoot discovery only if they do not join. |
| A gated model fails to download | Open **Settings → HuggingFace**, enter a token, and choose **Save changes**. Accept the model terms on the token's account, then retry the download. If it still fails, ask the administrator to check for a launch-time token override. |
| `sudo` looks frozen while asking for a password | Type the Linux account password and press Enter; no typing feedback is shown. |

The macOS app does not start Skulk automatically at login. On Linux desktop,
choosing **Start Skulk** enables the user service, so it normally starts again
at later logins. Linux headless operators can explicitly enable the user
service as described above.

## Source and development installs

Use the source installer when you are contributing to Skulk, testing the
development branch, installing on a Linux distribution that the packages do
not cover, or need direct control over the source environment:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash
```

The stable installer targets `main`. To install the development branch that
matches the `/next/` documentation:

```bash
curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash -s -- --ref dev
```

See [Source builds and runtime paths](build-and-runtime) for deterministic
commit pins, manual setup, and the exact work performed by the installer.
