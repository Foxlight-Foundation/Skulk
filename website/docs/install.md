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
| Apple Silicon Mac running macOS 15 or newer | Signed and notarized Skulk menu-bar app through Homebrew |
| Ubuntu or Debian desktop, `amd64` or `arm64` | Skulk app and runtime through the Foxlight APT repository |
| Headless Ubuntu or Debian, `amd64` or `arm64` | Runtime-only APT package |
| Contributor workstation, development branch, or another Linux distribution | [Source installer](#source-and-development-installs) |

## macOS

Install the signed and notarized Apple Silicon app from the official Foxlight
Homebrew tap:

```bash
brew install --cask Foxlight-Foundation/skulk/skulk
```

Open **Skulk** from Applications. Its menu-bar control shows the node state and
provides **Start Skulk**, **Stop Skulk**, **Open Dashboard**, and **Open Logs**.
The app includes the exact Skulk runtime and dashboard built for its release, so
you do not need to install or approve `uv`, Python, Node.js, or a source tree.

To update later:

```bash
brew upgrade --cask Foxlight-Foundation/skulk/skulk
```

Skulk checks Foxlight's stable release manifest when the app starts and also
offers **Check for Updates…** in the menu. When a newer stable version is
available, **Download _version_…** opens its signed DMG. The app never silently
downloads or replaces itself; Homebrew users can continue to update with the
command above.

To uninstall:

```bash
brew uninstall --cask skulk
```

## Ubuntu and Debian

Install the Foxlight repository keyring, refresh APT, and install Skulk:

```bash
curl -fLO https://apt.foxlight.ai/foxlight-archive-keyring.deb
sudo apt install ./foxlight-archive-keyring.deb
sudo apt update
sudo apt install skulk
```

The public keyring package contains only the repository's public signing key
and APT source definition. The repository's private signing key is never
distributed.

The `skulk` package installs both the desktop controller and the matching
runtime. Open **Skulk** from the application menu and select **Start Skulk**.
The app enables and starts the packaged `skulk.service` user unit, then gives
you controls for the dashboard, logs, node lifecycle, and cluster namespace.
Installing the package alone does not start the service without your action.

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

1. Install the same Skulk version on every machine.
2. Start Skulk on each machine from the app or service manager.
3. Open the dashboard from any running node and confirm the machines appear in
   the topology.
4. Pick and launch a model. Skulk places it on compatible hardware and begins
   serving when the placement is ready.

Nodes on the same network use Skulk's shared default namespace and discover one
another automatically. You do not need to invent an identifier for an ordinary
cluster. If multiple independent Skulk clusters share a network, set a custom
namespace in the app on **every** node that should belong together. Nodes with
different namespaces cannot form one cluster.

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
