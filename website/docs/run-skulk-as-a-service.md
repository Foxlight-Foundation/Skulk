---
title: Run Skulk as a service
description: Make Skulk start automatically when your computer turns on, and restart itself if it ever crashes.
sidebar_label: Run as a service
---

# Run Skulk as a service

Make Skulk start when your computer boots, and restart itself if it ever crashes. This is what you want for any always-on cluster node.

## What you'll have when you're done

- Skulk starts automatically — no more typing `uv run skulk` every time you reboot
- If Skulk crashes, it comes back up on its own
- You'll know exactly how to check it's running, see logs, restart it, and turn it off

About 5 minutes per machine. No coding. No sudo for the standard install.

## Before you start

You need:

1. **A working Skulk install.** You should already be able to run `uv run skulk` from your Skulk folder and have it boot cleanly. If you can't, do that first — the [Build and Runtime guide](./build-and-runtime.md) walks you through it.
2. **`uv` on your PATH.** Check by running `which uv`. If you see a path, you're good. If it says "not found", install `uv` from [docs.astral.sh/uv](https://docs.astral.sh/uv/) and come back.
3. **macOS** (any recent version) **or Linux** with systemd (Ubuntu, Debian, Fedora, Arch — anything modern).

That's it.

## Install — pick your platform

### macOS

Open Terminal, `cd` into your Skulk folder, then run:

```bash
deployment/install/install-launchd.sh
```

The script does everything for you. When it finishes (a few seconds), check it's running:

```bash
launchctl print gui/$(id -u)/foundation.foxlight.skulk | grep "state ="
```

You should see:

```
state = running
```

**That's it.** Skulk will start automatically the next time you log in, and will restart itself if it ever crashes.

### Linux

Open a terminal, `cd` into your Skulk folder, then run:

```bash
deployment/install/install-systemd.sh
```

The script does everything for you. When it finishes (a few seconds), check it's running:

```bash
systemctl --user status skulk
```

You should see a line that says:

```
Active: active (running)
```

**That's it.** Skulk will start automatically when the machine boots, and will restart itself if it ever crashes.

:::tip Why does the Linux installer ask for your password sometimes?
The installer enables "user lingering" so Skulk keeps running after you log out — that's what makes autostart work on a headless box. Lingering is normally root-only to enable, hence the password prompt. If you skip it, Skulk will only run while you're logged in.
:::

## Verify autostart actually works

The real test is rebooting your machine.

1. **Reboot.**
2. Wait about 30 seconds after it comes back.
3. Open `http://localhost:52415` in a browser. You should see the Skulk dashboard.

If the dashboard loads after a reboot without you typing anything, autostart is working. You're done.

If it doesn't, jump to [Things that go wrong](#things-that-go-wrong).

## Day-to-day operations

| What you want to do | macOS | Linux |
| --- | --- | --- |
| Check if it's running | `launchctl print gui/$(id -u)/foundation.foxlight.skulk \| grep "state ="` | `systemctl --user status skulk` |
| Watch the logs live | `tail -f ~/.cache/skulk/logs/skulk.stderr.log` | `journalctl --user -u skulk -f` |
| Restart it | `launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk` | `systemctl --user restart skulk` |
| Stop it (stays stopped) | `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk.plist` | `systemctl --user stop skulk` |
| Start it back up | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk.plist` | `systemctl --user start skulk` |

### Updating Skulk after `git pull`

Pull the new code, rebuild the dashboard, then restart the service so it picks up the changes:

```bash
git pull
cd dashboard-react && npm install && npm run build && cd ..

# macOS:
launchctl kickstart -k gui/$(id -u)/foundation.foxlight.skulk
# Linux:
systemctl --user restart skulk
```

That's the whole update flow.

## Things that go wrong

### "I rebooted but Skulk didn't come back up"

**On Linux**, this almost always means user lingering didn't get turned on. Linux user services normally stop the moment you log out — and rebooting counts as logging out. Fix:

```bash
sudo loginctl enable-linger $USER
sudo reboot
```

After lingering is on, Skulk will come up on its own at boot.

**On macOS**, LaunchAgents start when you log in, not at boot. If your Mac auto-logs-in at boot (System Settings → Users & Groups → Login Options → Automatic login), Skulk will start automatically. If you have to type a password to log in, Skulk waits for you to do that first.

### "The status says `running` / `active` but the dashboard doesn't load"

Wait 10–20 seconds — Skulk needs a moment to boot networking and load the dashboard.

If the dashboard still doesn't load, check the logs (see the table above). Look for lines that say `ERROR` or `CRITICAL`. The most common causes:

- **Another program is using port 52415.** Find it with `lsof -i :52415` and stop it (or change Skulk's port with `--api-port`).
- **A typo in your `skulk.yaml`.** Skulk logs the parse error on startup — search the log for "config".
- **You moved your Skulk folder after running the installer.** Re-run the installer; it'll update the path.

### "It keeps crashing in a loop"

Both macOS and Linux give up after 5 crashes within 5 minutes. This is on purpose — a broken config shouldn't hammer your machine forever. To get back to a working state:

```bash
# macOS — stop, fix the cause, then start again
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk.plist
# (fix whatever's broken)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk.plist

# Linux — same idea
systemctl --user stop skulk
# (fix whatever's broken)
systemctl --user reset-failed skulk
systemctl --user start skulk
```

To figure out what's wrong, read the logs. Skulk almost always writes a `CRITICAL ...` line right before it dies, telling you exactly what went wrong.

## Uninstall

Removes the service so Skulk doesn't start automatically anymore. Doesn't touch your models, configs, or anything under `~/.skulk` — only the service definition is removed.

### macOS

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/foundation.foxlight.skulk.plist
rm ~/Library/LaunchAgents/foundation.foxlight.skulk.plist
```

### Linux

```bash
systemctl --user disable --now skulk
rm ~/.config/systemd/user/skulk.service
systemctl --user daemon-reload
```

## What survives a reboot

You don't need to read this for normal use — Skulk handles reboots cleanly. Here for the curious.

**Survives:**

- Your downloaded models
- Custom model cards you've added
- Your `skulk.yaml` config
- The cluster's event log (Skulk replays this on startup to figure out what was going on)

**Doesn't survive (and doesn't need to):**

- In-flight inference requests (the client gets a connection error and retries)
- Currently-running model placements (the cluster re-elects and re-plans in seconds)
- libp2p peer connections (everyone redials)

## Power loss

If a node loses power without warning:

- Your filesystem (APFS on macOS, ext4 on Linux) protects the data on disk.
- Skulk's event log is written so that a half-written entry from sudden power loss is detected and dropped on the next startup. Your model state and config are safe.
- Any inference request that was running at the exact moment of power loss is lost. The client gets a connection error and retries.

For nodes hosting valuable session state, a small UPS that gives the OS a few seconds to shut down cleanly is worth the money. For experimental or worker-only nodes, no UPS is fine — they recover by themselves.

## Advanced: server-style install (Linux only)

The standard install is "user-level" — Skulk runs as you, under your login, with lingering enabled. This is the right choice for almost everyone, **including headless servers**.

You only need a system-level install if you have a strict "no user services" policy or your security team requires Skulk to run as a dedicated service user. Trade-off: no GPU/Metal access on Linux when running as a dedicated user, so this is only useful for CPU-only worker nodes.

```bash
sudo useradd --system --create-home --home-dir /var/lib/skulk --shell /bin/bash skulk
# Clone Skulk into /var/lib/skulk/repo as the skulk user, then:
sudo cp deployment/systemd/skulk.service /etc/systemd/system/skulk.service
sudo sed -i "s|__SKULK_REPO__|/var/lib/skulk/repo|g" /etc/systemd/system/skulk.service
sudo sed -i "/\[Service\]/a User=skulk\nGroup=skulk" /etc/systemd/system/skulk.service
sudo systemctl daemon-reload
sudo systemctl enable --now skulk
```
