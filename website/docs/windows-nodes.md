# Windows nodes (management only)

A Windows PC can join a Skulk cluster as a **management node**. It runs the
dashboard and the API and takes part in the cluster like any other node, and
it runs no models. Models stay on the Mac and Linux nodes.

This page covers what a Windows node does, how to install and start one, how
its plugin service works, and what has and has not been tested.

Windows support is on the development branch (`dev`), which the `/next/`
documentation describes. It is not in a stable release yet, and there is no
packaged app or installer for Windows: a Windows node runs from a source
checkout.

## What a Windows node does

- **Joins the cluster.** It finds the other nodes on the same network by
  itself, syncs the cluster's state and connects the data transport.
- **Serves the dashboard and the API** on port 52415, for this machine and for
  others on the network.
- **Takes part in choosing the master, as a last resort.** A management node
  becomes master only when no ordinary node is there to do it. A new master's
  `skulk.yaml` replaces every other node's, so a machine that runs no models
  should not be the one that decides.
- **Keeps its own plugin service**, described [below](#plugins-on-windows).

## What it does not do

- **It runs no models.** Skulk has no inference engine on Windows, so a Windows
  node always starts with management participation and is never chosen to hold
  a model. Setting `SKULK_NODE_PARTICIPATION` to `full` or `ffn_only` is
  refused with a message; leave it unset.
- **No plugin can be installed on it yet.** The plugin service can be set up,
  but no published plugin release lists Windows.
- **It does not start by itself.** There is no Windows service for the node.
  It runs for as long as its terminal window is open.
  [Run Skulk as a service](run-skulk-as-a-service.md) covers macOS and Linux
  only.

## What you need

- **Windows 11 on an x86-64 PC.** Windows on ARM64 has not been tried.
- **[Git](https://git-scm.com/download/win)** and
  **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. uv
  installs the right Python itself.
- **[Node.js](https://nodejs.org/)**, to build the dashboard once.
- **No compiler.** The networking component comes as a prebuilt package.
- **The same Skulk build as the rest of the cluster.** Every node in a cluster
  must run the same version of Skulk.
- **The same network as the other nodes**, for them to find each other
  automatically.

## Install

In PowerShell, from the folder you want Skulk in:

```powershell
git clone https://github.com/Foxlight-Foundation/Skulk.git
cd Skulk
git checkout dev
uv sync --locked
cd dashboard-react
npm ci
npm run build
cd ..
```

Check out the same branch or commit your other nodes run. `uv sync --locked`
installs exactly the packages the checkout names. It needs no compiler,
because the networking component is installed as a prebuilt package. On the
test machine it took 17 seconds.

Use an ordinary PowerShell window, not one opened with **Run as
administrator**. The plugin service refuses to be set up or to run as an
administrator.

## Start it

```powershell
uv run skulk
```

The first time, Windows Firewall has to allow Skulk to accept connections from
your network. If Windows asks, allow it on **private** networks. If it does not
ask and the other nodes cannot reach this one, add the permission by hand:
open **Windows Security**, then **Firewall & network protection**, then
**Allow an app through firewall**, and add the `python.exe` Skulk runs on.
This prints where it is:

```powershell
uv run python -c "import os, sys; print(os.path.realpath(sys._base_executable))"
```

The permission belongs to that exact file. Its folder is named after the
Python version, so after a Python update Windows has to be told again.

Then open `http://localhost:52415` in a browser. On the test machine the page
answered about 20 seconds after the start, and the node listed another node on
the network between half a minute and two minutes after the start.

To stop the node, press **Ctrl+Break** or **Ctrl+C** in its window. It leaves
the cluster and exits within a few seconds.

## Where Skulk keeps its files

| What | Where |
| --- | --- |
| The cluster's settings, `skulk.yaml` | The folder you start Skulk from |
| The node's own settings, data and cache | `.skulk` in your user folder (`%USERPROFILE%\.skulk`) |
| The plugin service | `%LOCALAPPDATA%\Skulk\plugins` |

Set `SKULK_HOME`, to a folder name inside your user folder or to a full path,
to keep the node's own settings, data and cache somewhere else. It moves
nothing else: `skulk.yaml` is still read from the folder you start Skulk in,
the ports stay the same, and your account still has one plugin service, which
belongs to the first configuration that set it up.

## Plugins on Windows

The first time you open the **Plugins** page, Skulk offers to set up the plugin
service, as it does on macOS and Linux. On Windows, setup:

1. Copies Skulk's own Python environment into
   `%LOCALAPPDATA%\Skulk\plugins`, about 450 MB.
2. Registers one scheduled task for your account in Task Scheduler. This needs
   no administrator prompt.
3. Starts the service and waits until it answers.

On the test machine this took four to four and a half minutes, most of it the
copy.

Things to know about the service:

- **It runs as you, and only while you are signed in.** Windows gives an
  ordinary account no other way to keep a program running. A service for the
  whole machine is not offered.
- **A restart can take up to a minute.** Task Scheduler starts the task once a
  minute. While the service is running, each of those starts finds it there
  and ends at once, so Task Scheduler shows the task as running. After the
  service stops, or after you sign in, the next start brings it back.
- **Its task is named after your account**:
  `foundation.foxlight.skulk.plugins.` followed by your account's identifier
  (it begins `S-1-5-21-`). Two people on one PC each get their own.
- **Stopping it is done by setup**, which switches the task off and asks the
  service to stop. Ending the task in Task Scheduler also stops it, but the
  next minute starts it again unless the task is disabled.

### If setup says the path is too long

Windows limits how long a file's path may be unless long paths are switched on
for the machine. An install whose files would not fit is refused before
anything is downloaded, with a sentence that says how many characters too long
the path is. Use a shorter folder, or switch long paths on in Windows.

## Troubleshooting

**"could not listen for peers on TCP port 52416".** Skulk could not take the
port it uses to talk to other nodes. The message goes on to give Windows' own
reason, with an error number; read that first. Most often another program
holds the port: the default lies in the range Windows hands out to programs
that ask for any free port, so this can happen by chance. In that case start
Skulk again, or give it another port with `--libp2p-port` or
`SKULK_LIBP2P_PORT`. If the reason is something else, a different port may not
help.

**The node runs but sees no other node.** Check, in this order: that Windows
Firewall allows Skulk on your network (see [Start it](#start-it)); that the
machines are on the same network; that every node uses the same
`SKULK_LIBP2P_NAMESPACE`, or none; and that every node runs the same Skulk
build.

**"SKULK_NODE_PARTICIPATION=full is not available on Windows".** Unset the
variable. A Windows node always runs with management participation.

**Windows blocks `uv run`.** On the test machine, with Smart App Control
switched on, Windows blocked the small launcher uv creates three times within
a few seconds and later allowed it. The cause was not established. Try again
before changing any setting.

## What has been tested

All on one Windows 11 machine (x86-64), from a clean checkout:

- Installing without a compiler, and starting as an ordinary user.
- Joining a Linux node by discovery alone, syncing state, connecting the data
  transport, staying up while the other node left and came back, and leaving
  cleanly.
- The dashboard and the API, from the machine itself and from another machine
  on the network.
- Setting up the plugin service from nothing to ready, reconnecting to it
  after the node restarted, and stopping it.

## What has not been tested

- Windows on ARM64.
- An account that was never an administrator. The test account is an
  administrator that runs Skulk with ordinary rights.
- The plugin service coming back after signing out and in again.
- A chat request sent through a Windows node to a model on another node.
- A cluster of more than two nodes with a Windows node in it.
