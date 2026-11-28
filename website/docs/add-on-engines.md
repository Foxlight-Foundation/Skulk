# Add-on engines

An engine is the program that loads a model and produces its output. Skulk
includes several: MLX on Apple Silicon, llama.cpp and vLLM on GPU machines,
and others for images, speech, music and video.

An **add-on engine** is one that is not part of Skulk. It is published
separately, and a machine installs it by itself the first time a model placed
there needs it. Nothing is installed on a machine that never runs such a
model, and a cluster that uses no add-on engine behaves exactly as it did
before.

This page is for the person who runs a cluster. The HTTP routes are in the
[API guide](api-guide.md#engine-packages-across-the-cluster) and the design is
in [Architecture](architecture.md).

## What you need

Three things have to be true before a model runs on an add-on engine:

1. **Add-on engines are turned on.** They are on by default. See
   [Settings](#settings).
2. **A build exists for the machine.** Each build is made for one operating
   system and CPU and one kind of accelerator: Apple GPU, NVIDIA GPU, AMD GPU
   (ROCm), GPU through Vulkan, or processor only. Some builds also need an
   exact GPU family. A machine is offered only builds it can run.
3. **The engine is listed for the model.** There are two ways:
   - The model's card names the engine in `placement.compatible_backends`,
     for example `tensorfold` or, for one accelerator only,
     `tensorfold-metal`. This is how you list an engine for a card you add
     yourself. See [Model cards](model-cards.md).
   - The Foxlight model registry lists the engine for the model in its signed
     engine-support list. This is how an engine is listed for a model from
     the registry, whose card you cannot edit. Skulk reads the list with the
     catalog; there is nothing to set.

   A model that neither names an add-on engine nor is listed for one is never
   placed on one.

## What happens the first time

1. You choose a model an add-on engine is listed for and a machine that
   could run it but does not have the engine yet.
2. The launch preview says that the engine will be installed with the model
   and how much there is to download.
3. When you launch, the machine downloads the engine and the model at the same
   time. The model loads when both are there.
4. Later launches on that machine use the engine already on its disk.

The engine files come from the Foxlight engine store. Before anything is
installed, the machine checks the publisher's signature on the build and the
digest of every file it downloads. A build that fails either check is not
installed, and the launch fails.

An engine is installed per machine. Two machines that run the same model each
install their own copy.

## Which engine runs a model

Several engines can often run the same model. For each machine, Skulk uses the
first of these that has an answer:

1. The order you set for that model.
2. The order on the model's card (`placement.backend_preference`).
3. The built-in order: engines that are part of Skulk first, add-on engines
   after them.

An order only ranks. Entries a machine does not have are skipped, so an order
never makes a model impossible to launch. You set an order for a model with
the [engine order routes](api-guide.md#engine-order-for-a-model), or for one
launch only with `engine_order` on the launch request.

## When an engine refuses a model

A build that is installed and intact can still refuse a particular model, for
example because the build does not support that model's architecture on that
accelerator. Only a load answers this, so the first launch finds out:

1. The engine refuses the load. This takes a few seconds.
2. Skulk places the model again on the next engine in its order, on the same
   machine when it has one, because the model's files are already there.
   Otherwise it looks at the rest of the cluster.
3. The model comes up under a new instance ID. The instance that was refused
   is removed, with a failure record that names the engine's answer.

When the engine's answer is that it cannot run the model at all
(`unsupported_model`), the machine remembers it for that build and model.
Later launches, repairs and previews skip that build for that model on every
machine, so the refused load is not repeated. The answer is forgotten when the
machine restarts or the build is replaced, and the engine is asked once more.

If no other engine can run the model, the launch fails and the record says the
engine refused it. Once the answer is remembered, a later launch is refused at
once, before anything is downloaded.

## Memory

An add-on engine uses memory beyond the model's own files. Each build declares
how much, and Skulk counts it when it decides whether a model fits on a
machine. Before the model loads, the machine also asks the engine what this
exact load needs and reserves the larger of the two figures. The engines panel
shows the declared amount for each build.

## Drafting

Some models come with a small draft head, which guesses several tokens ahead
so the model only has to check them. Drafting makes replies faster on some
engine builds and slightly slower on others, so each build's publisher says
whether that build drafts, as part of what they sign. Skulk follows that, and
there is nothing to set. On a build that drafts, the engine uses whatever the
model's own files bring and checks every drafted token against the model; a
model that brings nothing to draft with runs as it does without.

## See and manage engines

### In the dashboard

Open **Settings**, expand **Add-on engines**, and choose **Engines on your
machines**. The panel lists
each add-on engine and, for every machine, one of:

| Status | Meaning |
|---|---|
| Installed | The machine has a build that new launches use. |
| Not installed yet | The machine installs it the first time a model placed there needs it. The download size is shown. |
| Cannot be used | The machine has a build it cannot offer. The panel says why, for example that its files are not intact. |

For an installed engine the panel also shows how the last model load ended,
and any models the build cannot load.

From the panel you can:

- **Use this build**: make another build that is already on the machine the
  one new launches use.
- **Remove**: delete one earlier build, or the whole engine, from a machine.
  Model files are not touched.

A model that is already running keeps the build it started on until you stop
it, and an engine a running model uses cannot be removed. After you remove an
engine, the machine may install it again the next time a model needs it. To
stop an engine from being used, change the model's engine order or turn add-on
engines off.

### From a terminal

```bash
# Show every machine's add-on engines
skulk engines list

# Make a build on this machine's disk the one new launches use
skulk engines activate foxlight.tensorfold <build>

# Return to the build that was in use before
skulk engines rollback foxlight.tensorfold

# Delete one build, or the whole engine when no build is named
skulk engines remove foxlight.tensorfold [<build>]
```

Each command changes the machine it runs on. Add `--node <node id>` to change
another machine, and `--lane metal|cuda|rocm|vulkan|cpu` when a machine holds
the engine for more than one kind of accelerator. `list` prints the build
identifiers the other commands take.

### When something is wrong

`skulk doctor` reports an add-on engine a machine cannot use, such as one
whose files are missing or changed. `skulk doctor --fix` switches new launches
to another intact build on the machine when there is one. It deletes nothing.
See [Node doctor](node-doctor.md).

## Settings

Open **Settings** and expand **Add-on engines**. Both settings apply to every
machine in the cluster.

| Setting | Default | What it does |
|---|---|---|
| **Use add-on engines** | On | When off, every machine stops using its add-on engines within about a minute and installs no new ones. The engines stay on disk and come back when you turn the setting on. A model already running on one keeps running until you stop it. |
| **Install test builds** | Off | Machines also install builds that are still being tested, before they are released. Leave this off unless you are helping test an engine. An engine a machine already has is never replaced by this setting. |

The same settings in `skulk.yaml`:

```yaml
engines:
  packages_enabled: true
  package_channel: stable   # or: candidate, to install test builds too
```

Three launch-time switches also stop a machine from installing an add-on
engine:

- `SKULK_ENGINE_PACKAGES_DISABLE=1` turns add-on engines off on that machine
  only, whatever the cluster setting says.
- `SKULK_NO_ENGINE_AUTOPROVISION=1` stops every engine install on first use,
  built-in engines included.
- Offline mode (`SKULK_OFFLINE=true` or `skulk --offline`) stops all
  downloads. Engines already on disk keep working.

## Limits

- An add-on engine runs text generation on one machine. It does not split a
  model across machines and does not serve embeddings, images, speech, music
  or video.
- A request that carries tools or images is refused when the model runs on an
  add-on engine, even when the model itself supports them. The reply says so.
  A chat model that understands images can still run there and answer text.
  For the same reason the model behind [Talk to Skulk](steward.md), which
  works through tools, is never placed on an add-on engine.
- What a machine has learned about a build (its last load results and the
  models it cannot load) is kept in memory and cleared by a restart.
- A status of *Installed* says the build is intact and this version of Skulk
  can run it. It does not promise that a given model loads on it. Only a
  launch answers that.
- Published builds are never replaced. A build that turns out to be bad is
  withdrawn by its publisher, after which machines stop installing it.
