---
id: ecosystem
title: The Skulk Ecosystem
description: How Skulk nodes, operator apps, model metadata, published weights, and capabilities work together.
---

Skulk connects computers into a fabric that runs AI workloads and exposes them
through one operator experience and compatible APIs. A model can run on one
suitable node or, where its engine supports it, span multiple nodes. Other
capabilities extend what the fabric can do beyond inference.

## Which part does what?

| Part | Its role | What you use it for |
| --- | --- | --- |
| Skulk nodes | Discover peers, coordinate placement, load models, run inference, and expose the dashboard and APIs | Supply compute and operate the fabric |
| Desktop app | Starts and stops the local Skulk runtime and opens its dashboard | Run a node from a macOS menu bar or Linux desktop |
| [Native operator app](native-app.md) | Connects an iOS or Android device to one paired cluster | Monitor, manage models, chat, and follow activity remotely |
| [Relay and operator gateway](remote-access.md) | Carry authenticated, encrypted app requests to the cluster | Reach the cluster without installing a VPN on the phone |
| [Foxlight Model Registry](model-cards.md) | Publishes signed descriptions of exact model artifacts and separate engine-support evidence | Discover models and understand their requirements |
| [Model store](model-store.md) | Keeps the cluster's downloaded model artifacts and supplies worker staging | Download once and reuse artifacts across the cluster |
| Skulk Weights Publisher | Prepares and publishes model companions and other weight artifacts | Supply the bytes referenced by model cards |
| [Capabilities and plugins](capability-nodes.md) | Add discoverable services with their own permissions, configuration, and lifecycle | Extend the fabric with functions such as governed cloud capacity |
| Intelligent Fabric | Provides the cluster-aware Skulk conversation through the built-in Steward runtime | Ask about the fabric and prepare supported actions |

The desktop app and phone app have different jobs. The desktop app runs a
local compute node; the phone app is its remote operator. A phone does not add
model memory or hold the cluster's weight files.

## From a model listing to an answer

1. Skulk verifies the registry's signed catalog. A card identifies an exact
   artifact, its capabilities, companion dependencies, and placement requirements.
2. Downloading transfers the selected weights and required companions into the
   model store, when configured. A catalog listing alone does not mean those
   files have been downloaded.
3. Placement chooses compatible, available nodes. Workers stage and verify the
   artifacts, then start the appropriate inference engine. Downloaded files alone
   do not mean a model is running.
4. Once an instance is ready, dashboard chat, the operator app, and API clients
   can use it. The selected engine performs generation; the registry and relay
   do not perform inference.
5. Stopping the instance releases its runtime resources. Deleting stored weights
   and clearing staged copies are separate operations.

The registry's **model catalog** and the cluster's **store registry** are
different inventories: the first describes selectable artifacts; the second
tracks the cluster's canonical downloaded copies. The native app preserves
catalog, stored, staged, and running state separately for the same reason.

## Model metadata and published weights

Foxlight Model Registry publishes metadata through The Update Framework (TUF).
Its administrative services are separate from the public signed objects Skulk
reads. A valid signed card establishes artifact identity and catalog trust;
compatibility still depends on engine support, the actual node build and
hardware, and Skulk's runner limitations. An installed, complete artifact retains
its effective card for operation without a fresh registry connection.

Skulk Weights Publisher supplies artifacts rather than placement decisions. For
example, it extracts native multi-token-prediction heads into an MTP sidecar or
mirrors a required vision encoder. The registry can bind a completed sidecar's
immutable revision to a model card; Skulk then fetches and stages that companion
with the base model. Users normally select the base model, not its sidecar.
A companion is not an independently runnable chat model.

[Speculative decoding](speculative-decoding.md) also supports companions published
by model authors, such as Gemma assistant models, and engines that consume
prediction heads embedded in the main checkpoint. These are different execution
contracts, even when they serve the same goal of faster decoding.

## Extending the fabric

Built-in capabilities such as speech use Skulk's own runtime. Installed plugins
can expose additional capabilities through the same fabric discovery and call
surfaces. A capability node is a logical provider with its own status and
configuration; it need not be another physical computer.

The Skulk conversation uses the reserved `skulk/steward` model identity. Its
runtime belongs to core Skulk; the separate `skulk-steward` project supplies evaluation scenarios and a model-comparison harness for
cluster investigation, tool use, and evidence-grounded answers. Installed plugin tools appear
only when their configuration, readiness, permissions, and host policy allow
it. Describing an action in chat is not evidence that it executed, and preparing
a billable proposal does not approve spending.

Plugin availability depends on its publisher. In particular, the RunPod plugin
is supplied through private owner distribution, not a public plugin store.
See [Capabilities and plugins](capability-nodes.md) for the installation,
approval, and cleanup boundaries.
