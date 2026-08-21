---
id: integrations
title: Connect Coding Agents and Apps to Your Cluster
sidebar_label: Integrations
sidebar_position: 8
description: The dashboard's Integrations page generates ready-to-paste configuration for coding agents, chat front ends, and workflow tools, filled in with the models you actually have running.
---

# Connect coding agents and apps to your cluster

Your cluster speaks the API formats these tools already know, so pointing them at
it is a configuration change rather than an integration project. The dashboard's
**Integrations** page writes that configuration for you.

Open the dashboard and choose **Integrations** in the navigation, or go straight
to `/integrations`.

## What the page gives you

Pick a tool and you get the exact blocks it needs: a shell command, a config
file, or the settings to type into an application's own screen. Each block has a
copy button.

The blocks are not generic examples. They are generated from your cluster as it
stands right now, so they already contain:

- the address of this node that other machines can actually reach, rather than
  `localhost`
- the ids of the models that currently have a ready instance
- each model's real context window
- per-model capability flags, so a vision model is declared as accepting images
  and a reasoning model is set up to send its thinking back on later turns

If nothing is running yet, the blocks still show the correct shape with a
placeholder where the model id belongs. Mount a model and they fill themselves
in.

## Supported tools

**Coding agents**

| Tool | What you get |
| --- | --- |
| Claude Code | A shell command and a `~/.claude/settings.json` block. Choose which of your models answers as Opus, Sonnet and Haiku. |
| OpenCode | An `opencode.json` provider block covering every ready model. |
| Codex | A `~/.codex/config.toml` with the provider and a filesystem MCP server, plus a launch command. |
| Hermes | A `~/.hermes/config.yaml` for its custom endpoint provider, the interactive setup command, and the stream timeout to raise for long turns. |
| OpenClaw | A `~/.openclaw/openclaw.json` and the commands that start its gateway. |
| Pi | A `~/.pi/agent/models.json` and a launch command. |

**Applications and workflows**

| Tool | What you get |
| --- | --- |
| AnythingLLM | A Docker command, the equivalent desktop-app settings, and the optional embedder settings so document indexing also runs on the cluster. |
| Open WebUI | A Docker command using the Ollama-compatible surface, plus the equivalent Ollama CLI command. |
| n8n | A Docker command, the OpenAI credential to create, and the workflow nodes to wire up. |
| Firefox | The `about:config` keys that make this dashboard Firefox's built-in AI sidebar. |

## The three surfaces

Every recipe uses one of the three request formats the cluster serves, all shown
at the top of the page:

| Surface | Address | Used by |
| --- | --- | --- |
| OpenAI-compatible | `<node>/v1` | Most tools |
| Anthropic-compatible | `<node>` | Claude Code |
| Ollama-compatible | `<node>/ollama` | Open WebUI, the Ollama CLI |

Any tool that accepts a custom OpenAI base URL works even if it is not listed
above. Point it at `<node>/v1`, give it any non-empty API key, and use a model id
from the page's ready-models list.

## Choosing which address to embed

A configuration file is usually pasted into a tool running on a different
machine, so the page never embeds `localhost`. It uses the node's own routable
address instead.

When the node is also reachable over Tailscale, an **Address to use** control
appears so you can choose between the local network address and the Tailscale
one. Pick Tailscale when the tool runs outside your home network. See
[Tailscale](./tailscale.md) for setting that up.

Recipes that run in Docker rewrite the address to `host.docker.internal`
automatically, because a container that dialled the loopback address would reach
itself rather than the cluster.

## Authentication

Skulk does not authenticate requests on a trusted fabric, so the API key in
every snippet is a placeholder that exists only because most clients refuse to
start without one. Any non-empty value works. Treat the cluster API as you would
any other service on your local network.
