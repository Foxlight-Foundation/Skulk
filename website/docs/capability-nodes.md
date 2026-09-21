---
id: capability-nodes
title: Capabilities and Plugins
description: Install and operate capabilities, understand permissions, and review governed cloud-capacity proposals.
---

A **capability** is something the Skulk fabric can do. A **plugin** is an
installable package that adds capabilities. A **capability node** is a logical
provider with its own status and settings; several can share one physical host.
Built-in speech capabilities run through Skulk's own model capacity, while
installed plugins can supply other services.

Skulk core owns discovery, routing, plugin management, and the dashboard. The
plugin supplies its capability implementation, configuration schema, setup checks,
and any provider-specific policy. Process separation helps lifecycle management;
it is not a sandbox against malicious code running as the same operating-system
user. Install only bundles from publishers you trust.

![Managed plugin inventory in the live dashboard](./imgs/dashboard-plugins.png)

*The actual plugin inventory at capture time; installed capabilities vary by cluster.*

## Get a compatible plugin

Obtain the release and owner instructions from its publisher. Check supported
platforms, exact runtime compatibility, permissions, required services, costs,
and cleanup procedures before installing. A matching version label alone is not
proof of compatible bytes or dependencies.

The RunPod cloud-capacity plugin is distributed privately to owners with access.
There is no public plugin store. Its supplier must provide the compatible release,
trusted publisher information, download access, and complete administration
handoff. You do not need to clone its private source repository for ordinary use.

## Install and enable

Open **Plugins** on the host that will manage the installation. Initial service
setup and publisher trust require owner access; a paired browser cannot grant
itself that authority.

1. Prepare the host's plugin-management service according to the supplied release
   instructions. This can require a one-time local administrator step.
2. In **Add plugin**, enter the supplied release source, metadata filename,
   publisher trust, and feed credential where required.
3. Save the source and inspect the release. Review its signature, compatibility,
   expiry, and permissions before downloading.
4. Download and install, then accept the reviewed permissions and activate the
   staged release.
5. Configure the capability, supply credentials through the separate credential
   controls, complete its setup actions, and select **Check setup**.
6. Correct any reported issues and enable the capability once checks pass.

Installation, activation, configuration, readiness, and enablement are different
states. An installed plugin can expose setup controls without advertising a ready
capability. Credentials are write-only through management: read responses show
requirements and readiness, not the secret values.

If a download or setup request loses its connection, inspect the existing
operation and its recovery action. Accepted work may continue after the browser
closes; creating another installation is not a safe way to resume it.

## Permissions and remote management

[Browser access](remote-access.md#pair-a-browser-for-plugin-operations) supports
three separate grants:

| Scope | Authority |
| --- | --- |
| `plugins:read` | Read installation state, checks, setup progress, and proposal reviews |
| `plugins:manage` | Perform supported configuration and management actions |
| `plugins:approve` | Approve an exact reviewed proposal or a setup action that requires approval |

Ordinary pairing grants none of these automatically. Trust setup and changes to
credential destinations remain direct-owner operations. Plugin management does
not imply permission to spend money.

## Example: adding cloud GPU capacity

The RunPod plugin can prepare a temporary GPU for a selected model, join that
capacity to Skulk, and make the ready model available through ordinary inference.
The plugin's owner distribution requires a supported management host and a
separate cleanup host, a RunPod account and API key, a verified secure connection
between hosts, and explicit spending and lifetime limits. The cleanup host must
remain available until outstanding resources are confirmed gone.

After the supplier's setup procedure, enable **Intelligent Fabric** in Skulk
Settings and enable proposals in the plugin's configuration. These controls
permit proposal preparation, not purchases.

A typical request to the Skulk conversation is:

> Prepare a RunPod capacity proposal for MODEL_ID with an 8192-token context.
> Show the hardware, cost limits, and cleanup terms for review.

Replace `MODEL_ID` with an exact catalog selection. This example does not imply
that every model has suitable cloud capacity or that every placement shortfall
automatically rents a GPU.

Open **Plugins → Review proposals**, refresh, and review the retained proposal.
Check the model, context, hardware, location, price, maximum cost, campaign usage,
resource lifetime, and approval expiry. **Approve and execute reviewed proposal**
is the distinct action that can begin paid work. Keep its operation ID and follow
progress; a created Pod is not yet a ready model. Start chat only once the requested
model instance reports ready.

If approval or acquisition has an uncertain result, observe that same operation.
Do not submit another acquisition to retry an ambiguous create. Proposal records,
approval history, and cleanup receipts must be preserved across restarts.

## Finish work and confirm cleanup

For a cloud rental, prepare and review a release proposal, approve it, and follow
cleanup until the provider confirms the resource is **absent**. An acknowledged
delete is not proof of absence. Independent expiry cleanup retains its obligation
if the management host stops, but provider outages can delay observation or
termination; the deadline is not a guaranteed billing cutoff.

**Disable** and **Uninstall** withdraw future plugin work. Neither confirms that
an existing cloud resource is gone. RunPod Network Volumes are separate storage
resources and are not removed by Pod cleanup. Use the supplier's recovery
instructions before purging installation state or replacing an owner.

## For capability authors

Skulk exposes generic capability discovery and call contracts alongside separate
plugin-management APIs. Implementations should declare their schemas, effects,
readiness, bounds, and lifecycle explicitly. Model tools can prepare supported
proposals, but the provider's authority checks remain responsible for execution.
See [Extensions](extensions.md), the [Architecture reference](architecture-reference.md),
and the [API guide](api-guide.md) for core contracts. The private RunPod package is
an implementation of these boundaries, not a public general-purpose SDK promise.
