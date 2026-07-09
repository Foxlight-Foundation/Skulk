"""Public types for Skulk extensions (plugins).

Extensions are separately installed Python packages that Skulk discovers at
startup through the ``skulk.extensions`` entry-point group and loads as
first-class citizens of the fabric. The contract is deliberately small and
general: an extension can transform a chat request before it is dispatched to
the cluster and observe the completed response after it streams back. Nothing
in this module is specific to any particular extension.

Design invariants (enforced by the call sites in :mod:`skulk.api.main` and
:mod:`skulk.extensions.loader`):

- **Extensions must never degrade inference.** Every extension call is
  guarded; a raising extension is logged loudly and the request proceeds as
  if the extension did not exist.
- **Extensions never own the response stream.** Skulk accumulates the
  response and hands extensions an immutable summary, so a buggy extension
  cannot corrupt or stall token delivery.
- **No extension installed = Skulk unchanged.** The hooks are inert when the
  loaded-extension list is empty.
"""

from dataclasses import dataclass
from typing import Protocol, final

from pydantic import BaseModel

from skulk.extensions.telemetry import ClusterNodeView
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.text_generation import TextGenerationTaskParams


class EmbedTexts(Protocol):
    """Async callable that embeds texts via the cluster's embedding serving.

    Returns one embedding vector per input text, or ``None`` when no
    embedding instance is available or the call fails or times out. Callers
    must treat ``None`` as "degrade gracefully", never as an error to raise.
    """

    async def __call__(
        self, texts: list[str], model_id: ModelId | None = None
    ) -> list[list[float]] | None: ...


class ReadClusterTelemetry(Protocol):
    """Synchronous callable returning a snapshot of the telemetry plane.

    Each call returns an immutable, per-node projection of what this node
    currently sees on the telemetry plane (peers, their backends, participation
    role, accelerator vendor, version, and liveness). It is a read: an extension
    can observe the cluster it belongs to but never mutate telemetry. Cheap and
    side-effect free (an in-memory snapshot, no network I/O), so it is safe to
    call from an inline hook.
    """

    def __call__(self) -> tuple[ClusterNodeView, ...]: ...


class AdvertiseCapability(Protocol):
    """Synchronous callable that advertises a capability on the telemetry plane.

    The extension calls this to publish an opaque capability tag (for example
    ``"memory"``) that peers then see in their
    :attr:`ClusterNodeView.capabilities`. It is the *advertise* half of
    telemetry-plane access: how a plugin announces what it offers, the same way
    native nodes advertise their backends.

    Advertising is additive and idempotent: re-advertising the same tag is a
    no-op, and the tag keeps being gossiped (last-write-wins) until the node
    leaves the cluster. Cheap and side-effect free beyond recording the tag (it
    only mutates a local set; the gossip happens on the node's normal telemetry
    poll), so it is safe to call from an inline hook (typically once at startup).
    """

    def __call__(self, capability: str) -> None: ...


@final
@dataclass(frozen=True)
class ExtensionContext:
    """Capabilities Skulk hands to an extension at each hook invocation.

    Attributes:
        node_id: The identity of the node running this API process.
        skulk_version: The installed Skulk version (PEP 440 string).
        embed_texts: Programmatic access to the cluster's embedding serving
            (the in-process equivalent of ``POST /v1/embeddings``).
        read_cluster: The telemetry-plane read surface: returns an immutable
            per-node snapshot of the cluster (fabric-citizenship Phase 1). How a
            plugin discovers its peers and their health.
        advertise_capability: The telemetry-plane advertise surface: publishes
            an opaque capability tag this node offers so peers discover it via
            their own ``read_cluster`` snapshots.
    """

    node_id: NodeId
    skulk_version: str
    embed_texts: EmbedTexts
    read_cluster: ReadClusterTelemetry
    advertise_capability: AdvertiseCapability


@final
class ChatResponseSummary(BaseModel, frozen=True):
    """Immutable summary of a completed chat generation, for observers.

    Attributes:
        text: The concatenated non-thinking output text.
        thinking_text: The concatenated reasoning/thinking text, if any.
        finish_reason: The terminal finish reason (``stop``, ``length``,
            ``tool_calls``, ``error``, ...), or ``None`` when the stream
            ended without a terminal chunk (for example a client disconnect).
        had_error: Whether the stream terminated with an error chunk.
    """

    text: str
    thinking_text: str
    finish_reason: str | None
    had_error: bool


class ChatMiddleware(Protocol):
    """Per-request chat hooks an extension can provide.

    Both methods are optional in spirit; subclass :class:`BaseChatMiddleware`
    to implement only one of them.
    """

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        """Return (possibly) modified task params before cluster dispatch.

        Runs on the API node after the OpenAI adapter and before the command
        is sent to the master. Raising is safe: the request proceeds with the
        params returned by the previous middleware (or the originals).
        """
        ...

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        """Observe a completed generation.

        Scheduled as a background task once the response finishes (or the
        stream ends); never blocks token delivery. Raising is safe and only
        logs.
        """
        ...


class BaseChatMiddleware:
    """No-op :class:`ChatMiddleware` base; override what you need."""

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        """Return the params unchanged (identity transform)."""
        return task_params

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        """Ignore the response (no-op observer)."""
        return None


class SkulkExtension(Protocol):
    """The object a ``skulk.extensions`` entry point must produce.

    The entry point value is a zero-argument callable (class or factory)
    returning an instance satisfying this protocol.
    """

    @property
    def name(self) -> str:
        """Short unique extension name, used in logs."""
        ...

    @property
    def skulk_requires(self) -> str:
        """PEP 440 version specifier for compatible Skulk versions.

        Example: ``==1.3.1`` or ``>=1.3.1,<1.4``. An extension whose
        specifier does not match the running Skulk is refused at load time
        with a loud error (mixed plugin/fabric versions are the same
        anti-pattern as mixed-version clusters).
        """
        ...

    def chat_middleware(self) -> ChatMiddleware | None:
        """Return this extension's chat middleware, or ``None`` for none."""
        ...
