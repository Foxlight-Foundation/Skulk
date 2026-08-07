"""Reference Skulk provider extension: serves a trivial ``echo`` capability.

The smallest complete example of the provider role (fabric-citizenship):

- ``capabilities()`` publishes a self-describing :class:`CapabilityDescriptor`
  (id + semantic version + JSON Schemas + I/O mode). The descriptor's id is
  auto-advertised as the node's telemetry discovery tag, and peers fetch the
  full descriptor via ``describe_node`` / ``GET /v1/capabilities``.
- ``on_start`` receives the live ``ExtensionContext`` at node startup, so a
  provider can do startup registration without depending on a chat request.

Install into a node's venv (``uv pip install -e .``) and restart the node;
every peer then sees ``echo`` in its ``read_cluster()`` snapshot for this node
and can describe it. Serving actual calls arrives with the capability-call
phase; until then this extension is discovery-complete but not yet callable.
"""

from loguru import logger

from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    ExtensionContext,
)

_ECHO_DESCRIPTOR = CapabilityDescriptor(
    id="echo",
    version="1.0.0",
    title="Echo",
    description=(
        "Returns the input text unchanged. A reference capability proving the "
        "provider contract end to end; call it with {\"text\": \"...\"}."
    ),
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo."}},
        "required": ["text"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The same text."}},
        "required": ["text"],
    },
    io_mode="unary",
)


class EchoProviderExtension:
    """Provider extension serving the ``echo`` capability."""

    @property
    def name(self) -> str:
        """Short unique extension name, used in logs."""
        return "echo-provider"

    @property
    def skulk_requires(self) -> str:
        """Compatible with any Skulk that ships the provider surface."""
        return ">=1.4.2"

    def chat_middleware(self) -> None:
        """A pure provider: no chat hooks."""
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        """The capabilities this extension serves."""
        return [_ECHO_DESCRIPTOR]

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        """Serve one echo call: return the input text unchanged."""
        return {"text": call.payload["text"]}

    def on_start(self, context: ExtensionContext) -> None:
        """Log the fabric this provider joined (startup registration hook)."""
        logger.info(
            f"echo-provider started on node {context.node_id} "
            f"(skulk {context.skulk_version}); serving "
            f"{_ECHO_DESCRIPTOR.qualified_id}"
        )
