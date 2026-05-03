# pyright: reportAny=false
"""
exo cluster configuration schema — ``exo.yaml``.

Overview
--------
This module defines the Pydantic models for ``exo.yaml``, the optional
cluster-level configuration file for exo.  The file is placed alongside
the project root on **every node** in the cluster.  If the file is absent,
exo behaves identically to the upstream default (zero-config compatibility).

The ``model_store`` section is the first feature gated by this config.  The
design deliberately leaves room for future sections (e.g. networking tuning,
custom inference backends, cluster-level resource policies) without breaking
existing deployments.

Model store design summary
--------------------------
In a standard exo cluster every node independently downloads its assigned
model shard from HuggingFace at inference time.  For a small home cluster
this means:

* Redundant external bandwidth — every node pulls the same files.
* Slow cold starts on large models (30B+).
* Version drift between nodes.
* No offline capability after the first run.

The model store solves this by designating one node as the **store host**.
The store host serves model files to all other nodes over HTTP on the local
network (typically Thunderbolt or 10 GbE).  Worker nodes stage their
assigned shard to node-local storage before MLX loads it.  MLX always
receives a **local filesystem path** — the inference stack is completely
unaware the store exists.

HuggingFace is used as a fallback when a model is not yet in the store (if
``download.allow_hf_fallback`` is ``true``).  For air-gapped deployments set
it to ``false`` to ensure all models come from the store.

Configuration file location
---------------------------
``exo.yaml`` must be placed at the **same relative path on every node** (i.e.
alongside the ``exo`` project root).  Node-specific behaviour is handled via
``node_overrides`` keyed by hostname or node_id rather than per-node files.

Example ``exo.yaml``::

    model_store:
      enabled: true
      store_host: mac-studio-1    # hostname of the node with attached storage
      store_port: 58080
      store_path: /Volumes/ModelStore/models

      download:
        allow_hf_fallback: true

      staging:
        enabled: true
        node_cache_path: ~/.exo/staging
        cleanup_on_deactivate: false

      node_overrides:
        mac-studio-1:
          staging:
            node_cache_path: /Volumes/ModelStore/models  # load directly from store
            cleanup_on_deactivate: false
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Literal, final

import yaml
from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel


def _normalize_hostname(hostname: str) -> str:
    """Return a comparable hostname token.

    Hostname matching is intended to be user-friendly for the common macOS
    case where one surface reports ``kite3`` while config may use
    ``kite3.local``.  We therefore compare hostnames case-insensitively and
    ignore any trailing dot used in FQDN notation.
    """

    return hostname.strip().rstrip(".").lower()


def hostname_aliases(hostname: str) -> set[str]:
    """Return hostname aliases that should identify the same machine.

    The alias set includes the normalized hostname itself, the short hostname,
    and a ``.local`` variant for short names.  This keeps config matching
    stable across macOS host reporting differences without weakening node-id
    matching semantics.
    """

    normalized = _normalize_hostname(hostname)
    if not normalized:
        return set()

    aliases = {normalized}
    short_hostname = normalized.split(".", 1)[0]
    if short_hostname:
        aliases.add(short_hostname)
        aliases.add(f"{short_hostname}.local")
    return aliases


def node_matches_store_host(
    store_host: str,
    node_id: str,
    hostname: str | None = None,
) -> bool:
    """Return whether the local node should consider itself the store host.

    Matching remains strict for libp2p peer IDs, while hostname matching is
    tolerant of short-name versus ``.local`` spelling differences.
    """

    if store_host == node_id:
        return True
    return _normalize_hostname(store_host) in hostname_aliases(
        hostname or socket.gethostname()
    )


@final
class StagingNodeConfig(FrozenModel):
    """Per-node staging configuration.

    Controls where model files are staged on the node-local filesystem
    before MLX loads them, and whether those files are cleaned up when
    the model instance is deactivated.

    Attributes:
        enabled: When ``False`` staging is skipped entirely and MLX loads
            directly from the store path (only useful on the store host when
            ``node_cache_path == store_path``).
        node_cache_path: Absolute or ``~``-prefixed path to the directory
            where staged model files are written.  Each model occupies a
            subdirectory named ``<org>--<model>`` (e.g.
            ``mlx-community--Qwen3-30B-A3B-4bit``).
        cleanup_on_deactivate: When ``True``, staged files are deleted when
            the model instance is shut down, freeing local disk space at the
            cost of making later placements cold again. The default keeps the
            staged files so large models can be reused from local storage;
            use the explicit staging-cache purge endpoint for cleanup.
    """

    enabled: bool = True
    node_cache_path: str = "~/.exo/staging"
    cleanup_on_deactivate: bool = False


@final
class DownloadStoreConfig(FrozenModel):
    """Download policy for the model store.

    Attributes:
        allow_hf_fallback: When ``True`` (the default), nodes fall back to
            downloading from HuggingFace if a requested model is not present
            in the store.  Set to ``False`` for air-gapped clusters where all
            models must be pre-staged in the store.
    """

    allow_hf_fallback: bool = True


@final
class NodeOverrideConfig(FrozenModel):
    """Per-node configuration overrides.

    Overrides are matched by **hostname** (``socket.gethostname()``) or by
    libp2p **node_id**.  The first match wins.  Unspecified fields fall back
    to the base configuration.

    Attributes:
        staging: Node-specific staging settings.  ``None`` means "use the
            base staging config unchanged".
    """

    staging: StagingNodeConfig | None = None


@final
class ModelStoreConfig(FrozenModel):
    """Configuration for the cluster-wide model store feature.

    All fields except ``store_host`` and ``store_path`` have sensible
    defaults so most clusters only need to set those two values.

    Attributes:
        enabled: Master switch.  ``False`` disables the store entirely even
            if the config file is present — useful for temporarily reverting
            to standard HF downloads without removing the file.
        store_host: Hostname or node_id of the node that hosts the model
            store.  Used only for identity resolution (determining whether
            this node is the store host).  Hostname matching accepts the
            usual short-name and ``.local`` variants for the same machine,
            while node_id matching remains exact.
        store_http_host: Hostname or IP address used by worker nodes to
            reach the store host over HTTP.  Defaults to ``store_host``
            when ``None``.  Set this when ``store_host`` is a libp2p peer
            ID (which is not a valid DNS name) so that workers can still
            resolve the HTTP address (e.g. ``mac-studio-1`` or
            ``192.168.1.10``).
        store_port: HTTP port for ``ModelStoreServer`` on the store host.
            Must be reachable from all worker nodes.
        store_path: Absolute path to the model store root on the store host.
            All model directories live here as
            ``<store_path>/<org>--<model>/``.
        download: Download fallback policy.
        staging: Default staging config applied to all nodes that do not have
            a matching entry in ``node_overrides``.
        node_overrides: Per-node config overrides keyed by hostname or
            node_id.  Typically used to configure the store host to load
            directly from ``store_path`` instead of making a local copy.
    """

    enabled: bool = True
    store_host: str
    store_http_host: str | None = None
    store_port: int = 58080
    store_path: str
    download: DownloadStoreConfig = DownloadStoreConfig()
    staging: StagingNodeConfig = StagingNodeConfig()
    node_overrides: dict[str, NodeOverrideConfig] = {}


@final
class TailscaleConnectivityConfig(FrozenModel):
    """Tailscale connectivity settings.

    When present and ``enabled`` is ``True``, Skulk queries tailscaled at
    startup, logs the node's Tailscale IP and tailnet name, and merges
    ``bootstrap_peers`` into the libp2p peer list so nodes can discover each
    other across the Tailscale overlay network.

    Attributes:
        enabled: Master switch.  ``False`` disables all Tailscale-aware
            behaviour while preserving the config section for reference.
        bootstrap_peers: libp2p multiaddrs using Tailscale IPs, e.g.
            ``/ip4/100.101.102.103/tcp/52416``.  These are merged with any
            peers supplied via ``--bootstrap-peers`` or
            ``EXO_BOOTSTRAP_PEERS``.
    """

    enabled: bool = Field(default=True, description="Master switch for Tailscale-aware behaviour.")
    bootstrap_peers: list[str] = Field(
        default_factory=list,
        description="libp2p multiaddrs with Tailscale IPs, e.g. /ip4/100.x.x.x/tcp/52416.",
    )


@final
class ConnectivityConfig(FrozenModel):
    """Cluster connectivity settings.

    Attributes:
        tailscale: Tailscale overlay network settings.  ``None`` means the
            Tailscale integration is disabled.
    """

    tailscale: TailscaleConnectivityConfig | None = None


@final
class ExoConfig(FrozenModel):
    """Root configuration model for ``exo.yaml``.

    This is the top-level object parsed from the config file.  The design
    leaves room for future top-level sections without breaking existing
    deployments.

    Attributes:
        model_store: Model store configuration.  ``None`` when the section is
            absent, which means the model store feature is disabled.
        inference: Inference settings (KV cache backend).  ``None`` uses
            defaults.
        logging: Centralized logging configuration (enabled toggle, ingest
            URL).  ``None`` disables remote log shipping.
        connectivity: Cluster connectivity settings.  ``None`` means all
            connectivity options use their defaults (mDNS + CLI bootstrap peers
            only).
        hf_token: HuggingFace API token.  Stripped from ``GET /config``
            responses for security.
    """

    model_store: ModelStoreConfig | None = None
    inference: "InferenceConfig | None" = None
    logging: "LoggingConfig | None" = None
    tracing: "TracingConfig | None" = None
    connectivity: ConnectivityConfig | None = None
    hf_token: str | None = None


@final
class LoggingConfig(FrozenModel):
    """Central log aggregation configuration.

    When ``enabled`` is ``True`` and ``ingest_url`` is set, exo emits
    structured JSON on stdout (one object per line) alongside the
    human-readable stderr output.  A local log shipper such as Vector
    reads stdout and forwards to the ingest endpoint.

    All fields are synced to all nodes via gossipsub.

    Attributes:
        enabled: Master switch for centralized logging.  Nodes only emit
            structured stdout when this is ``True`` and ``ingest_url`` is
            set.
        ingest_url: Full VictoriaLogs (or compatible) ingest URL, e.g.
            ``http://192.168.0.118:9428/insert/jsonline?_stream_fields=node_id,component&_msg_field=msg&_time_field=ts``.
    """

    enabled: bool = False
    ingest_url: str = ""


@final
class TracingConfig(FrozenModel):
    """Runtime tracing configuration.

    Saved Chrome-trace JSON files accumulate under
    ``SKULK_TRACING_CACHE_DIR`` indefinitely otherwise; the janitor task
    in the API process drops files older than ``retention_days``.

    Attributes:
        retention_days: Saved traces older than this number of days are
            deleted by the API's hourly trace janitor. Defaults to 3 — short
            enough to keep disk use bounded, long enough to cover a weekend
            of "did this happen yesterday?" debugging. Set to 0 to disable
            pruning entirely.
    """

    retention_days: int = 3


@final
class InferenceConfig(FrozenModel):
    """Inference-related configuration.

    Attributes:
        kv_cache_backend: KV cache backend to use.
    """

    kv_cache_backend: Literal[
        "default", "mlx_quantized", "turboquant", "turboquant_adaptive", "optiq"
    ] = "default"


def resolve_config_path() -> Path:
    """Find the config file, preferring ``skulk.yaml`` over legacy ``exo.yaml``."""
    skulk = Path("skulk.yaml")
    exo = Path("exo.yaml")
    if skulk.exists():
        return skulk
    if exo.exists():
        return exo
    # Neither exists — return the preferred name so callers get a clear path
    return skulk


def load_exo_config(
    path: Path | None = None,
) -> ExoConfig | None:
    """Load cluster config from ``skulk.yaml`` (preferred) or ``exo.yaml`` (legacy fallback).

    Returns ``None`` if no config file exists, preserving zero-config
    compatibility: all downstream code must check for ``None`` before using
    the returned config and fall back to default behaviour.

    Args:
        path: Explicit path override.  When ``None`` (the default), the
              function checks for ``skulk.yaml`` first, then ``exo.yaml``
              in the current working directory.

    Returns:
        Parsed :class:`ExoConfig` instance, or ``None`` if the file is absent.

    Raises:
        :class:`pydantic.ValidationError`: If the file exists but contains
            invalid configuration.
        :class:`yaml.YAMLError`: If the file exists but is not valid YAML.
    """
    if path is None:
        path = resolve_config_path()
    if not path.exists():
        return None
    with path.open() as f:
        raw = yaml.safe_load(f)
    # An empty or comment-only file yields None from safe_load — treat
    # it the same as a missing file to preserve zero-config compatibility.
    if raw is None:
        return None
    return ExoConfig.model_validate(raw)


def resolve_node_staging(
    config: ModelStoreConfig,
    node_id: str,
) -> StagingNodeConfig:
    """Return the effective :class:`StagingNodeConfig` for a node.

    Resolution order (first match wins):

    1. ``node_overrides[<node_id>].staging`` — matched by libp2p peer ID.
    2. ``node_overrides[<hostname>].staging`` — matched by
       ``socket.gethostname()``.
    3. ``config.staging`` — the base (default) staging config.

    Args:
        config: The parsed ``model_store`` config section.
        node_id: The libp2p peer ID of this node (as a string).

    Returns:
        The :class:`StagingNodeConfig` that should be used on this node.
    """
    local_hostname_aliases = hostname_aliases(socket.gethostname())
    for key, override in config.node_overrides.items():
        key_matches_node = key == node_id
        key_matches_hostname = _normalize_hostname(key) in local_hostname_aliases
        if (
            override.staging is not None
            and (key_matches_node or key_matches_hostname)
        ):
            # Merge: start from the base config and overlay only the fields
            # that the override explicitly sets, so a partial override like
            # ``cleanup_on_deactivate: false`` inherits node_cache_path etc.
            # from the base rather than silently resetting to defaults.
            base = config.staging.model_dump()
            override_data = override.staging.model_dump(
                exclude_unset=True,
            )
            base.update(override_data)
            return StagingNodeConfig.model_validate(base)
    return config.staging
