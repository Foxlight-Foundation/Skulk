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
        cleanup_on_deactivate: true

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

from exo.utils.pydantic_ext import FrozenModel


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
            the model instance is shut down, freeing local disk space.
            Set to ``False`` on the store host so the canonical copy is
            never removed.
    """

    enabled: bool = True
    node_cache_path: str = "~/.exo/staging"
    cleanup_on_deactivate: bool = True


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
            this node is the store host).  Must match
            ``socket.gethostname()`` or the libp2p peer ID of the
            designated store node.
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
        hf_token: HuggingFace API token.  Stripped from ``GET /config``
            responses for security.
    """

    model_store: ModelStoreConfig | None = None
    inference: "InferenceConfig | None" = None
    logging: "LoggingConfig | None" = None
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
class InferenceConfig(FrozenModel):
    """Inference-related configuration.

    Attributes:
        kv_cache_backend: KV cache backend to use.
    """

    kv_cache_backend: Literal[
        "default",
        "mlx_quantized",
        "turboquant",
        "turboquant_adaptive",
        "optiq",
        "rotorquant",
        "rotorquant_adaptive",
    ] = "default"


def update_config_field(section: str, key: str, value: object) -> bool:
    """Update a single field in the config file, preserving all other content.

    Reads the raw YAML, patches ``raw[section][key] = value``, and writes
    it back.  Returns ``True`` if the file was updated, ``False`` if no
    config file exists.
    """
    path = resolve_config_path()
    if not path.exists():
        return False
    with path.open() as f:
        raw: dict[str, object] = yaml.safe_load(f) or {}
    sec = raw.get(section)
    if not isinstance(sec, dict):
        sec = {}
        raw[section] = sec
    sec[key] = value
    with path.open("w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    return True


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
    hostname = socket.gethostname()
    for key in (node_id, hostname):
        override = config.node_overrides.get(key)
        if override is not None and override.staging is not None:
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
