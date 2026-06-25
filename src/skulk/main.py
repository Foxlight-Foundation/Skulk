import argparse
import hashlib
import ipaddress
import multiprocessing as mp
import os
import resource
import signal
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

import anyio
import psutil
from loguru import logger
from pydantic import PositiveInt

import skulk.routing.topics as topics
from skulk.api.main import API
from skulk.connectivity.local_network import (
    check_local_network_access,
    local_network_denied_message,
)
from skulk.connectivity.tailscale import query_tailscale_status
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.impl_shard_downloader import skulk_shard_downloader
from skulk.master.main import Master
from skulk.routing.event_router import EventRouter
from skulk.routing.router import Router, get_node_id_keypair
from skulk.shared.constants import SKULK_LOG
from skulk.shared.election import Election, ElectionResult
from skulk.shared.logging import (
    external_log_pipe_enabled,
    logger_cleanup,
    logger_setup,
)
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import ForwarderDownloadCommand, SyncConfig
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.startup_recovery import preflight_api_port
from skulk.store.config import (
    SkulkConfig,
    load_skulk_config,
    node_matches_store_host,
    resolve_config_path,
    resolve_node_staging,
)
from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader
from skulk.store.model_store_server import ModelStoreServer
from skulk.utils.channels import Receiver, channel
from skulk.utils.pydantic_ext import CamelCaseModel
from skulk.utils.task_group import TaskGroup
from skulk.worker.main import Worker


def _derive_zenoh_namespace(raw: str) -> str:
    """Map a libp2p namespace to a Zenoh key-expr namespace segment (#308).

    This is the Zenoh data-plane isolation boundary, so distinct libp2p
    namespaces must not collide on the same Zenoh namespace, or peers on
    different libp2p namespaces could read each other's ``data``. We SHA-256-hash
    unconditionally rather than a verbatim/hash split: a char-replacement
    sanitizer collapses ``prod/main`` and ``prod_main`` (#312 review P1), and a
    verbatim-when-safe split still lets a fleet named literally
    ``ns<sha256(victim)>`` collide with the victim's hashed namespace (#312 review
    P2). A SHA-256 hex digest is collision-resistant (no two distinct namespaces
    collide in practice) and always a valid key-expr segment; the ``ns`` prefix
    keeps it from starting with a digit. The trade-off is a non-human-readable
    namespace, which is fine for an internal key prefix. Note: neither this
    derived namespace nor the raw libp2p token is ever logged (with no TLS the
    namespace is itself the isolation value); startup logs only a non-routing
    fingerprint of it (see ``_namespace_fingerprint``).
    """
    return "ns" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Keep in sync with rust/networking/src/swarm.rs: NETWORK_VERSION default and
# the OVERRIDE_VERSION_ENV_VAR name used to build the libp2p private-network key.
_LIBP2P_NETWORK_VERSION = "v0.0.1"
_LIBP2P_NAMESPACE_ENV_VAR = "SKULK_LIBP2P_NAMESPACE"


def _libp2p_namespace_token(environ: Mapping[str, str]) -> str:
    """Return the exact token libp2p isolates on, for the Zenoh namespace (#312).

    The Zenoh namespace MUST derive from the identical token that builds the
    libp2p private-network key in ``swarm.rs`` (``PNET_PRESHARED_KEY``); otherwise
    two nodes in the same libp2p cluster can land in different Zenoh namespaces
    and silently drop all cross-node generation output. ``swarm.rs`` uses
    ``SKULK_LIBP2P_NAMESPACE`` when the var is *present* (Rust ``env::var``
    returns ``Ok`` even for an empty value) and the ``NETWORK_VERSION`` default
    (``v0.0.1``) otherwise.
    We mirror that precisely: presence (not truthiness) selects the override, and
    an unset var falls back to ``v0.0.1`` rather than a Skulk-only default.
    """
    override = environ.get(_LIBP2P_NAMESPACE_ENV_VAR)
    if override is not None:
        return override
    return _LIBP2P_NETWORK_VERSION


def _namespace_fingerprint(namespace: str) -> str:
    """Return a short non-routing fingerprint of a Zenoh namespace (#312 review).

    With no transport auth/TLS the namespace prefix is itself the isolation
    value: a peer that learns it can subscribe to the fully prefixed key and read
    ``data``. So startup logging emits this fingerprint instead of the namespace.
    It is a truncated second hash, so it cannot be used to subscribe and cannot be
    reversed to the namespace, yet it is stable per namespace, which is all an
    operator needs to confirm two nodes resolved to the same isolation segment.
    """
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]


def _require_zenoh_listen(env_value: str) -> str:
    """Return the explicit Zenoh listen endpoint, or raise (#308 bind restriction).

    When the Zenoh data plane is enabled, the listen endpoint must be set
    explicitly; Skulk refuses to default it to ``tcp/0.0.0.0:7447`` (all
    interfaces) so a shared-network deployment doesn't silently expose the plane.
    """
    listen = env_value.strip()
    if not listen:
        raise ValueError(
            "SKULK_ZENOH_DATA_PLANE is enabled but SKULK_ZENOH_LISTEN is unset. "
            "Set it explicitly (e.g. tcp/<this-node-ip>:7447); Skulk refuses to "
            "default the Zenoh listen endpoint to 0.0.0.0 / all interfaces "
            "(#308 bind restriction)."
        )
    return listen


def _resolve_zenoh_enabled(data_plane_env: str, listen_env: str) -> bool:
    """Resolve whether the Zenoh DATA plane is enabled (soft default-on, #315).

    The DATA plane defaults to Zenoh when a node is configured for it, but never
    crashes a bare node:

    - Explicit truthy (``1``/``true``/``yes``/``on``) -> enabled. The caller
      still requires an explicit listen endpoint via :func:`_require_zenoh_listen`,
      so an explicit opt-in with no listen is a loud error, not a silent default.
    - Explicit falsy (``0``/``false``/``no``/``off``) -> disabled (gossipsub).
    - Unset/blank -> soft default: enabled only when ``SKULK_ZENOH_LISTEN`` is
      configured. A node with no Zenoh config at all (e.g. a fresh ``uv run
      skulk``) stays on gossipsub instead of failing the #308 listen
      requirement, so the listen endpoint is the opt-in signal under the default.
    - Any other non-empty value -> ``ValueError``. An unrecognized value
      (a typo, or a boolean spelling we don't accept) must NOT silently fall
      through to the listen-based default, or an operator who wrote
      ``SKULK_ZENOH_DATA_PLANE=disable`` could unexpectedly get Zenoh ON; we
      refuse to guess the transport (#315 review).

    ``listen_env`` is the raw ``SKULK_ZENOH_LISTEN`` value; only its presence
    (after stripping) matters here.
    """
    value = data_plane_env.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    if value:
        raise ValueError(
            f"SKULK_ZENOH_DATA_PLANE={data_plane_env!r} is not a recognized "
            "boolean. Use 1/true/yes/on or 0/false/no/off, or leave it unset to "
            "use the default (Zenoh when SKULK_ZENOH_LISTEN is set, else "
            "gossipsub). Refusing to guess the DATA transport (#315)."
        )
    return bool(listen_env.strip())


def _add_model_search_path(path: Path) -> None:
    """Ensure the given model path is visible to the current process and children."""

    expanded = path.expanduser()
    existing_path = os.environ.get(
        "SKULK_MODELS_PATH", os.environ.get("SKULK_MODELS_PATH", "")
    )
    paths = [p for p in existing_path.split(":") if p]
    path_str = str(expanded)
    if path_str not in paths:
        paths.append(path_str)
    joined = ":".join(paths)
    os.environ["SKULK_MODELS_PATH"] = joined
    os.environ["SKULK_MODELS_PATH"] = joined  # legacy compat

    from skulk.shared.constants import add_model_search_path

    add_model_search_path(expanded)


_VIRTUAL_IFACE_PREFIXES: Final = (
    "docker",
    "br-",
    "virbr",
    "vmnet",
    "vboxnet",
    "veth",
    "cni",
    "flannel",
    "kube",
)


def _is_virtual_iface(name: str) -> bool:
    """Whether an interface name looks like a Docker/VM/container bridge.

    These carry RFC1918 addresses (e.g. Docker's ``172.17.0.1``) that are not
    reachable from peers on the real LAN, so they must not be advertised as the
    store host. VPN tunnels (Tailscale ``utun``/``tailscale0``) are deliberately
    NOT excluded: they are a valid fallback path and are already ranked below the
    LAN address.
    """
    lowered = name.lower()
    return lowered.startswith(_VIRTUAL_IFACE_PREFIXES)


def _routable_store_advertise_host(configured: str | None, hostname_fallback: str) -> str:
    """Pick an address other nodes can actually reach the model store host at.

    The store host broadcasts this as ``store_http_host`` so workers build the
    download URL ``http://<host>:<port>``. A bare hostname (``kite3.local``) is
    fragile on a Thunderbolt-meshed fleet: mDNS can resolve it to the host's
    link-local TB address (``169.254.x``), which a peer lacking a direct TB link
    cannot route to, so its downloads fail while the LAN path works fine.

    An operator-supplied **routable IP literal** is honored as-is. Anything else
    (a hostname, or a loopback/link-local literal) is replaced with this host's
    own best routable IPv4: a private LAN address (RFC1918) is preferred over any
    other routable address, and loopback / link-local / unspecified addresses are
    skipped. Falls back to the hostname only when no routable IPv4 is found.
    """
    if configured:
        try:
            literal = ipaddress.ip_address(configured)
        except ValueError:
            literal = None  # a hostname, not an IP -> recompute below
        # Honor only a routable IPv4 literal: the store URL is built as
        # http://{host}:{port} with no IPv6-bracket handling, so an IPv6 literal
        # would produce an invalid URL -> treat it like a hostname and recompute.
        if (
            literal is not None
            and literal.version == 4
            and not (
                literal.is_loopback or literal.is_link_local or literal.is_unspecified
            )
        ):
            return configured

    routable: list[str] = []
    for iface_name, addresses in psutil.net_if_addrs().items():
        # Skip virtual bridge / container interfaces (docker0, br-*, virbr*,
        # vmnet*, vboxnet*, veth*, k8s): they carry RFC1918 IPs that peers on the
        # real LAN cannot route to and could otherwise outrank the LAN address.
        if _is_virtual_iface(iface_name):
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(address.address)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
                continue
            routable.append(address.address)

    # Prefer an RFC1918 LAN address (fast, reachable across the local switch)
    # over a Tailscale/CGNAT (100.64.0.0/10) address over anything else; all beat
    # the hostname fallback. CGNAT is also "private", so rank the ranges
    # explicitly rather than relying on ``is_private``.
    lan_nets = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    cgnat_net = ipaddress.ip_network("100.64.0.0/10")

    def _rank(address: str) -> int:
        ip = ipaddress.ip_address(address)
        if any(ip in net for net in lan_nets):
            return 0
        if ip in cgnat_net:
            return 1
        return 2

    routable.sort(key=_rank)
    return routable[0] if routable else hostname_fallback


def _configure_model_store_runtime(
    node_id: NodeId,
    skulk_config: SkulkConfig | None,
) -> tuple[ModelStoreClient | None, ModelStoreServer | None]:
    """Build store client/server wiring from the current config."""

    if (
        skulk_config is None
        or skulk_config.model_store is None
        or not skulk_config.model_store.enabled
    ):
        return None, None

    ms = skulk_config.model_store
    is_store_host = node_matches_store_host(
        ms.store_host,
        str(node_id),
        hostname=socket.gethostname(),
    )

    local_store_path: Path | None = Path(ms.store_path) if is_store_host else None
    store_client = ModelStoreClient(
        store_host=ms.store_http_host or ms.store_host,
        store_port=ms.store_port,
        local_store_path=local_store_path,
    )

    store_server: ModelStoreServer | None = None
    if is_store_host:
        model_store = ModelStore(Path(ms.store_path))
        store_server = ModelStoreServer(model_store, port=ms.store_port)
        logger.info(
            f"ModelStore: this node is the store host — "
            f"store at {ms.store_path}, server on port {ms.store_port}"
        )

    staging_cfg = resolve_node_staging(ms, str(node_id))
    staging_path = Path(staging_cfg.node_cache_path)
    _add_model_search_path(staging_path)
    logger.info(
        f"ModelStore: added staging path {staging_path.expanduser()} to SKULK_MODELS_PATH"
    )

    if is_store_host:
        store_root = Path(ms.store_path)
        _add_model_search_path(store_root)
        logger.info(
            f"ModelStore: store host — added store root {store_root.expanduser()} to SKULK_MODELS_PATH (skip staging)"
        )

    return store_client, store_server


@dataclass
class Node:
    router: Router
    event_router: EventRouter
    download_coordinator: DownloadCoordinator | None
    worker: Worker | None
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    offline: bool
    skulk_config: SkulkConfig | None
    store_client: ModelStoreClient | None
    store_server: ModelStoreServer | None
    # Live node telemetry off the event log (#279). Node-owned so it survives
    # master re-election; the subscriber feeds it, the master/API read it.
    telemetry_view: TelemetryView
    telemetry_receiver: Receiver[NodeTelemetry]
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_node_id())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        # Zenoh data plane (#279 follow-on). The DATA topic (per-token output)
        # rides a Zenoh peer session instead of gossipsub; all other planes stay
        # on libp2p. Endpoints are per-node (multicast off), so they come from
        # the environment, not the gossip-synced config. Soft default-on (#315):
        # Zenoh is the default WHEN configured (SKULK_ZENOH_LISTEN set) and a bare
        # node with no Zenoh config falls back to gossipsub rather than crashing
        # on the #308 listen requirement; SKULK_ZENOH_DATA_PLANE=1/0 forces it
        # on/off explicitly. See _resolve_zenoh_enabled.
        _zenoh_data_plane_env = os.environ.get("SKULK_ZENOH_DATA_PLANE", "")
        _zenoh_listen_env = os.environ.get("SKULK_ZENOH_LISTEN", "")
        _zenoh_on = _resolve_zenoh_enabled(_zenoh_data_plane_env, _zenoh_listen_env)
        if not _zenoh_on and not _zenoh_data_plane_env.strip():
            # Default path on a node with no listen configured: say why we're on
            # gossipsub so the operator knows how to opt into Zenoh.
            logger.info(
                "DATA plane on gossipsub (default): SKULK_ZENOH_LISTEN unset. "
                "Set it to use the Zenoh data plane."
            )
        _zenoh_connect = [
            endpoint.strip()
            for endpoint in os.environ.get("SKULK_ZENOH_CONNECT", "").split(",")
            if endpoint.strip()
        ]
        _zenoh_listen_endpoints: list[str] | None = None
        _zenoh_namespace: str | None = None
        if _zenoh_on:
            # Bind restriction (#308): require SKULK_ZENOH_LISTEN explicitly
            # rather than silently defaulting to tcp/0.0.0.0:7447 (all
            # interfaces). The operator picks the interface (a private IP on a
            # shared network); an explicit 0.0.0.0 is still allowed but is then
            # a deliberate choice, not a silent default.
            _zenoh_listen = _require_zenoh_listen(
                os.environ.get("SKULK_ZENOH_LISTEN", "")
            )
            _zenoh_listen_endpoints = [_zenoh_listen]
            # Namespace isolation (#308): Zenoh transparently prefixes all keys
            # with this segment, so a peer on a different namespace cannot read
            # this fleet's `data`. Derive it from the EXACT token libp2p isolates
            # on (_libp2p_namespace_token mirrors swarm.rs), via a
            # collision-resistant SHA-256 hash (see _derive_zenoh_namespace). If
            # the source diverged from libp2p (legacy env, different default),
            # two nodes in one libp2p cluster could land in different Zenoh
            # namespaces and silently drop all cross-node output (#312 review).
            # We never log the raw token or the derived namespace: the raw token
            # seeds libp2p's private-network PSK (swarm.rs PNET_PRESHARED_KEY), and
            # because the plane has no transport auth/TLS the derived namespace IS
            # the only isolation value (a peer that learns it can subscribe to the
            # prefixed key and read `data`). Log only a non-routing fingerprint and
            # whether an override was set, so operators can still confirm two nodes
            # share a namespace without exposing it (#312 review).
            _ns_raw = _libp2p_namespace_token(os.environ)
            _zenoh_namespace = _derive_zenoh_namespace(_ns_raw)
            _ns_override_set = _LIBP2P_NAMESPACE_ENV_VAR in os.environ
            if "0.0.0.0" in _zenoh_listen:
                logger.warning(
                    f"SKULK_ZENOH_LISTEN={_zenoh_listen} binds all interfaces; "
                    f"prefer a specific private IP on a shared network (#308)."
                )
            logger.warning(
                f"Zenoh DATA plane ENABLED: generation "
                f"output is served over Zenoh on {_zenoh_listen}, namespace"
                f"-isolated (fingerprint {_namespace_fingerprint(_zenoh_namespace)}; "
                f"{_LIBP2P_NAMESPACE_ENV_VAR} "
                f"{'set' if _ns_override_set else 'unset, using default'}). There "
                f"is still NO transport auth/TLS, so on an untrusted network "
                f"enable Zenoh TLS or keep it firewalled (#308)."
            )
        router = Router.create(
            keypair,
            bootstrap_peers=args.bootstrap_peers,
            listen_port=args.libp2p_port,
            zenoh_listen_endpoints=_zenoh_listen_endpoints,
            zenoh_connect_endpoints=_zenoh_connect,
            node_id=str(node_id),
            zenoh_namespace=_zenoh_namespace,
        )
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        await router.register_topic(topics.STATE_SYNC_MESSAGES)
        await router.register_topic(topics.TELEMETRY)
        await router.register_topic(topics.DATA)
        telemetry_view = TelemetryView()
        event_router = EventRouter(
            node_id,
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            state_sync_sender=router.sender(topics.STATE_SYNC_MESSAGES),
            state_sync_receiver=router.receiver_with_origin(topics.STATE_SYNC_MESSAGES),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")

        # Load skulk.yaml (returns None if absent, for zero-config compatibility:
        # when skulk.yaml is missing, all store references stay None and the
        # node behaves identically to the zero-config default).
        skulk_config = load_skulk_config()

        # Track whether user provided the KV backend env var at launch —
        # if so, config syncs must not overwrite it.
        _user_set_kv_backend = "SKULK_KV_CACHE_BACKEND" in os.environ
        os.environ["_SKULK_KV_BACKEND_USER_SET"] = "1" if _user_set_kv_backend else ""

        # Apply inference config to env var so runner subprocesses inherit it.
        # Env var takes precedence if user set it at launch.
        if (
            skulk_config is not None
            and skulk_config.inference is not None
            and not _user_set_kv_backend
        ):
            os.environ["SKULK_KV_CACHE_BACKEND"] = skulk_config.inference.kv_cache_backend
            logger.info(
                f"Inference config: kv_cache_backend={skulk_config.inference.kv_cache_backend}"
            )

        # Apply HF token from config if not already set via env
        if (
            skulk_config is not None
            and skulk_config.hf_token
            and "HF_TOKEN" not in os.environ
        ):
            os.environ["HF_TOKEN"] = skulk_config.hf_token
            logger.info("HF token loaded from config")

        store_client, store_server = _configure_model_store_runtime(node_id, skulk_config)

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            base_downloader = skulk_shard_downloader(offline=args.offline)
            if (
                skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
                and store_client is not None
            ):
                ms = skulk_config.model_store
                staging_cfg = resolve_node_staging(ms, str(node_id))
                shard_downloader = ModelStoreDownloader(
                    inner=base_downloader,
                    store_client=store_client,
                    staging_config=staging_cfg,
                    allow_hf_fallback=ms.download.allow_hf_fallback,
                )
            else:
                shard_downloader = base_downloader

            coordinator_staging_path = (
                Path(
                    resolve_node_staging(
                        skulk_config.model_store, str(node_id)
                    ).node_cache_path
                )
                if skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
                else None
            )
            download_coordinator = DownloadCoordinator(
                node_id,
                shard_downloader,
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
                staging_cache_path=coordinator_staging_path,
            )
        else:
            download_coordinator = None

        if args.spawn_api:
            api = API(
                node_id,
                port=args.api_port,
                event_receiver=event_router.receiver(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
                skulk_config=skulk_config,
                store_client=store_client,
                telemetry_view=telemetry_view,
                data_receiver=router.receiver(topics.DATA),
                data_plane_zenoh=_zenoh_on,
            )
        else:
            api = None

        if not args.no_worker:
            worker_store_client: ModelStoreClient | None = store_client
            if (
                skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
            ):
                worker_staging_cfg = resolve_node_staging(
                    skulk_config.model_store, str(node_id)
                )
            else:
                worker_staging_cfg = None
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                telemetry_sender=router.sender(topics.TELEMETRY),
                telemetry_view=telemetry_view,
                data_sender=router.sender(topics.DATA),
                store_client=worker_store_client,
                staging_config=worker_staging_cfg,
            )
            if api is not None:
                api.set_runner_diagnostics_provider(worker.collect_runner_diagnostics)
                api.set_runner_cancel_provider(worker.cancel_runner_task)
        else:
            worker = None

        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            event_sender=event_router.sender(),
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            state_sync_receiver=router.receiver(topics.STATE_SYNC_MESSAGES),
            state_sync_sender=router.sender(topics.STATE_SYNC_MESSAGES),
            download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
            telemetry_view=telemetry_view,
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into a Skulk cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
        )

        return cls(
            router,
            event_router,
            download_coordinator,
            worker,
            election,
            er_recv,
            master,
            api,
            node_id,
            args.offline,
            skulk_config,
            store_client,
            store_server,
            telemetry_view,
            router.receiver(topics.TELEMETRY),
        )

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self.election.run)
            tg.start_soon(self._run_telemetry)
            if self.store_server:
                tg.start_soon(self.store_server.start)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    async def _run_telemetry(self) -> None:
        """Maintain the node-owned TelemetryView from the telemetry plane (#279).

        Runs for the node's lifetime, independent of master election, so the
        view of every node's resources persists across a master flip. Each
        message coalesces last-write-wins; there is no ordering or persistence.
        """
        with self.telemetry_receiver as messages:
            async for message in messages:
                self.telemetry_view.apply(message)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_called():
            import sys

            sys.exit(1)
        self._tg.cancel_tasks()

    async def _request_cluster_config(self, session_id: SessionId) -> str | None:
        """Request the authoritative cluster config from the current master."""

        requester = SystemId()
        state_sync_sender = self.router.sender(topics.STATE_SYNC_MESSAGES)
        state_sync_receiver = self.router.receiver_with_origin(
            topics.STATE_SYNC_MESSAGES
        )
        with state_sync_receiver as messages:
            for attempt in range(3):
                await state_sync_sender.send(
                    StateSyncMessage(
                        kind="request",
                        requester=requester,
                        session_id=session_id,
                    )
                )
                with anyio.move_on_after(1.0):
                    async for origin, message in messages:
                        if message.kind != "response":
                            continue
                        if message.requester != requester:
                            continue
                        if message.session_id != session_id:
                            continue
                        if origin != str(session_id.master_node_id):
                            continue
                        return message.config_yaml
                if attempt < 2:
                    await anyio.sleep(0.2)
        return None

    def _apply_cluster_config_yaml(self, config_yaml: str) -> None:
        """Persist cluster config locally and rebuild derived runtime wiring."""

        config_path = resolve_config_path()
        config_path.write_text(config_yaml)
        self.skulk_config = load_skulk_config(config_path)

    async def _broadcast_config_if_store_host(self) -> None:
        """If this node is the store host, broadcast a valid config to all nodes.

        Resolves ``store_http_host`` to a routable IPv4 (see
        ``_routable_store_advertise_host``) so worker nodes receive an address
        they can actually reach over HTTP, rather than a hostname that may
        mDNS-resolve to an unreachable link-local address, ``127.0.0.1``, or
        None. An operator-supplied routable IPv4 literal is honored as-is;
        otherwise this host's best routable LAN address is used.

        The resolved address is broadcast over the cluster config-sync path,
        which every node (including this store host, via local delivery) applies
        and persists. We therefore do not separately write the local config file
        here: a second write would only be clobbered by the host applying its
        own broadcast.
        """
        if self.skulk_config is None or self.skulk_config.model_store is None:
            return
        ms = self.skulk_config.model_store
        if not ms.enabled:
            return
        local_hostname = socket.gethostname()
        is_store_host = node_matches_store_host(
            ms.store_host,
            str(self.node_id),
            hostname=local_hostname,
        )
        if not is_store_host:
            return

        # Advertise a routable IP, not a hostname. A bare hostname (e.g.
        # ``kite3.local``) can mDNS-resolve on a Thunderbolt-meshed fleet to the
        # store host's link-local TB address (169.254.x), which peers without a
        # direct TB link cannot route to, so their store downloads fail even
        # though they can reach the host fine over the LAN.
        reachable_host = _routable_store_advertise_host(
            ms.store_http_host, local_hostname
        )

        import copy

        import yaml

        # Broadcast the resolved reachable host to the cluster (secrets stripped).
        # The store host applies its own broadcast via local delivery and persists
        # it through the normal config-sync path, so there is no separate local
        # write here (it would only be clobbered by that same broadcast).
        broadcast_dict = copy.deepcopy(self.skulk_config.model_dump())
        broadcast_dict["model_store"]["store_http_host"] = reachable_host
        broadcast_dict.pop("hf_token", None)
        broadcast_yaml = yaml.safe_dump(
            broadcast_dict, default_flow_style=False, sort_keys=False
        )

        await self.router.sender(topics.DOWNLOAD_COMMANDS).send(
            ForwarderDownloadCommand(
                origin=SystemId(),
                command=SyncConfig(config_yaml=broadcast_yaml),
            )
        )
        logger.info(
            f"ModelStore: broadcast config to cluster (store_http_host={reachable_host})"
        )

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

                start_replacement_event_router = False
                previous_store_server = self.store_server
                if result.is_new_master:
                    await anyio.sleep(0)
                    self.event_router.shutdown()
                    self.event_router = EventRouter(
                        self.node_id,
                        result.session_id,
                        command_sender=self.router.sender(topics.COMMANDS),
                        state_sync_sender=self.router.sender(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        state_sync_receiver=self.router.receiver_with_origin(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        external_inbound=self.router.receiver(topics.GLOBAL_EVENTS),
                        external_outbound=self.router.sender(topics.LOCAL_EVENTS),
                    )
                    # Wait to bootstrap the replacement event router until the
                    # replacement worker/API receivers are attached. Otherwise,
                    # a fast snapshot hydrate can be emitted before those
                    # consumers exist, and the next live event will arrive out
                    # of sequence against blank local state.
                    start_replacement_event_router = True
                    if previous_store_server is None and self.store_server is not None:
                        self._tg.start_soon(self.store_server.start)

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    # Seed the new session from this node's replicated view
                    # (captured before the worker below is torn down and
                    # re-created): placements survive master failover (#273)
                    # instead of every worker reconciling its healthy runners
                    # away against an empty snapshot. apply() replaces the
                    # worker's state wholesale (immutable convention), so the
                    # reference read here is a consistent snapshot.
                    prior_state = self.worker.state if self.worker is not None else None
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        initial_state=(
                            seed_state_for_new_session(prior_state)
                            if prior_state is not None
                            else None
                        ),
                        event_sender=self.event_router.sender(),
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                        state_sync_receiver=self.router.receiver(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        state_sync_sender=self.router.sender(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                        telemetry_view=self.telemetry_view,
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if (
                    result.is_new_master
                    and result.session_id.master_node_id != self.node_id
                ):
                    authoritative_config_yaml = await self._request_cluster_config(
                        result.session_id
                    )
                    if authoritative_config_yaml is not None:
                        self._apply_cluster_config_yaml(authoritative_config_yaml)
                        new_store_client, new_store_server = (
                            _configure_model_store_runtime(
                                self.node_id, self.skulk_config
                            )
                        )
                        self.store_client = new_store_client
                        self.store_server = (
                            previous_store_server
                            if previous_store_server is not None
                            else new_store_server
                        )
                if result.is_new_master:
                    if self.download_coordinator:
                        await self.download_coordinator.shutdown()
                        base_dl = skulk_shard_downloader(offline=self.offline)
                        ms = (
                            self.skulk_config.model_store
                            if self.skulk_config is not None
                            else None
                        )
                        if (
                            ms is not None
                            and ms.enabled
                            and self.store_client is not None
                        ):
                            elect_staging = resolve_node_staging(ms, str(self.node_id))
                            elect_downloader = ModelStoreDownloader(
                                inner=base_dl,
                                store_client=self.store_client,
                                staging_config=elect_staging,
                                allow_hf_fallback=ms.download.allow_hf_fallback,
                            )
                        else:
                            elect_downloader = base_dl
                        elect_staging_path = (
                            Path(
                                resolve_node_staging(
                                    ms, str(self.node_id)
                                ).node_cache_path
                            )
                            if ms is not None and ms.enabled
                            else None
                        )
                        self.download_coordinator = DownloadCoordinator(
                            self.node_id,
                            elect_downloader,
                            event_sender=self.event_router.sender(),
                            download_command_receiver=self.router.receiver(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            offline=self.offline,
                            staging_cache_path=elect_staging_path,
                        )
                        self._tg.start_soon(self.download_coordinator.run)
                    if self.worker:
                        await self.worker.shutdown()
                        ms2 = (
                            self.skulk_config.model_store
                            if self.skulk_config is not None
                            else None
                        )
                        elect_staging2 = (
                            resolve_node_staging(ms2, str(self.node_id))
                            if ms2 is not None and ms2.enabled
                            else None
                        )
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            event_receiver=self.event_router.receiver(),
                            event_sender=self.event_router.sender(),
                            command_sender=self.router.sender(topics.COMMANDS),
                            download_command_sender=self.router.sender(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            store_client=self.store_client,
                            staging_config=elect_staging2,
                            # Must match Node.create's Worker wiring: without this
                            # the recreated worker stops publishing NodeResources
                            # telemetry, and after a master restart (fresh
                            # telemetry_view) the node never reappears in
                            # node_resources, so placement silently treats a
                            # management/edge node as eligible (#279 review).
                            telemetry_sender=self.router.sender(topics.TELEMETRY),
                            telemetry_view=self.telemetry_view,
                            # Must ALSO match Node.create's wiring: without this
                            # the recreated worker has no data sender, so every
                            # generation output chunk falls back to the event
                            # plane — which the API no longer routes (#279 Phase
                            # 2a) — and every completion stream hangs forever.
                            data_sender=self.router.sender(topics.DATA),
                        )
                        self._tg.start_soon(self.worker.run)
                        if self.api is not None:
                            self.api.set_runner_diagnostics_provider(
                                self.worker.collect_runner_diagnostics
                            )
                            self.api.set_runner_cancel_provider(
                                self.worker.cancel_runner_task
                            )
                    if self.api:
                        self.api.reset(
                            result.won_clock,
                            self.event_router.receiver(),
                            result.session_id.master_node_id,
                        )
                    if start_replacement_event_router:
                        self._tg.start_soon(self.event_router.run)
                    # Broadcast config to cluster so worker nodes get the right store address
                    await self._broadcast_config_if_store_host()
                else:
                    if self.api:
                        self.api.unpause(
                            result.won_clock,
                            master_node_id=result.session_id.master_node_id,
                        )


def main():
    args = Args.parse()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)

    # Load config early so the logging section is available before anything
    # else runs.  The full config is loaded again inside Node.create() for
    # the model store and inference sections.  If the file is malformed we
    # fall back gracefully — logging will start without the JSON sink and
    # the validation error is logged once the logger is up.
    _log_cfg = None
    _early_config: SkulkConfig | None = None
    try:
        _early_config = load_skulk_config()
        _log_cfg = _early_config.logging if _early_config else None
    except Exception:
        pass  # Logged after logger_setup below

    # External-shipper mode (SKULK_LOGGING_EXTERNAL=1, set by the
    # launchd / systemd wrapper when an external Vector agent is
    # installed) implies "structured logging on at boot" without
    # requiring an `enabled: true` in skulk.yaml — the env var is the
    # operator's signal that they have a shipper hooked up. The
    # dashboard / config sync still controls the sink at runtime via
    # set_structured_stdout, so an operator can disable shipping live.
    _structured = external_log_pipe_enabled() or bool(_log_cfg and _log_cfg.enabled)
    logger_setup(
        SKULK_LOG,
        args.verbosity,
        structured_stdout=_structured,
        ingest_url=_log_cfg.ingest_url if _log_cfg else "",
    )
    logger.info("Starting Skulk")
    # The libp2p namespace token seeds the private-network PSK (swarm.rs) and, when
    # the Zenoh data plane is on, the no-TLS Zenoh namespace too; logging its value
    # would let anyone with log access compute the namespace and subscribe to
    # `data` (#312 review). Log only whether it is set and a non-routing
    # fingerprint, which is enough to confirm two nodes share a namespace.
    _libp2p_ns = os.environ.get(_LIBP2P_NAMESPACE_ENV_VAR)
    if _libp2p_ns is not None:
        logger.info(
            f"{_LIBP2P_NAMESPACE_ENV_VAR} set (fingerprint "
            f"{_namespace_fingerprint(_libp2p_ns)})"
        )
    else:
        logger.info(f"{_LIBP2P_NAMESPACE_ENV_VAR} unset, using default")

    if args.spawn_api:
        preflight_api_port(args.api_port)

    # Tailscale: if configured, query tailscaled and merge bootstrap peers.
    _ts_config = (
        _early_config.connectivity.tailscale
        if _early_config and _early_config.connectivity
        else None
    )
    if _ts_config and _ts_config.enabled:
        import asyncio as _asyncio

        _ts_status = _asyncio.run(query_tailscale_status())
        if _ts_status.running:
            logger.info(
                f"Tailscale: running | IP {_ts_status.self_ip} | {_ts_status.dns_name}"
            )
        else:
            logger.warning(
                "Tailscale connectivity configured but tailscaled is not running"
            )
        # Auto-discover tailnet peers as bootstrap addresses so a Tailscale
        # cluster needs no hand-maintained IP list — just `enabled: true`. Each
        # peer is dialed on this node's libp2p port; non-Skulk tailnet peers
        # fail the private-network handshake and are harmlessly ignored.
        #
        # Auto-discovery needs a fixed, known port to build the peer multiaddrs.
        # With --libp2p-port 0 (OS-assigned) the listen port differs per node and
        # is unknown to peers, so an auto-built /tcp/0 address could never be
        # dialed. Skip auto-discovery in that case and tell the operator how to
        # make it work, rather than silently producing dead /tcp/0 peers.
        if args.libp2p_port == 0:
            _auto_peers: list[str] = []
            if _ts_status.peer_ips:
                logger.warning(
                    "Tailscale auto-discovery is enabled but --libp2p-port is 0 "
                    "(OS-assigned); auto-discovered peers need a fixed port and were "
                    "skipped. Set --libp2p-port / SKULK_LIBP2P_PORT (default 52416) or "
                    "list connectivity.tailscale.bootstrap_peers explicitly."
                )
        else:
            _auto_peers = [
                f"/ip4/{ip}/tcp/{args.libp2p_port}" for ip in _ts_status.peer_ips
            ]
        # Merge auto-discovered + config-listed peers, de-duplicating against
        # CLI/existing peers and each other while preserving order.
        _seen = set(args.bootstrap_peers)
        _extra: list[str] = []
        for _peer in _auto_peers + list(_ts_config.bootstrap_peers):
            if _peer not in _seen:
                _seen.add(_peer)
                _extra.append(_peer)
        if _extra:
            args = args.model_copy(
                update={"bootstrap_peers": args.bootstrap_peers + _extra}
            )
            logger.info(
                f"Tailscale: added {len(_extra)} bootstrap peer(s) "
                f"({len(_auto_peers)} auto-discovered, "
                f"{len(_ts_config.bootstrap_peers)} from config)"
            )

    # macOS Local Network Privacy: a denied process silently fails to reach LAN
    # / Thunderbolt peers (EHOSTUNREACH), so cluster discovery never forms.
    # Detect it early and tell the operator how to grant access.
    if check_local_network_access() == "blocked":
        logger.warning(local_network_denied_message())

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        logger.info(f"Bootstrap peers: {args.bootstrap_peers}")

    if args.no_batch:
        os.environ["SKULK_NO_BATCH"] = "1"
        os.environ["SKULK_NO_BATCH"] = "1"  # legacy compat
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["SKULK_FAST_SYNCH"] = "on"
        os.environ["SKULK_FAST_SYNCH"] = "on"  # legacy compat
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["SKULK_FAST_SYNCH"] = "off"
        os.environ["SKULK_FAST_SYNCH"] = "off"  # legacy compat
        logger.info("FAST_SYNCH forced OFF")

    node = anyio.run(Node.create, args)
    try:
        anyio.run(node.run)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "Skulk terminated due to unhandled exception"
        )
        raise
    finally:
        logger.info("Skulk shutdown complete")
        logger_cleanup()


class Args(CamelCaseModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    no_worker: bool = False
    no_downloads: bool = False
    offline: bool = os.getenv("SKULK_OFFLINE", "false").lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="skulk")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--no-worker",
            action="store_true",
        )
        parser.add_argument(
            "--no-downloads",
            action="store_true",
            help="Disable the download coordinator (node won't download models)",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            default=os.getenv("SKULK_OFFLINE", "false").lower() == "true",
            help="Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models",
        )
        parser.add_argument(
            "--no-batch",
            action="store_true",
            help="Disable continuous batching, use sequential generation",
        )
        parser.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p for p in s.split(",") if p],
            default=os.getenv("SKULK_BOOTSTRAP_PEERS", "").split(",")
            if os.getenv("SKULK_BOOTSTRAP_PEERS")
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: SKULK_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--libp2p-port",
            type=int,
            # Default to a fixed, well-known port rather than an OS-assigned one
            # so that bootstrap-peer multiaddrs (Tailscale, cross-subnet) have a
            # predictable port to dial — a user can write
            # /ip4/<peer>/tcp/52416 without first inspecting each node's random
            # port. mDNS discovery advertises the real port either way, so this
            # is harmless on a single local network. Pass 0 for OS-assigned.
            default=int(os.getenv("SKULK_LIBP2P_PORT", "52416")),
            dest="libp2p_port",
            help="Fixed TCP port for libp2p to listen on (default 52416; 0 = OS-assigned; env: SKULK_LIBP2P_PORT).",
        )
        fast_synch_group = parser.add_mutually_exclusive_group()
        fast_synch_group.add_argument(
            "--fast-synch",
            action="store_true",
            dest="fast_synch",
            default=None,
            help="Force MLX FAST_SYNCH on (for JACCL backend)",
        )
        fast_synch_group.add_argument(
            "--no-fast-synch",
            action="store_false",
            dest="fast_synch",
            help="Force MLX FAST_SYNCH off",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically
