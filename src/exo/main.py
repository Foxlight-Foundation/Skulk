import argparse
import multiprocessing as mp
import os
import resource
import signal
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import anyio
from loguru import logger
from pydantic import PositiveInt

import exo.routing.topics as topics
from exo.api.main import API
from exo.connectivity.tailscale import query_tailscale_status
from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import exo_shard_downloader
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import SKULK_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import (
    external_log_pipe_enabled,
    logger_cleanup,
    logger_setup,
)
from exo.shared.types.commands import ForwarderDownloadCommand, SyncConfig
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.state_sync import StateSyncMessage
from exo.startup_recovery import preflight_api_port
from exo.store.config import (
    ExoConfig,
    load_exo_config,
    node_matches_store_host,
    resolve_config_path,
    resolve_node_staging,
)
from exo.store.model_store import ModelStore
from exo.store.model_store_client import ModelStoreClient, ModelStoreDownloader
from exo.store.model_store_server import ModelStoreServer
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import CamelCaseModel
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker


def _add_model_search_path(path: Path) -> None:
    """Ensure the given model path is visible to the current process and children."""

    expanded = path.expanduser()
    existing_path = os.environ.get("SKULK_MODELS_PATH", os.environ.get("EXO_MODELS_PATH", ""))
    paths = [p for p in existing_path.split(":") if p]
    path_str = str(expanded)
    if path_str not in paths:
        paths.append(path_str)
    joined = ":".join(paths)
    os.environ["SKULK_MODELS_PATH"] = joined
    os.environ["EXO_MODELS_PATH"] = joined  # legacy compat

    from exo.shared.constants import add_model_search_path

    add_model_search_path(expanded)


def _configure_model_store_runtime(
    node_id: NodeId,
    exo_config: ExoConfig | None,
) -> tuple[ModelStoreClient | None, ModelStoreServer | None]:
    """Build store client/server wiring from the current config."""

    if (
        exo_config is None
        or exo_config.model_store is None
        or not exo_config.model_store.enabled
    ):
        return None, None

    ms = exo_config.model_store
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
    exo_config: ExoConfig | None
    store_client: ModelStoreClient | None
    store_server: ModelStoreServer | None
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_node_id())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(
            keypair,
            bootstrap_peers=args.bootstrap_peers,
            listen_port=args.libp2p_port,
        )
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        await router.register_topic(topics.STATE_SYNC_MESSAGES)
        event_router = EventRouter(
            node_id,
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            state_sync_sender=router.sender(topics.STATE_SYNC_MESSAGES),
            state_sync_receiver=router.receiver_with_origin(
                topics.STATE_SYNC_MESSAGES
            ),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")

        # Load exo.yaml (returns None if absent — zero-config compatibility:
        # when exo.yaml is missing, all store references stay None and exo
        # behaves identically to the upstream default).
        exo_config = load_exo_config()

        # Track whether user provided the KV backend env var at launch —
        # if so, config syncs must not overwrite it.
        _user_set_kv_backend = (
            "SKULK_KV_CACHE_BACKEND" in os.environ
            or "EXO_KV_CACHE_BACKEND" in os.environ
        )
        os.environ["_SKULK_KV_BACKEND_USER_SET"] = "1" if _user_set_kv_backend else ""
        os.environ["_EXO_KV_BACKEND_USER_SET"] = (
            "1" if _user_set_kv_backend else ""
        )  # legacy compat

        # Apply inference config to env var so runner subprocesses inherit it.
        # Env var takes precedence if user set it at launch.
        if (
            exo_config is not None
            and exo_config.inference is not None
            and not _user_set_kv_backend
        ):
            os.environ["SKULK_KV_CACHE_BACKEND"] = exo_config.inference.kv_cache_backend
            os.environ["EXO_KV_CACHE_BACKEND"] = (
                exo_config.inference.kv_cache_backend
            )  # legacy compat
            logger.info(
                f"Inference config: kv_cache_backend={exo_config.inference.kv_cache_backend}"
            )

        # Apply HF token from config if not already set via env
        if (
            exo_config is not None
            and exo_config.hf_token
            and "HF_TOKEN" not in os.environ
        ):
            os.environ["HF_TOKEN"] = exo_config.hf_token
            logger.info("HF token loaded from config")

        store_client, store_server = _configure_model_store_runtime(
            node_id, exo_config
        )

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            base_downloader = exo_shard_downloader(offline=args.offline)
            if (
                exo_config is not None
                and exo_config.model_store is not None
                and exo_config.model_store.enabled
                and store_client is not None
            ):
                ms = exo_config.model_store
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
                        exo_config.model_store, str(node_id)
                    ).node_cache_path
                )
                if exo_config is not None
                and exo_config.model_store is not None
                and exo_config.model_store.enabled
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
                exo_config=exo_config,
                store_client=store_client,
            )
        else:
            api = None

        if not args.no_worker:
            worker_store_client: ModelStoreClient | None = store_client
            if (
                exo_config is not None
                and exo_config.model_store is not None
                and exo_config.model_store.enabled
            ):
                worker_staging_cfg = resolve_node_staging(
                    exo_config.model_store, str(node_id)
                )
            else:
                worker_staging_cfg = None
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
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
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into an exo cluster then. well done. good job champ.
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
            exo_config,
            store_client,
            store_server,
        )

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self.election.run)
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
        self.exo_config = load_exo_config(config_path)

    async def _broadcast_config_if_store_host(self) -> None:
        """If this node is the store host, broadcast a valid config to all nodes.

        Fixes up ``store_http_host`` so that worker nodes receive a reachable
        address (the store host's hostname) rather than ``127.0.0.1`` or None.
        """
        if self.exo_config is None or self.exo_config.model_store is None:
            return
        ms = self.exo_config.model_store
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

        # Fix up store_http_host to be reachable by other nodes
        reachable_host = local_hostname
        if ms.store_http_host and ms.store_http_host not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            reachable_host = ms.store_http_host

        config_dict = self.exo_config.model_dump()
        config_dict["model_store"]["store_http_host"] = reachable_host

        import yaml

        config_yaml = yaml.safe_dump(
            config_dict, default_flow_style=False, sort_keys=False
        )

        # Also update local config file with the fixed host
        try:
            resolve_config_path().write_text(config_yaml)
        except Exception as exc:
            logger.warning(f"Failed to update local config: {exc}")

        # Strip secrets before broadcasting over gossipsub
        import copy

        broadcast_dict = copy.deepcopy(config_dict)
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
                        state_sync_sender=self.router.sender(topics.STATE_SYNC_MESSAGES),
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
                    self.master = Master(
                        self.node_id,
                        result.session_id,
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
                            _configure_model_store_runtime(self.node_id, self.exo_config)
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
                        base_dl = exo_shard_downloader(offline=self.offline)
                        ms = (
                            self.exo_config.model_store
                            if self.exo_config is not None
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
                            self.exo_config.model_store
                            if self.exo_config is not None
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
    _early_config: ExoConfig | None = None
    try:
        _early_config = load_exo_config()
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
    logger.info(f"LIBP2P_NAMESPACE: {os.environ.get('SKULK_LIBP2P_NAMESPACE', os.getenv('EXO_LIBP2P_NAMESPACE'))}")

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
        if _ts_config.bootstrap_peers:
            _seen = set(args.bootstrap_peers)
            _extra = [p for p in _ts_config.bootstrap_peers if p not in _seen]
            if _extra:
                args = args.model_copy(
                    update={"bootstrap_peers": args.bootstrap_peers + _extra}
                )
                logger.info(
                    f"Tailscale: added {len(_extra)} bootstrap peer(s) from config"
                )

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        logger.info(f"Bootstrap peers: {args.bootstrap_peers}")

    if args.no_batch:
        os.environ["SKULK_NO_BATCH"] = "1"
        os.environ["EXO_NO_BATCH"] = "1"  # legacy compat
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["SKULK_FAST_SYNCH"] = "on"
        os.environ["EXO_FAST_SYNCH"] = "on"  # legacy compat
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["SKULK_FAST_SYNCH"] = "off"
        os.environ["EXO_FAST_SYNCH"] = "off"  # legacy compat
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
    offline: bool = os.getenv("EXO_OFFLINE", "false").lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="EXO")
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
            default=os.getenv("EXO_OFFLINE", "false").lower() == "true",
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
            default=os.getenv("EXO_BOOTSTRAP_PEERS", "").split(",")
            if os.getenv("EXO_BOOTSTRAP_PEERS")
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--libp2p-port",
            type=int,
            default=0,
            dest="libp2p_port",
            help="Fixed TCP port for libp2p to listen on (0 = OS-assigned).",
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
