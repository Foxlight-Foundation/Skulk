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
from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import exo_shard_downloader
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import EXO_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.commands import ForwarderDownloadCommand, SyncConfig
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.store.config import ExoConfig, load_exo_config, resolve_node_staging
from exo.store.model_store import ModelStore
from exo.store.model_store_client import ModelStoreClient, ModelStoreDownloader
from exo.store.model_store_server import ModelStoreServer
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import CamelCaseModel
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker


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
        router = Router.create(keypair)
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        event_router = EventRouter(
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")

        # Load exo.yaml (returns None if absent — zero-config compatibility:
        # when exo.yaml is missing, all store references stay None and exo
        # behaves identically to the upstream default).
        exo_config = load_exo_config(Path("exo.yaml"))
        store_client: ModelStoreClient | None = None
        store_server: ModelStoreServer | None = None

        if (
            exo_config is not None
            and exo_config.model_store is not None
            and exo_config.model_store.enabled
        ):
            ms = exo_config.model_store
            local_hostname = socket.gethostname()
            is_store_host = ms.store_host in (str(node_id), local_hostname)

            # Store host gets a local path so the client uses shutil instead of HTTP.
            # Use store_http_host (when set) as the HTTP hostname so that a PeerId
            # in store_host does not get used as a DNS name by worker nodes.
            local_store_path: Path | None = (
                Path(ms.store_path) if is_store_host else None
            )
            store_client = ModelStoreClient(
                store_host=ms.store_http_host or ms.store_host,
                store_port=ms.store_port,
                local_store_path=local_store_path,
            )

            if is_store_host:
                model_store = ModelStore(Path(ms.store_path))
                store_server = ModelStoreServer(model_store, port=ms.store_port)
                logger.info(
                    f"ModelStore: this node is the store host — "
                    f"store at {ms.store_path}, server on port {ms.store_port}"
                )

            # Register the node-local staging directory in the model search
            # path so that build_model_path() / MLX finds staged models.
            # Set via environment variable so that runner subprocesses
            # (spawned via multiprocessing) also pick it up at import time.
            staging_cfg = resolve_node_staging(ms, str(node_id))
            staging_path = str(Path(staging_cfg.node_cache_path).expanduser())
            existing_path = os.environ.get("EXO_MODELS_PATH", "")
            paths = [p for p in existing_path.split(":") if p]
            if staging_path not in paths:
                paths.append(staging_path)
            os.environ["EXO_MODELS_PATH"] = ":".join(paths)
            # Also update the in-process constants for the main process
            from exo.shared.constants import add_model_search_path

            add_model_search_path(Path(staging_cfg.node_cache_path))
            logger.info(
                f"ModelStore: added staging path {staging_path} to EXO_MODELS_PATH"
            )

            # If this node is the store host, also add the store root to the
            # search path.  This lets the worker detect store-resident models
            # as "pre-downloaded" (via resolve_model_in_path), which skips
            # the redundant staging copy to the system disk.
            if is_store_host:
                store_root = str(Path(ms.store_path).expanduser())
                if store_root not in paths:
                    paths.append(store_root)
                    os.environ["EXO_MODELS_PATH"] = ":".join(paths)
                    add_model_search_path(Path(ms.store_path))
                    logger.info(
                        f"ModelStore: store host — added store root {store_root} to EXO_MODELS_PATH (skip staging)"
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
        is_store_host = ms.store_host in (str(self.node_id), local_hostname)
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

        # Also update local exo.yaml with the fixed host
        try:
            Path("exo.yaml").write_text(config_yaml)
        except Exception as exc:
            logger.warning(f"Failed to update local exo.yaml: {exc}")

        await self.router.sender(topics.DOWNLOAD_COMMANDS).send(
            ForwarderDownloadCommand(
                origin=SystemId(),
                command=SyncConfig(config_yaml=config_yaml),
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

                if result.is_new_master:
                    await anyio.sleep(0)
                    self.event_router.shutdown()
                    self.event_router = EventRouter(
                        result.session_id,
                        self.router.sender(topics.COMMANDS),
                        self.router.receiver(topics.GLOBAL_EVENTS),
                        self.router.sender(topics.LOCAL_EVENTS),
                    )
                    self._tg.start_soon(self.event_router.run)

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
                if result.is_new_master:
                    if self.download_coordinator:
                        self.download_coordinator.shutdown()
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
                        self.worker.shutdown()
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
                    if self.api:
                        self.api.reset(result.won_clock, self.event_router.receiver())
                    # Broadcast config to cluster so worker nodes get the right store address
                    await self._broadcast_config_if_store_host()
                else:
                    if self.api:
                        self.api.unpause(result.won_clock)


def main():
    args = Args.parse()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)
    # TODO: Refactor the current verbosity system
    logger_setup(EXO_LOG, args.verbosity)
    logger.info("Starting EXO")
    logger.info(f"EXO_LIBP2P_NAMESPACE: {os.getenv('EXO_LIBP2P_NAMESPACE')}")

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.no_batch:
        os.environ["EXO_NO_BATCH"] = "1"
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["EXO_FAST_SYNCH"] = "on"
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["EXO_FAST_SYNCH"] = "off"
        logger.info("FAST_SYNCH forced OFF")

    node = anyio.run(Node.create, args)
    try:
        anyio.run(node.run)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "EXO terminated due to unhandled exception"
        )
        raise
    finally:
        logger.info("EXO Shutdown complete")
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
