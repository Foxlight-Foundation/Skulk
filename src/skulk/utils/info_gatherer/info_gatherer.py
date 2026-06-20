import os
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    fail_after,
    open_process,
    to_thread,
)
from anyio.streams.buffered import BufferedByteReceiveStream
from loguru import logger
from pydantic import ValidationError

from skulk.shared.constants import SKULK_CONFIG_FILE, SKULK_MODELS_DIR
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    DiskUsage,
    MachMemoryCategories,
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeResources,
    SystemPerformanceProfile,
    ThunderboltBridgeStatus,
    parse_vm_stat_output,
)
from skulk.shared.types.thunderbolt import (
    ThunderboltConnection,
    ThunderboltConnectivity,
    ThunderboltIdentifier,
)
from skulk.shared.version import get_skulk_version
from skulk.utils.channels import Sender
from skulk.utils.pydantic_ext import TaggedModel
from skulk.utils.task_group import TaskGroup

from .linux_gpu import (
    LinuxGpuMetrics,
    find_amd_gpu_device,
    read_accelerator_metrics,
)
from .mactop import MacmonMetrics, MactopMetrics
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    get_os_build_version,
    get_os_version,
)

IS_DARWIN = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


_vm_stat_failure_logged = False


async def _read_mach_memory_categories() -> MachMemoryCategories | None:
    """One ``vm_stat`` snapshot of Mach page categories, or ``None`` on failure.

    Runs once per mactop sample (1 s cadence); ``vm_stat`` is a single
    ``host_statistics64`` call, a few milliseconds. Failures (binary missing,
    non-zero exit, format drift) degrade to ``None`` so telemetry falls back to
    mactop's raw availability instead of flapping — logged once, not at the
    sample cadence.
    """
    global _vm_stat_failure_logged
    try:
        result = await anyio.run_process(["vm_stat"], check=False)
    except OSError as error:
        if not _vm_stat_failure_logged:
            _vm_stat_failure_logged = True
            logger.warning(
                f"vm_stat unavailable; placement availability falls back to "
                f"mactop's cache-deflated figure: {error}"
            )
        return None
    if result.returncode != 0:
        if not _vm_stat_failure_logged:
            _vm_stat_failure_logged = True
            logger.warning(
                f"vm_stat exited {result.returncode}; placement availability "
                f"falls back to mactop's cache-deflated figure"
            )
        return None
    categories = parse_vm_stat_output(result.stdout.decode("utf-8", errors="replace"))
    if categories is None and not _vm_stat_failure_logged:
        _vm_stat_failure_logged = True
        logger.warning(
            "vm_stat output did not match the expected format; placement "
            "availability falls back to mactop's cache-deflated figure"
        )
    return categories


async def _get_thunderbolt_devices() -> set[str] | None:
    """Get Thunderbolt interface device names (e.g., en2, en3) from hardware ports.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listallhardwareports"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listallhardwareports failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    output = result.stdout.decode()
    thunderbolt_devices: set[str] = set()
    current_port: str | None = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            if "thunderbolt" in current_port.lower():
                thunderbolt_devices.add(device)
            current_port = None

    return thunderbolt_devices


async def _get_bridge_services() -> dict[str, str] | None:
    """Get mapping of bridge device -> service name from network service order.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listnetworkserviceorder"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listnetworkserviceorder failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    # Parse service order to find bridge devices and their service names
    # Format: "(1) Service Name\n(Hardware Port: ..., Device: bridge0)\n"
    service_order_output = result.stdout.decode()
    bridge_services: dict[str, str] = {}  # device -> service name
    current_service: str | None = None

    for line in service_order_output.splitlines():
        line = line.strip()
        # Match "(N) Service Name" or "(*) Service Name" (disabled)
        # but NOT "(Hardware Port: ...)" lines
        if (
            line
            and line.startswith("(")
            and ")" in line
            and not line.startswith("(Hardware Port:")
        ):
            paren_end = line.index(")")
            if paren_end + 2 <= len(line):
                current_service = line[paren_end + 2 :]
        # Match "(Hardware Port: ..., Device: bridgeX)"
        elif current_service and "Device: bridge" in line:
            # Extract device name from "..., Device: bridge0)"
            device_start = line.find("Device: ") + len("Device: ")
            device_end = line.find(")", device_start)
            if device_end > device_start:
                device = line[device_start:device_end]
                bridge_services[device] = current_service

    return bridge_services


async def _get_bridge_members(bridge_device: str) -> set[str]:
    """Get member interfaces of a bridge device via ifconfig."""
    result = await anyio.run_process(
        ["ifconfig", bridge_device],
        check=False,
    )
    if result.returncode != 0:
        logger.debug(f"ifconfig {bridge_device} failed with code {result.returncode}")
        return set()

    members: set[str] = set()
    ifconfig_output = result.stdout.decode()
    for line in ifconfig_output.splitlines():
        line = line.strip()
        if line.startswith("member:"):
            parts = line.split()
            if len(parts) > 1:
                members.add(parts[1])

    return members


async def _find_thunderbolt_bridge(
    bridge_services: dict[str, str], thunderbolt_devices: set[str]
) -> str | None:
    """Find the service name of a bridge containing Thunderbolt interfaces.

    Returns the service name if found, None otherwise.
    """
    for bridge_device, service_name in bridge_services.items():
        members = await _get_bridge_members(bridge_device)
        if members & thunderbolt_devices:  # intersection is non-empty
            return service_name
    return None


async def _is_service_enabled(service_name: str) -> bool | None:
    """Check if a network service is enabled.

    Returns True if enabled, False if disabled, None on error.
    """
    result = await anyio.run_process(
        ["networksetup", "-getnetworkserviceenabled", service_name],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -getnetworkserviceenabled '{service_name}' "
            f"failed with code {result.returncode}: {result.stderr.decode()}"
        )
        return None

    stdout = result.stdout.decode().strip().lower()
    return stdout == "enabled"


class StaticNodeInformation(TaggedModel):
    """Node information that should NEVER change, to be gathered once at startup"""

    model: str
    chip: str
    os_version: str
    os_build_version: str
    skulk_version: str
    skulk_commit: str

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        return cls(
            model=model,
            chip=chip,
            os_version=get_os_version(),
            os_build_version=await get_os_build_version(),
            skulk_version=_get_exo_version(),
            skulk_commit=_get_git_commit(),
        )


def _get_exo_version() -> str:
    """Get the Skulk app version from shared package metadata."""
    return get_skulk_version()


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class MacThunderboltIdentifiers(TaggedModel):
    idents: Sequence[ThunderboltIdentifier]


class MacThunderboltConnections(TaggedModel):
    conns: Sequence[ThunderboltConnection]


class RdmaCtlStatus(TaggedModel):
    enabled: bool
    interfaces_present: bool

    @classmethod
    async def gather(cls) -> Self | None:
        if not IS_DARWIN or shutil.which("rdma_ctl") is None:
            return None
        try:
            with anyio.fail_after(5):
                proc = await anyio.run_process(["rdma_ctl", "status"], check=False)
        except (TimeoutError, OSError):
            return None
        if proc.returncode != 0:
            return None
        output = proc.stdout.decode("utf-8").lower().strip()
        if "enabled" in output:
            return cls(enabled=True, interfaces_present=_rdma_interfaces_exist())
        if "disabled" in output:
            return cls(enabled=False, interfaces_present=False)
        return None


def _rdma_interfaces_exist() -> bool:
    """Check if any rdma_* network interfaces actually exist at the OS level.

    On TB4 hardware, rdma_ctl may report enabled but no rdma_* interfaces
    are created because the hardware doesn't support RDMA.
    """
    import socket

    try:
        return any(name.startswith("rdma_") for _, name in socket.if_nameindex())
    except OSError:
        return False


class ThunderboltBridgeInfo(TaggedModel):
    status: ThunderboltBridgeStatus

    @classmethod
    async def gather(cls) -> Self | None:
        """Check if a Thunderbolt Bridge network service is enabled on this node.

        Detection approach:
        1. Find all Thunderbolt interface devices (en2, en3, etc.) from hardware ports
        2. Find bridge devices from network service order (not hardware ports, as
           bridges may not appear there)
        3. Check each bridge's members via ifconfig
        4. If a bridge contains Thunderbolt interfaces, it's a Thunderbolt Bridge
        5. Check if that network service is enabled
        """
        if not IS_DARWIN:
            return None

        def _no_bridge_status() -> Self:
            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=False, exists=False, service_name=None
                )
            )

        try:
            tb_devices = await _get_thunderbolt_devices()
            if tb_devices is None:
                return _no_bridge_status()

            bridge_services = await _get_bridge_services()
            if not bridge_services:
                return _no_bridge_status()

            tb_service_name = await _find_thunderbolt_bridge(
                bridge_services, tb_devices
            )
            if not tb_service_name:
                return _no_bridge_status()

            enabled = await _is_service_enabled(tb_service_name)
            if enabled is None:
                return cls(
                    status=ThunderboltBridgeStatus(
                        enabled=False, exists=True, service_name=tb_service_name
                    )
                )

            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=enabled,
                    exists=True,
                    service_name=tb_service_name,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to gather Thunderbolt Bridge info: {e}")
            return None


class NodeConfig(TaggedModel):
    """Node configuration from SKULK_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(SKULK_CONFIG_FILE)
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError):
                logger.warning("Invalid config file, skipping...")
                return None


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


class NodeDiskUsage(TaggedModel):
    """Disk space information for the models directory."""

    disk_usage: DiskUsage

    @classmethod
    async def gather(cls) -> Self:
        return cls(
            disk_usage=await to_thread.run_sync(
                lambda: DiskUsage.from_path(SKULK_MODELS_DIR)
            )
        )


async def _gather_iface_map() -> dict[str, str] | None:
    proc = await anyio.run_process(
        ["networksetup", "-listallhardwareports"], check=False
    )
    if proc.returncode != 0:
        return None

    ports: dict[str, str] = {}
    port = ""
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.startswith("Hardware Port:"):
            port = line.split(": ")[1]
        elif line.startswith("Device:"):
            ports[port] = line.split(": ")[1]
            port = ""
    if "" in ports:
        del ports[""]
    return ports


GatheredInfo = (
    MactopMetrics
    # Decode-only: lets a newly-upgraded node still apply telemetry from macOS
    # workers on a pre-mactop build during a rolling upgrade (see MacmonMetrics).
    | MacmonMetrics
    | LinuxGpuMetrics
    | MemoryUsage
    | NodeNetworkInterfaces
    | MacThunderboltIdentifiers
    | MacThunderboltConnections
    | RdmaCtlStatus
    | ThunderboltBridgeInfo
    | NodeConfig
    | MiscData
    | StaticNodeInformation
    | NodeResources
    | NodeDiskUsage
)


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    interface_watcher_interval: float | None = 10
    misc_poll_interval: float | None = 60
    system_profiler_interval: float | None = 5 if IS_DARWIN else None
    memory_poll_rate: float | None = None if IS_DARWIN else 1
    mactop_interval: float | None = 1 if IS_DARWIN else None
    thunderbolt_bridge_poll_interval: float | None = 10 if IS_DARWIN else None
    static_info_poll_interval: float | None = 60
    node_resources_poll_interval: float | None = 60
    rdma_ctl_poll_interval: float | None = 10 if IS_DARWIN else None
    disk_poll_interval: float | None = 30
    gpu_linux_poll_interval: float | None = 2 if IS_LINUX else None
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    async def run(self):
        try:
            async with self._tg as tg:
                if IS_DARWIN:
                    if (mactop_path := shutil.which("mactop")) is not None:
                        tg.start_soon(self._monitor_mactop, mactop_path)
                    else:
                        # mactop not installed — fall back to psutil for memory.
                        # (mactop replaced macmon, whose IOGPUFamily GPU polling
                        # crashed/hung MLX inference — see mactop.py / exo#2088.)
                        logger.warning(
                            "mactop not found, falling back to psutil for "
                            "memory monitoring"
                        )
                        self.memory_poll_rate = 1
                    tg.start_soon(self._monitor_system_profiler_thunderbolt_data)
                    tg.start_soon(self._monitor_thunderbolt_bridge_status)
                    tg.start_soon(self._monitor_rdma_ctl_status)
                if IS_LINUX:
                    tg.start_soon(self._monitor_gpu_linux)
                tg.start_soon(self._watch_system_info)
                tg.start_soon(self._monitor_memory_usage)
                tg.start_soon(self._monitor_misc)
                tg.start_soon(self._monitor_static_info)
                tg.start_soon(self._monitor_node_resources)
                tg.start_soon(self._monitor_disk_usage)

                nc = await NodeConfig.gather()
                if nc is not None:
                    await self.info_sender.send(nc)
        except BaseExceptionGroup as exception_group:
            # A closed/broken info channel means our consumer is gone — the
            # worker is shutting down or being replaced on a master
            # transition (main.py tears down the old Worker and builds a
            # fresh one with a fresh gatherer). That is a stop signal, not a
            # fault: crashing here took down whole healthy nodes when a PEER
            # restarted (#266 — the consumer's forwarder exits first when
            # the event stream closes, and the next telemetry send raced
            # into the closed channel). anyio folds body and child-task
            # exceptions into one group, so this covers every send site.
            # Any non-channel exception still propagates.
            consumer_gone, other = exception_group.split(
                (ClosedResourceError, BrokenResourceError)
            )
            if consumer_gone is None:
                # Nothing channel-related in the group — re-raise it exactly
                # as it arrived (a `from` link here would point the group's
                # __cause__ at itself).
                raise
            if other is not None:
                raise other from exception_group
            logger.info(
                "InfoGatherer stopped: telemetry consumer closed its channel "
                "(worker shutdown/replacement); the replacement worker's "
                "gatherer owns telemetry from here"
            )

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _monitor_static_info(self):
        if self.static_info_poll_interval is None:
            return
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await StaticNodeInformation.gather())
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering static node info: {e}")
            await anyio.sleep(self.static_info_poll_interval)

    async def _monitor_node_resources(self):
        if self.node_resources_poll_interval is None:
            return
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await NodeResources.gather())
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone: stop signal, not a fault. Escape the
                # per-iteration catch-all so the loop cannot spin on a dead
                # channel (#266); run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering node resources: {e}")
            await anyio.sleep(self.node_resources_poll_interval)

    async def _monitor_misc(self):
        if self.misc_poll_interval is None:
            return
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(await MiscData.gather())
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering misc data: {e}")
            await anyio.sleep(self.misc_poll_interval)

    async def _monitor_system_profiler_thunderbolt_data(self):
        if self.system_profiler_interval is None:
            return

        while True:
            try:
                with fail_after(30):
                    iface_map = await _gather_iface_map()
                    if iface_map is None:
                        raise ValueError("Failed to gather interface map")

                    data = await ThunderboltConnectivity.gather()
                    assert data is not None

                    idents = [
                        it for i in data if (it := i.ident(iface_map)) is not None
                    ]

                    # Filter to only interfaces that actually exist at the OS level.
                    # On TB4 hardware, system_profiler reports rdma_* identifiers
                    # but the interfaces are never created — RDMA requires TB5.
                    if idents and not _rdma_interfaces_exist():
                        logger.debug(
                            "Thunderbolt: rdma_ctl reports RDMA but no rdma_* interfaces "
                            "exist (TB4 hardware?) — suppressing RDMA identifiers"
                        )
                        idents = []

                    await self.info_sender.send(
                        MacThunderboltIdentifiers(idents=idents)
                    )

                    # Only emit connections if we have valid identifiers
                    conns = (
                        [it for i in data if (it := i.conn()) is not None]
                        if idents
                        else []
                    )
                    await self.info_sender.send(MacThunderboltConnections(conns=conns))
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering Thunderbolt data: {e}")
            await anyio.sleep(self.system_profiler_interval)

    async def _monitor_memory_usage(self):
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        if self.memory_poll_rate is None:
            return
        while True:
            try:
                await self.info_sender.send(
                    MemoryUsage.from_psutil(override_memory=override_memory)
                )
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering memory usage: {e}")
            await anyio.sleep(self.memory_poll_rate)

    async def _watch_system_info(self):
        if self.interface_watcher_interval is None:
            return
        while True:
            try:
                with fail_after(10):
                    nics = await get_network_interfaces()
                    await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering network interfaces: {e}")
            await anyio.sleep(self.interface_watcher_interval)

    async def _monitor_thunderbolt_bridge_status(self):
        if self.thunderbolt_bridge_poll_interval is None:
            return
        while True:
            try:
                with fail_after(30):
                    curr = await ThunderboltBridgeInfo.gather()
                    if curr is not None:
                        await self.info_sender.send(curr)
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering Thunderbolt Bridge status: {e}")
            await anyio.sleep(self.thunderbolt_bridge_poll_interval)

    async def _monitor_rdma_ctl_status(self):
        if self.rdma_ctl_poll_interval is None:
            return
        while True:
            try:
                curr = await RdmaCtlStatus.gather()
                if curr is not None:
                    await self.info_sender.send(curr)
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering RDMA ctl status: {e}")
            await anyio.sleep(self.rdma_ctl_poll_interval)

    async def _monitor_disk_usage(self):
        if self.disk_poll_interval is None:
            return
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(await NodeDiskUsage.gather())
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                logger.warning(f"Error gathering disk usage: {e}")
            await anyio.sleep(self.disk_poll_interval)

    async def _monitor_gpu_linux(self):
        """Publish normalized AMD/Linux GPU telemetry from passive sysfs reads.

        Resolves the amdgpu device once; if none is present (no GPU, or a driver
        that does not expose ``gpu_busy_percent``) the node simply reports no
        accelerator, which the dashboard renders as "not reported". Reads are
        passive sysfs, never a GPU-colliding poll (see ``linux_gpu.py`` and the
        macmon crash mechanism it avoids).
        """
        if self.gpu_linux_poll_interval is None:
            return
        device = find_amd_gpu_device()
        if device is None:
            logger.info(
                "no AMD GPU sysfs device (gpu_busy_percent) found; skipping GPU "
                "telemetry on this node"
            )
            return
        logger.info(f"reporting GPU telemetry from {device}")
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(
                        LinuxGpuMetrics(
                            system_profile=SystemPerformanceProfile(
                                accelerator=read_accelerator_metrics(device)
                            )
                        )
                    )
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault; must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                raise
            except Exception as e:
                logger.warning(f"Error gathering Linux GPU telemetry: {e}")
            await anyio.sleep(self.gpu_linux_poll_interval)

    async def _monitor_mactop(self, mactop_path: str):
        if self.mactop_interval is None:
            return
        # `mactop --headless --format json --interval <ms> --count 0` streams one
        # JSON object per line, forever. mactop reads Apple's IOReport/SMC
        # counters (NOT the IOGPUFamily command-buffer interface that macmon
        # used), so it is safe to sample concurrently with active MLX inference
        # — see mactop.py for the macmon crash mechanism it avoids.
        # Timeout: if mactop produces no output for this long, restart it
        # (10x the interval is generous; mactop writes every interval seconds).
        read_timeout = max(self.mactop_interval * 10, 30)
        while True:
            try:
                async with await open_process(
                    [
                        mactop_path,
                        "--headless",
                        "--format",
                        "json",
                        "--interval",
                        str(int(self.mactop_interval * 1000)),
                        "--count",
                        "0",
                    ]
                ) as p:
                    if not p.stdout:
                        logger.critical("mactop closed stdout")
                        return
                    stream = BufferedByteReceiveStream(p.stdout)
                    while True:
                        # Only the read is timeout-guarded; parsing/sending are
                        # not I/O against mactop.
                        with fail_after(read_timeout):
                            # 1 MiB/line: receive_until raises if the newline is
                            # not found within max_bytes, which the outer handler
                            # turns into a respawn — so the cap must clear the
                            # largest plausible sample. mactop's optional arrays
                            # (top-N processes, per-volume disk, Thunderbolt, net)
                            # can push one JSON line well past tens of KiB; 1 MiB
                            # is far above any real sample while still bounding a
                            # pathological/never-terminating stream.
                            data = await stream.receive_until(
                                delimiter=b"\n", max_bytes=1024 * 1024
                            )
                        text = data.decode("utf-8", errors="replace").strip()
                        # A blank or partial line (e.g. mactop startup/shutdown)
                        # must not tear down the whole subprocess — skip it and
                        # keep reading so telemetry doesn't flap.
                        if not text:
                            continue
                        # A vm_stat snapshot taken alongside the mactop sample
                        # corrects ram_available to the GPU-wireable figure —
                        # mactop's own "available" counts reclaimable file
                        # cache as used (see mactop.py). None on failure keeps
                        # the raw mactop figure rather than dropping the sample.
                        mach_categories = await _read_mach_memory_categories()
                        try:
                            metrics = MactopMetrics.from_raw_json(text, mach_categories)
                        except ValidationError as e:
                            logger.warning(f"Skipping unparseable mactop line: {e}")
                            continue
                        await self.info_sender.send(metrics)
            except TimeoutError:
                logger.warning(
                    f"mactop produced no output for {read_timeout}s, restarting"
                )
            except (ClosedResourceError, BrokenResourceError):
                # Consumer gone (worker shutdown/replacement): a stop signal,
                # not a gathering fault — must escape this per-iteration
                # catch-all or the loop spins on a dead channel (#266).
                # run() converts it into a clean stop.
                raise
            except Exception as e:
                # anyio's open_process()/async-with does not raise
                # CalledProcessError (no check=True semantics); mactop dying just
                # closes stdout, surfacing here as EndOfStream. Respawn either way.
                logger.warning(f"Error in mactop monitor: {e}")
            await anyio.sleep(self.mactop_interval)
