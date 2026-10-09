"""Planning installs an on-demand engine beside the download, then loads."""

from collections.abc import Set

import skulk.worker.plan as plan_mod
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import InstallEngine, LoadModel, Task
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgress,
    DownloadProgressData,
)
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerIdle
from skulk.worker.tests.constants import INSTANCE_1_ID, MODEL_A_ID, NODE_A, RUNNER_1_ID
from skulk.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

_ON_DEMAND = frozenset({"comfy", "comfy-cuda"})


def _plan(
    *,
    downloaded: bool,
    on_demand: Set[str] = _ON_DEMAND,
    requested: Set[InstanceId] = frozenset(),
) -> Task | None:
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0).model_copy(
        update={"resolved_backend": "comfy-cuda"}
    )
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    runner = FakeRunnerSupervisor(
        bound_instance=BoundInstance(
            instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
        ),
        status=RunnerIdle(),
    )
    download: DownloadProgress
    if downloaded:
        download = DownloadCompleted(shard_metadata=shard, node_id=NODE_A, total=Memory())
    else:
        download = DownloadOngoing(
            shard_metadata=shard,
            node_id=NODE_A,
            download_progress=DownloadProgressData(
                total=Memory(),
                downloaded=Memory(),
                downloaded_this_session=Memory(),
                completed_files=0,
                total_files=1,
                speed=0.0,
                eta_ms=0,
                files={},
            ),
        )
    return plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={NODE_A: [download]},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerIdle()},
        tasks={},
        on_demand_backends=on_demand,
        engine_install_requested=requested,
    )


def test_engine_install_starts_while_the_model_downloads() -> None:
    task = _plan(downloaded=False)
    assert isinstance(task, InstallEngine)
    assert task.instance_id == INSTANCE_1_ID and task.engine == "comfy"
    # The size rides on the task so the dashboard can quote it while it runs.
    assert task.approximate_download_bytes == 7 * 1024**3


def test_engine_install_is_requested_once_and_the_load_waits_for_it() -> None:
    assert _plan(downloaded=True, requested={INSTANCE_1_ID}) is None


def test_model_loads_once_its_engine_is_installed() -> None:
    task = _plan(downloaded=True, on_demand=frozenset(), requested={INSTANCE_1_ID})
    assert isinstance(task, LoadModel) and task.instance_id == INSTANCE_1_ID


def test_installed_engines_never_request_an_install() -> None:
    assert isinstance(_plan(downloaded=True, on_demand=frozenset()), LoadModel)
