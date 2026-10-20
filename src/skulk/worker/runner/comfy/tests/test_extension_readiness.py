"""Trusted-node startup is isolated and failed readiness releases its process."""

from pathlib import Path

import httpx
import pytest

from skulk.worker.runner.comfy.server import ComfyServer, server_args


def test_extension_launch_keeps_all_unrelated_custom_and_api_nodes_disabled(tmp_path: Path) -> None:
    args = server_args(tmp_path / "python", tmp_path / "comfy", port=55000, extra_model_paths=tmp_path / "models.yaml", input_dir=tmp_path / "input", output_dir=tmp_path / "output", temp_dir=tmp_path / "temp", user_dir=tmp_path / "user", trusted_extension=tmp_path / "lib/skulk_comfy_ltx", extra=("--fast-disk",))
    assert args[:5] == [str(tmp_path / "python"), "-I", "-B", str(tmp_path / "lib/skulk_comfy_ltx/launcher.py"), str(tmp_path / "comfy")]
    assert "--disable-all-custom-nodes" in args and "--disable-api-nodes" in args
    assert args[args.index("--whitelist-custom-nodes") + 1:] == ["skulk_comfy_ltx", "--fast-disk"]


@pytest.mark.parametrize("error", [ValueError("bad schema"), httpx.ReadTimeout("schema timed out"), RuntimeError("missing node")])
def test_schema_failure_releases_server_before_the_runner_owns_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    server = ComfyServer(interpreter=tmp_path / "python", root=tmp_path / "comfy", extra_model_paths=tmp_path / "models.yaml", input_dir=tmp_path / "input", output_dir=tmp_path / "output", temp_dir=tmp_path / "temp", user_dir=tmp_path / "user", log_path=tmp_path / "log", trusted_extension=tmp_path / "lib/skulk_comfy_ltx")
    stopped: list[bool] = []

    def no_process() -> None:
        return None

    def bad_schema() -> None:
        raise error

    def teardown() -> None:
        stopped.append(True)

    monkeypatch.setattr(server, "_spawn", no_process)
    monkeypatch.setattr(server, "_await_health", no_process)
    monkeypatch.setattr(server, "_await_extension_schema", bad_schema)
    monkeypatch.setattr(server, "teardown", teardown)
    with pytest.raises(RuntimeError):
        server.start()
    assert stopped == [True]
