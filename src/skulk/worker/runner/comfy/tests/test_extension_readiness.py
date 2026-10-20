# pyright: reportPrivateUsage=false
"""Trusted-node startup is isolated and failed readiness releases its process."""

from collections.abc import Iterator
from contextlib import contextmanager
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


@pytest.mark.parametrize("temporal,dfr", [(False, False), (True, False), (True, True)])
def test_readiness_checks_only_nodes_implemented_by_the_issued_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temporal: bool, dfr: bool) -> None:
    """An old sampler-only wheel remains valid; temporal wheels prove all four nodes."""
    from skulk.provisioning.comfy_extension import ComfyExtensionManifest
    from skulk.provisioning.manifest import COMFY_PIN

    package = tmp_path / "skulk_comfy_ltx"
    package.mkdir()
    manifest = ComfyExtensionManifest(protocol=1, version="0.1.0", comfy_revision=COMFY_PIN, ltx_revision="a" * 40, features=("res2s", "retake", "extend", "dfr") if dfr else ("res2s", "retake", "extend") if temporal else ("res2s",), payload_sha256="b" * 64, members={"skulk_comfy_ltx/nodes.py": "c" * 64})
    (package / "manifest.json").write_text(manifest.model_dump_json())
    expected = {"SkulkLTXRes2sSampler": ["SAMPLER"], "SkulkLTXGuidanceRescale": ["MODEL"]}
    if temporal:
        expected.update(SkulkLTXEditVideoLatent=["LATENT"], SkulkLTXEditAudioLatent=["LATENT", "AUDIO"], SkulkLTXPreserveVideo=["IMAGE"], SkulkLTXPreserveAudio=["AUDIO"])
    if dfr:
        expected["SkulkLTXAddGeneratedKeyframes"] = ["CONDITIONING", "CONDITIONING", "LATENT"]
    visited: list[str] = []

    @contextmanager
    def response(method: str, url: str, **_options: object) -> Iterator[httpx.Response]:
        assert method == "GET"
        name = url.rsplit("/", 1)[1]
        visited.append(name)
        with (
            httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={name: {"output": expected[name]}}))) as client,
            client.stream(method, url) as reply,
        ):
            yield reply

    monkeypatch.setattr(httpx, "stream", response)
    server = ComfyServer(interpreter=tmp_path / "python", root=tmp_path / "comfy", extra_model_paths=tmp_path / "models.yaml", input_dir=tmp_path / "input", output_dir=tmp_path / "output", temp_dir=tmp_path / "temp", user_dir=tmp_path / "user", log_path=tmp_path / "log", trusted_extension=package, base_url="http://127.0.0.1:55000")
    server._await_extension_schema()
    assert visited == list(expected)
    if temporal:
        expected["SkulkLTXEditAudioLatent"] = ["LATENT"]
        with pytest.raises(RuntimeError, match="exact SkulkLTXEditAudioLatent"):
            server._await_extension_schema()

    if dfr:
        expected["SkulkLTXEditAudioLatent"] = ["LATENT", "AUDIO"]
        expected["SkulkLTXAddGeneratedKeyframes"] = ["LATENT"]
        with pytest.raises(RuntimeError, match="exact SkulkLTXAddGeneratedKeyframes"):
            server._await_extension_schema()
