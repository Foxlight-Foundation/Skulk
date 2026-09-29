"""Promotion cannot replace a qualified CPU set with incomplete or changed bytes."""

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Literal, Protocol, cast

import pytest


class PromotionModule(Protocol):
    """Typed contract for the standalone artifact verification script."""

    verify_artifacts: Callable[
        [Path, Literal["cpu", "vulkan", "cuda"], str], tuple[Path, ...]
    ]


def _verifier() -> PromotionModule:
    script = Path(__file__).resolve().parents[4] / "scripts" / "verify_audio_cpp_promotion.py"
    spec = spec_from_file_location("audio_cpp_promotion_under_test", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(PromotionModule, cast(object, module))


def _cpu_set(directory: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for platform in ("macosx_15_0_arm64", "manylinux_2_35_x86_64", "manylinux_2_35_aarch64"):
        name = f"skulk_audio_cpp_cpu-0.8.2.post2-py3-none-{platform}.whl"
        data = platform.encode()
        (directory / name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    return expected


def test_promotes_complete_cpu_set_while_ignoring_other_variants(tmp_path: Path) -> None:
    """Downloading all workflow artifacts must not publish a GPU wheel as CPU."""
    expected = _cpu_set(tmp_path)
    (tmp_path / "skulk_audio_cpp_cuda-unrelated.whl").write_bytes(b"unrelated")

    wheels = _verifier().verify_artifacts(tmp_path, "cpu", json.dumps(expected))

    assert {wheel.name for wheel in wheels} == set(expected)


@pytest.mark.parametrize("change", ["missing", "extra", "mixed_version", "wrong_platform", "changed_bytes"])
def test_rejects_changed_or_incomplete_cpu_publication(tmp_path: Path, change: str) -> None:
    """Each promoted platform must be the exact version and bytes qualified."""
    expected = _cpu_set(tmp_path)
    name = next(iter(expected))
    path = tmp_path / name
    if change == "missing":
        path.unlink()
    elif change == "extra":
        extra = name.replace("post2", "post1")
        (tmp_path / extra).write_bytes(path.read_bytes())
        expected[extra] = expected[name]
    elif change in {"mixed_version", "wrong_platform"}:
        replacement = name.replace("post2", "post1") if change == "mixed_version" else name.replace("macosx_15_0_arm64", "win_amd64")
        path.rename(tmp_path / replacement)
        expected[replacement] = expected.pop(name)
    else:
        path.write_bytes(b"rebuilt bytes")

    with pytest.raises(ValueError):
        _verifier().verify_artifacts(tmp_path, "cpu", json.dumps(expected))


def test_rejects_duplicate_digest_map_keys(tmp_path: Path) -> None:
    """JSON's last-value-wins behavior must not conceal a duplicate wheel pin."""
    expected = _cpu_set(tmp_path)
    name, digest = next(iter(expected.items()))
    encoded = json.dumps(expected)[:-1] + f', "{name}": "{digest}"' + "}"

    with pytest.raises(ValueError, match="duplicate expected wheel"):
        _verifier().verify_artifacts(tmp_path, "cpu", encoded)


@pytest.mark.parametrize("encoded", ['[]', '{"file": 12}', '{"file": "not-a-digest"}'])
def test_rejects_invalid_cpu_digest_input(tmp_path: Path, encoded: str) -> None:
    """Malformed dispatch inputs cannot authorize publication."""
    _cpu_set(tmp_path)
    with pytest.raises(ValueError):
        _verifier().verify_artifacts(tmp_path, "cpu", encoded)


@pytest.mark.parametrize("variant", ["vulkan", "cuda"])
def test_preserves_single_gpu_wheel_contract(
    tmp_path: Path, variant: Literal["vulkan", "cuda"],
) -> None:
    """Existing single-wheel promotions still require exact bytes and one artifact."""
    wheel = tmp_path / f"skulk_audio_cpp_{variant}-qualified.whl"
    wheel.write_bytes(b"qualified")
    digest = hashlib.sha256(b"qualified").hexdigest()
    verifier = _verifier()
    assert verifier.verify_artifacts(tmp_path, variant, digest) == (wheel,)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verifier.verify_artifacts(tmp_path, variant, "0" * 64)
    (tmp_path / f"skulk_audio_cpp_{variant}-extra.whl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exactly one"):
        verifier.verify_artifacts(tmp_path, variant, digest)
