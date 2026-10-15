"""Assemble a pinned MLX video runtime and its complete offline dependency set."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

SOURCE_REVISION = "9e4a1a54194460547a103334b5e06382724fbba6"
VERSION = "0.2.1"
FEATURES = ("audio_conditioning", "retake", "extend", "prompt_relay")
PLATFORM_TAG = "macosx_15_0_arm64"
FILENAME = f"skulk_mlx_video-{VERSION}-cp313-cp313-{PLATFORM_TAG}.whl"


def _dependency_target_matches(filename: str) -> bool:
    """Check the wheel ABI/OS target rather than trusting an all-platform lock hash."""
    interpreter, abi, platform_tag = filename.removesuffix(".whl").split("-")[-3:]
    if platform_tag == "any":
        return abi == "none" and "py3" in interpreter.split(".")
    match = re.fullmatch(r"macosx_(\d+)_(\d+)_(arm64|universal2)", platform_tag)
    if match is None or (int(match[1]), int(match[2])) > (15, 0):
        return False
    if interpreter == "cp313" and abi == "cp313":
        return True
    if abi == "abi3" and re.fullmatch(r"cp3\d+", interpreter):
        return int(interpreter[3:]) <= 13
    return interpreter == "py3" and abi == "none"


def assemble_wheel(source: Path, wrapper: Path, wheelhouse: Path, lock: Path, output: Path) -> Path:
    """Validate source and locked wheels, then emit deterministic bytes and a descriptor.

    All third-party runtime dependencies travel inside this Foxlight wheel.
    Installation never resolves from an index or builds from source.
    """
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("MLX video source must be clean and match the immutable pin")
    lock_bytes = lock.read_bytes()
    requirements = {}
    for entry in re.split(r"\n(?=[a-zA-Z0-9])", lock_bytes.decode()):
        match = re.match(r"([\w.-]+)==([^\s\\]+)", entry)
        if match:
            requirements[match[1].replace("-", "_").lower()] = (match[2], set(re.findall(r"--hash=sha256:([a-f0-9]{64})", entry)))
    payload: dict[str, bytes] = {}
    for package in ("ltx_core_mlx", "ltx_pipelines_mlx"):
        root = source / "packages" / package.replace("_", "-") / "src" / package
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                payload[f"{package}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    for path in sorted(wrapper.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            payload[f"skulk_mlx_video/{path.relative_to(wrapper).as_posix()}"] = path.read_bytes()
    payload["skulk_mlx_video/media.py"] = (Path(__file__).resolve().parents[1] / "src/skulk/worker/runner/video_input.py").read_bytes()
    payload["skulk_mlx_video/telemetry.py"] = (Path(__file__).resolve().parents[1] / "src/skulk/worker/runner/video_observability.py").read_bytes()
    seen: set[str] = set()
    for path in sorted(wheelhouse.iterdir()):
        if path.suffix != ".whl":
            raise ValueError("dependency directory contains a non-wheel")
        if not _dependency_target_matches(path.name):
            raise ValueError(f"dependency ABI differs from the published target: {path.name}")
        name, version, *_ = path.name.split("-")
        name = name.lower()
        data = path.read_bytes()
        if name in seen or name not in requirements or requirements[name][0] != version or hashlib.sha256(data).hexdigest() not in requirements[name][1]:
            raise ValueError(f"unlocked or duplicate dependency wheel: {path.name}")
        seen.add(name)
        payload[f"skulk_mlx_video/wheels/{path.name}"] = data
    if seen != requirements.keys():
        raise ValueError("offline wheelhouse does not contain every locked dependency")
    payload["skulk_mlx_video/requirements.lock"] = lock_bytes
    payload["skulk_mlx_video/licenses/LTX_MLX_LICENSE.txt"] = (source / "LICENSE").read_bytes()
    payload["skulk_mlx_video/licenses/FOXLight_APACHE_LICENSE.txt"] = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    members = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())}
    identity = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["skulk_mlx_video/build.json"] = json.dumps({"source_revision": revision, "payload_sha256": identity, "members": members, "features": FEATURES}, sort_keys=True, indent=2).encode()
    distribution = f"skulk_mlx_video-{VERSION}.dist-info"
    payload[f"{distribution}/METADATA"] = (f"Metadata-Version: 2.4\nName: skulk-mlx-video\nVersion: {VERSION}\nRequires-Python: >=3.13,<3.14\nSummary: Offline GPU video runtime for Skulk\nLicense: See bundled component license notices\n\n").encode()
    payload[f"{distribution}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: foxlight-mlx-video\nRoot-Is-Purelib: false\nTag: cp313-cp313-{PLATFORM_TAG}\n".encode()
    records = []
    for name, data in sorted(payload.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append(f"{name},sha256={digest},{len(data)}")
    records.append(f"{distribution}/RECORD,,")
    payload[f"{distribution}/RECORD"] = ("\n".join(records) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / FILENAME
    if destination.exists():
        raise FileExistsError("refusing to overwrite immutable wheel bytes")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data)
    destination.with_suffix(".json").write_text(json.dumps({"filename": FILENAME, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "payload_sha256": identity, "size_bytes": destination.stat().st_size, "source_revision": revision, "features": FEATURES}, sort_keys=True, indent=2) + "\n")
    return destination


def main() -> None:
    """Build a candidate from explicit immutable inputs without publishing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "wrapper", "wheelhouse", "lock", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    arguments = parser.parse_args()
    print(assemble_wheel(arguments.source, arguments.wrapper, arguments.wheelhouse, arguments.lock, arguments.output))


if __name__ == "__main__":
    main()
