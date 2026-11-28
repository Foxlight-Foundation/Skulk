"""The signed bindings wheel is the tested wheel plus a signature, and nothing else."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "sign_bindings_wheel.py"
    spec = importlib.util.spec_from_file_location("sign_bindings_wheel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


signing = _load()

VERSION = "9.9.9"
WHEEL = f"skulk_pyo3_bindings-{VERSION}-cp313-cp313-win_amd64.whl"
LIBRARY = "skulk_pyo3_bindings/skulk_pyo3_bindings.cp313-win_amd64.pyd"
RECORD = f"skulk_pyo3_bindings-{VERSION}.dist-info/RECORD"


def pe_file(body: bytes = b"code" * 61, *, magic: int = 0x20B) -> bytes:
    """The smallest file the PE reader accepts: headers and a body."""
    header = 0x80
    optional_size = 240 if magic == 0x20B else 224
    data = bytearray(header + 24 + optional_size)
    data[:2] = b"MZ"
    data[60:64] = header.to_bytes(4, "little")
    data[header : header + 4] = b"PE\0\0"
    data[header + 20 : header + 22] = optional_size.to_bytes(2, "little")
    optional = header + 24
    data[optional : optional + 2] = magic.to_bytes(2, "little")
    directories = optional + (112 if magic == 0x20B else 96)
    data[directories - 4 : directories] = (16).to_bytes(4, "little")
    return bytes(data) + body


def signed(data: bytes, certificate: bytes = b"\x07" * 100) -> bytes:
    """What signing does to a PE file: pad, append the table, record it."""
    layout = signing.pe_layout(data)
    assert layout is not None
    padded = data + bytes(-len(data) % 8)
    table = certificate + bytes(-len(certificate) % 8)
    result = bytearray(padded + table)
    entry = layout.security_entry_offset
    result[entry : entry + 4] = len(padded).to_bytes(4, "little")
    result[entry + 4 : entry + 8] = len(table).to_bytes(4, "little")
    result[layout.checksum_offset : layout.checksum_offset + 4] = (0xBEEF).to_bytes(
        4, "little"
    )
    return bytes(result)


def _record_hash(data: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def build_wheel(
    directory: Path, files: dict[str, bytes] | None = None, *, record: str | None = None
) -> Path:
    contents = files or {
        "skulk_pyo3_bindings/__init__.py": b"from .skulk_pyo3_bindings import *\n",
        LIBRARY: pe_file(),
        f"skulk_pyo3_bindings-{VERSION}.dist-info/METADATA": b"Name: skulk_pyo3_bindings\n",
    }
    if record is None:
        record = "".join(
            f"{name},{_record_hash(data)},{len(data)}\n"
            for name, data in contents.items()
        )
        record += f"{RECORD},,\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WHEEL
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in [*contents.items(), (RECORD, record.encode())]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            # Store the name exactly as given: Python would otherwise rewrite
            # a backslash on Windows and cut the name at a NUL byte.
            info.filename = name
            info.external_attr = (
                0o100755 << 16 if name.endswith(".pyd") else 0o100644 << 16
            )
            info.create_system = 0
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return path


def unpacked(tmp_path: Path) -> tuple[Path, Path]:
    wheel = build_wheel(tmp_path / "tested")
    target = tmp_path / "work"
    signing.unpack(wheel, signing.file_sha256(wheel), VERSION, target)
    return wheel, target


def sign_in_place(target: Path) -> None:
    for path in signing._catalog_paths(target):
        path.write_bytes(signed(path.read_bytes()))


# --- PE files -------------------------------------------------------------


def test_a_file_that_is_not_pe_has_no_layout() -> None:
    assert signing.pe_layout(b"") is None
    assert signing.pe_layout(b"PK\x03\x04" + bytes(200)) is None
    assert signing.pe_layout(b"MZ" + bytes(200)) is None


@pytest.mark.parametrize("magic", [0x20B, 0x10B])
def test_a_signature_is_seen_in_both_pe_formats(magic: int) -> None:
    plain = pe_file(magic=magic)

    assert signing.carries_signature(plain) is False
    assert signing.carries_signature(signed(plain)) is True


def test_a_pe_file_cut_short_is_refused() -> None:
    with pytest.raises(signing.RefusedError):
        signing.pe_layout(pe_file()[:200])


def test_signing_alone_is_accepted() -> None:
    for body in (b"code" * 61, b"odd", b""):
        plain = pe_file(body)
        assert signing.signature_only_difference(plain, signed(plain)) is None


def test_a_changed_byte_outside_the_signature_is_named() -> None:
    plain = pe_file()
    changed = bytearray(signed(plain))
    changed[len(plain) - 1] ^= 1

    assert signing.signature_only_difference(plain, bytes(changed)) == (
        "bytes outside the signature differ from the tested file"
    )


def test_a_changed_header_byte_is_named() -> None:
    plain = pe_file()
    changed = bytearray(signed(plain))
    changed[0x80 + 6] ^= 1

    assert signing.signature_only_difference(plain, bytes(changed)) is not None


def test_signature_shapes_that_are_not_plain_signing_are_refused() -> None:
    plain = pe_file(b"odd")
    good = signed(plain)

    assert signing.signature_only_difference(plain, plain) == "no signature was added"
    assert (
        signing.signature_only_difference(good, signed(good))
        == "the tested file already carried a signature"
    )
    assert (
        signing.signature_only_difference(plain, good + b"\x01")
        == "the certificate table does not end the file"
    )
    dirty = bytearray(good)
    dirty[len(plain)] = 1
    assert signing.signature_only_difference(plain, bytes(dirty)) == (
        "the padding before the certificate table is not empty"
    )
    assert signing.signature_only_difference(plain, signed(plain + bytes(16))) == (
        "the certificate table does not follow the tested file directly"
    )
    assert signing.signature_only_difference(b"text", good) == "not a PE file"


# --- wheels ---------------------------------------------------------------


def test_a_wheel_is_read_with_its_record(tmp_path: Path) -> None:
    entries, record_name = signing.read_wheel(build_wheel(tmp_path))

    assert record_name == RECORD
    assert [entry.name for entry in entries][-1] == RECORD
    assert LIBRARY in [entry.name for entry in entries]


def test_a_record_that_does_not_match_is_refused(tmp_path: Path) -> None:
    library = pe_file()
    wrong = f"{LIBRARY},{_record_hash(b'other')},{len(library)}\n{RECORD},,\n"

    with pytest.raises(signing.RefusedError, match="RECORD does not match"):
        signing.read_wheel(build_wheel(tmp_path, {LIBRARY: library}, record=wrong))


def test_a_file_missing_from_the_record_is_refused(tmp_path: Path) -> None:
    library = pe_file()
    partial = f"{LIBRARY},{_record_hash(library)},{len(library)}\n{RECORD},,\n"

    with pytest.raises(signing.RefusedError, match="does not list every file"):
        signing.read_wheel(
            build_wheel(tmp_path, {LIBRARY: library, "extra.py": b""}, record=partial)
        )


def test_a_record_row_for_a_missing_file_is_refused(tmp_path: Path) -> None:
    library = pe_file()
    extra = f"{LIBRARY},{_record_hash(library)},{len(library)}\nghost.py,{_record_hash(b'')},0\n{RECORD},,\n"

    with pytest.raises(signing.RefusedError, match="does not name one file"):
        signing.read_wheel(build_wheel(tmp_path, {LIBRARY: library}, record=extra))


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "a\\b.py",
        "c:drive.py",
        "a//b.py",
        "a/./b.py",
        "a\x00b.py",
        "a\nb.py",
    ],
)
def test_an_unsafe_name_is_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(signing.RefusedError, match="unsafe path"):
        signing.read_wheel(build_wheel(tmp_path, {LIBRARY: pe_file(), name: b""}))


def test_names_that_differ_only_in_case_are_refused(tmp_path: Path) -> None:
    with pytest.raises(signing.RefusedError, match="repeats a file name"):
        signing.read_wheel(
            build_wheel(tmp_path, {LIBRARY: pe_file(), "a.py": b"", "A.py": b""})
        )


# --- unpack ---------------------------------------------------------------


def test_unpack_extracts_the_wheel_and_lists_the_library(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)

    listed = (target / "signing-catalog.txt").read_text(encoding="utf-8").splitlines()
    assert [Path(line) for line in listed] == [Path("wheel", *LIBRARY.split("/"))]
    entries, _ = signing.read_wheel(wheel)
    for entry in entries:
        assert (
            target.joinpath("wheel", *entry.name.split("/")).read_bytes() == entry.data
        )


def test_unpack_refuses_another_digest_name_or_place(tmp_path: Path) -> None:
    wheel = build_wheel(tmp_path / "tested")
    digest = signing.file_sha256(wheel)

    with pytest.raises(signing.RefusedError, match="given SHA-256"):
        signing.unpack(wheel, "0" * 64, VERSION, tmp_path / "a")
    with pytest.raises(signing.RefusedError, match="malformed"):
        signing.unpack(wheel, "not a digest", VERSION, tmp_path / "b")
    with pytest.raises(
        signing.RefusedError, match="expected skulk_pyo3_bindings-9.9.10-"
    ):
        signing.unpack(wheel, digest, "9.9.10", tmp_path / "c")
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        signing.unpack(wheel, digest, VERSION, tmp_path / "taken")
    assert not (tmp_path / "a").exists()


def test_unpack_refuses_a_wheel_that_is_already_signed(tmp_path: Path) -> None:
    wheel = build_wheel(tmp_path, {LIBRARY: signed(pe_file())})

    with pytest.raises(signing.RefusedError, match="already signed"):
        signing.unpack(wheel, signing.file_sha256(wheel), VERSION, tmp_path / "work")


def test_unpack_refuses_a_wheel_without_a_library(tmp_path: Path) -> None:
    wheel = build_wheel(tmp_path, {"skulk_pyo3_bindings/__init__.py": b""})

    with pytest.raises(signing.RefusedError, match="no native library"):
        signing.unpack(wheel, signing.file_sha256(wheel), VERSION, tmp_path / "work")


# --- pack -----------------------------------------------------------------


def test_pack_rebuilds_the_tested_wheel_around_the_signed_library(
    tmp_path: Path,
) -> None:
    wheel, target = unpacked(tmp_path)
    sign_in_place(target)

    digest = signing.pack(wheel, target, tmp_path / "signed")

    rebuilt = tmp_path / "signed" / WHEEL
    assert signing.file_sha256(rebuilt) == digest != signing.file_sha256(wheel)
    before, _ = signing.read_wheel(wheel)
    after, _ = signing.read_wheel(rebuilt)
    assert [entry.name for entry in after] == [entry.name for entry in before]
    for old, new in zip(before, after, strict=True):
        assert (
            new.date_time,
            new.external_attr,
            new.create_system,
            new.compress_type,
        ) == (
            old.date_time,
            old.external_attr,
            old.create_system,
            old.compress_type,
        )
        if old.name == LIBRARY:
            assert new.data == signed(old.data)
        elif old.name != RECORD:
            assert new.data == old.data
    signing.check(rebuilt, unsigned=False)


def test_pack_changes_only_the_library_row_of_the_record(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)
    sign_in_place(target)
    signing.pack(wheel, target, tmp_path / "signed")

    def rows(path: Path) -> dict[str, str]:
        entries, record_name = signing.read_wheel(path)
        text = next(
            entry.data for entry in entries if entry.name == record_name
        ).decode()
        return {line.split(",", 1)[0]: line for line in text.splitlines()}

    before = rows(wheel)
    after = rows(tmp_path / "signed" / WHEEL)
    assert list(after) == list(before)
    assert {name for name in before if before[name] != after[name]} == {LIBRARY}


def test_pack_writes_a_signing_record(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)
    sign_in_place(target)

    digest = signing.pack(wheel, target, tmp_path / "signed")

    record = json.loads(
        (tmp_path / "signed" / "signing-record.json").read_text(encoding="utf-8")
    )
    library = pe_file()
    assert record == {
        "schema_version": "skulk.bindings-wheel-signing.v1",
        "wheel": WHEEL,
        "tested_sha256": signing.file_sha256(wheel),
        "signed_sha256": digest,
        "signed_files": [
            {
                "path": LIBRARY,
                "tested_sha256": hashlib.sha256(library).hexdigest(),
                "signed_sha256": hashlib.sha256(signed(library)).hexdigest(),
            }
        ],
    }


def test_pack_refuses_an_unsigned_library(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)

    with pytest.raises(signing.RefusedError, match="no signature was added"):
        signing.pack(wheel, target, tmp_path / "signed")
    assert not (tmp_path / "signed").exists()


def test_pack_refuses_a_library_changed_beyond_its_signature(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)
    path = signing._catalog_paths(target)[0]
    changed = bytearray(signed(path.read_bytes()))
    changed[400] ^= 1
    path.write_bytes(bytes(changed))

    with pytest.raises(signing.RefusedError, match="differ from the tested file"):
        signing.pack(wheel, target, tmp_path / "signed")


def test_pack_refuses_any_other_changed_added_or_missing_file(tmp_path: Path) -> None:
    wheel, target = unpacked(tmp_path)
    sign_in_place(target)
    module = target / "wheel" / "skulk_pyo3_bindings" / "__init__.py"
    original = module.read_bytes()

    module.write_bytes(original + b"import os\n")
    with pytest.raises(signing.RefusedError, match="not a file that is signed"):
        signing.pack(wheel, target, tmp_path / "one")

    module.write_bytes(original)
    (target / "wheel" / "added.py").write_bytes(b"")
    with pytest.raises(signing.RefusedError, match="no longer has exactly"):
        signing.pack(wheel, target, tmp_path / "two")

    (target / "wheel" / "added.py").unlink()
    module.unlink()
    with pytest.raises(signing.RefusedError, match="no longer has exactly"):
        signing.pack(wheel, target, tmp_path / "three")


# --- check ----------------------------------------------------------------


def test_check_tells_a_signed_wheel_from_an_unsigned_one(tmp_path: Path) -> None:
    unsigned_wheel = build_wheel(tmp_path / "unsigned")
    signed_wheel = build_wheel(tmp_path / "signed", {LIBRARY: signed(pe_file())})

    signing.check(unsigned_wheel, unsigned=True)
    signing.check(signed_wheel, unsigned=False)
    with pytest.raises(signing.RefusedError, match="not signed"):
        signing.check(unsigned_wheel, unsigned=False)
    with pytest.raises(signing.RefusedError, match="carries a signature"):
        signing.check(signed_wheel, unsigned=True)


def test_the_command_line_exits_with_two_on_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = build_wheel(tmp_path)

    signing.main(["check", "--wheel", str(wheel), "--unsigned"])
    with pytest.raises(SystemExit) as stopped:
        signing.main(["check", "--wheel", str(wheel)])

    assert stopped.value.code == 2
    assert "not signed" in capsys.readouterr().err


# --- what Windows says ----------------------------------------------------


def _reading(
    status: str = "Valid",
    subject: str = "CN=FoxlightAI Inc., O=FoxlightAI Inc., C=US",
    stamped: bool = True,
):
    return {"status": status, "subject": subject, "timestamped": stamped}


def test_only_a_valid_timestamped_signature_of_the_publisher_passes() -> None:
    path = Path("library.pyd")

    def problems(reading: dict[str, object] | None) -> list[str]:
        return signing.signature_problems(
            [path], {} if reading is None else {str(path): reading}
        )

    assert problems(_reading()) == []
    assert (
        problems(_reading(subject='O=FoxlightAI Inc., CN="FoxlightAI Inc.", C=US'))
        == []
    )
    assert problems(None) == ["library.pyd: Windows reported nothing"]
    assert problems(_reading(status="NotSigned")) == [
        "library.pyd: signature status NotSigned"
    ]
    assert problems(_reading(status="UnknownError")) == [
        "library.pyd: signature status UnknownError"
    ]
    assert problems(_reading(stamped=False)) == [
        "library.pyd: the signature has no timestamp"
    ]
    assert problems(_reading(subject="CN=Someone Else, O=FoxlightAI Inc.")) == [
        "library.pyd: signed by 'Someone Else', not 'FoxlightAI Inc.'"
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows judges signatures itself")
def test_verify_is_refused_where_windows_cannot_be_asked(tmp_path: Path) -> None:
    _, target = unpacked(tmp_path)

    with pytest.raises(signing.RefusedError, match="only Windows"):
        signing.verify(target)


@pytest.mark.skipif(sys.platform != "win32", reason="asks Windows about a real file")
def test_windows_does_not_call_an_unsigned_file_valid(tmp_path: Path) -> None:
    _, target = unpacked(tmp_path)

    paths = signing._catalog_paths(target)

    readings = signing.read_signatures(paths)

    assert list(readings) == [str(path) for path in paths]
    assert readings[str(paths[0])]["status"] == "NotSigned"
    assert readings[str(paths[0])]["timestamped"] is False
    with pytest.raises(signing.RefusedError, match="signature check failed"):
        signing.verify(target)


# --- settings -------------------------------------------------------------

SETTINGS = {
    "WINDOWS_SIGNING_AZURE_TENANT_ID": "11111111-2222-3333-4444-555555555555",
    "WINDOWS_SIGNING_AZURE_CLIENT_ID": "66666666-7777-8888-9999-000000000000",
    "WINDOWS_SIGNING_ENDPOINT": "https://eus.codesigning.azure.net/",
    "WINDOWS_SIGNING_ACCOUNT": "example-account",
    "WINDOWS_SIGNING_CERTIFICATE_PROFILE": "example-profile",
}


def test_signing_does_not_start_without_every_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)
    signing.config()

    monkeypatch.setenv("WINDOWS_SIGNING_ENDPOINT", "https://example.com/")
    with pytest.raises(
        signing.RefusedError, match="WINDOWS_SIGNING_ENDPOINT does not look like"
    ):
        signing.config()

    monkeypatch.setenv("WINDOWS_SIGNING_ENDPOINT", SETTINGS["WINDOWS_SIGNING_ENDPOINT"])
    monkeypatch.delenv("WINDOWS_SIGNING_ACCOUNT")
    monkeypatch.setenv("WINDOWS_SIGNING_CERTIFICATE_PROFILE", " ")
    with pytest.raises(
        signing.RefusedError,
        match="missing WINDOWS_SIGNING_ACCOUNT, WINDOWS_SIGNING_CERTIFICATE_PROFILE",
    ):
        signing.config()
