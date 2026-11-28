"""Steps around signing the Windows bindings wheel.

``.github/workflows/bindings-wheel.yml`` builds and tests an unsigned wheel of
``skulk_pyo3_bindings``. Windows can refuse to load an unsigned library (Smart
App Control does), so before publication the wheel's native library is signed
with Azure Artifact Signing. The signing itself is done by the workflow; this
script holds the checks around it, so that the published wheel is the tested
wheel plus a signature and nothing else:

  config   refuse to start without the five Artifact Signing settings
  unpack   check the tested wheel's digest and contents, extract it, and list
           the files to sign
  verify   require every listed file to carry a valid, timestamped signature
           of the publisher (Windows only: it asks Windows itself)
  pack     require each signed file to be the tested file plus a signature,
           every other file to be unchanged, and rebuild the wheel
  check    say whether a wheel is internally consistent and signed (any
           platform: it reads the files and does not judge the signature)

A wheel is a zip archive whose ``RECORD`` lists the SHA-256 and size of every
other file, so signing a file inside it means rewriting ``RECORD`` as well.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PUBLISHER = "FoxlightAI Inc."
SETTINGS = {
    "WINDOWS_SIGNING_AZURE_TENANT_ID": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    "WINDOWS_SIGNING_AZURE_CLIENT_ID": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    "WINDOWS_SIGNING_ENDPOINT": r"https://[a-z0-9]+\.codesigning\.azure\.net/?",
    "WINDOWS_SIGNING_ACCOUNT": r"[A-Za-z0-9][A-Za-z0-9-]{1,62}[A-Za-z0-9]",
    "WINDOWS_SIGNING_CERTIFICATE_PROFILE": r"[A-Za-z0-9][A-Za-z0-9-]{3,98}[A-Za-z0-9]",
}
WHEEL_DIRECTORY = "wheel"
CATALOG = "signing-catalog.txt"
RECORD_SCHEMA = "skulk.bindings-wheel-signing.v1"
SIGNATURES_PS1 = r"""
$ErrorActionPreference = 'Stop'
# ReadAllLines, not Get-Content: Get-Content decorates each line, and
# ConvertTo-Json would then write the path as an object, not as a string.
$result = foreach ($path in [System.IO.File]::ReadAllLines($args[0])) {
    $signature = Get-AuthenticodeSignature -LiteralPath $path
    [pscustomobject]@{
        path = $path
        status = [string]$signature.Status
        subject = if ($signature.SignerCertificate) { $signature.SignerCertificate.Subject } else { '' }
        timestamped = [bool]$signature.TimeStamperCertificate
    }
}
ConvertTo-Json -InputObject @($result) -Depth 3 -Compress
"""


class RefusedError(Exception):
    """The wheel or its surroundings are not what this step accepts."""


@dataclass(frozen=True)
class Entry:
    """One file of a wheel, with the archive attributes a rebuild keeps."""

    name: str
    data: bytes
    date_time: tuple[int, int, int, int, int, int]
    external_attr: int
    create_system: int
    compress_type: int


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 of ``data`` as lowercase hexadecimal."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the file at ``path`` as lowercase hexadecimal.

    The file is read in blocks, so a large wheel is never held in memory whole.
    """
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record_hash(data: bytes) -> str:
    """Return the digest of ``data`` as a wheel ``RECORD`` writes it.

    That is ``sha256=`` followed by the URL-safe base64 of the SHA-256 without
    padding.
    """
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


# --- PE files -------------------------------------------------------------


@dataclass(frozen=True)
class PeLayout:
    """Where a PE file keeps its checksum and its certificate table entry."""

    checksum_offset: int
    security_entry_offset: int
    certificate_offset: int
    certificate_size: int


def pe_layout(data: bytes) -> PeLayout | None:
    """The layout of a Windows PE file, or None when ``data`` is not one.

    A file that starts like a PE file and cannot be read as one is refused.
    """
    if len(data) < 64 or data[:2] != b"MZ":
        return None
    header = int.from_bytes(data[60:64], "little")
    if header < 64 or header + 24 > len(data) or data[header : header + 4] != b"PE\0\0":
        return None
    optional = header + 24
    optional_size = int.from_bytes(data[header + 20 : header + 22], "little")
    magic = int.from_bytes(data[optional : optional + 2], "little")
    if magic == 0x20B:
        directories = optional + 112
    elif magic == 0x10B:
        directories = optional + 96
    else:
        raise RefusedError(f"unknown PE optional header {magic:#x}")
    security_entry = directories + 4 * 8
    if (
        optional + optional_size > len(data)
        or security_entry + 8 > optional + optional_size
    ):
        raise RefusedError(
            "PE optional header is too short for a certificate table entry"
        )
    count = int.from_bytes(data[directories - 4 : directories], "little")
    if count < 5:
        raise RefusedError("PE file has no certificate table entry")
    return PeLayout(
        checksum_offset=optional + 64,
        security_entry_offset=security_entry,
        certificate_offset=int.from_bytes(
            data[security_entry : security_entry + 4], "little"
        ),
        certificate_size=int.from_bytes(
            data[security_entry + 4 : security_entry + 8], "little"
        ),
    )


def is_pe(data: bytes) -> bool:
    """Return whether ``data`` is a Windows PE file, judged by its headers.

    Raises ``RefusedError`` for a file that starts like one and cannot be read.
    """
    return pe_layout(data) is not None


def carries_signature(data: bytes) -> bool:
    """Whether a PE file has an attached certificate table.

    This says a signature is present. It does not say the signature is valid;
    only Windows judges that (see ``verify``).
    """
    layout = pe_layout(data)
    if layout is None:
        raise RefusedError("not a PE file")
    return layout.certificate_size > 0


def _without_signature_fields(data: bytes, layout: PeLayout, length: int) -> bytes:
    plain = bytearray(data[:length])
    plain[layout.checksum_offset : layout.checksum_offset + 4] = bytes(4)
    plain[layout.security_entry_offset : layout.security_entry_offset + 8] = bytes(8)
    return bytes(plain)


def signature_only_difference(original: bytes, signed: bytes) -> str | None:
    """Why ``signed`` is not ``original`` plus a signature, or None when it is.

    Signing a PE file appends the certificate table (after padding the file to
    a multiple of eight bytes), records where that table is, and updates the
    checksum. Nothing else may differ.
    """
    before = pe_layout(original)
    after = pe_layout(signed)
    if before is None or after is None:
        return "not a PE file"
    if before.certificate_size:
        return "the tested file already carried a signature"
    if not after.certificate_size:
        return "no signature was added"
    start = after.certificate_offset
    end = start + after.certificate_size
    if start < len(original) or start - len(original) >= 8 or start % 8:
        return "the certificate table does not follow the tested file directly"
    if any(signed[len(original) : start]):
        return "the padding before the certificate table is not empty"
    if end > len(signed) or len(signed) - end >= 8 or any(signed[end:]):
        return "the certificate table does not end the file"
    if (
        before.checksum_offset != after.checksum_offset
        or before.security_entry_offset != after.security_entry_offset
    ):
        return "the PE headers moved"
    if _without_signature_fields(
        original, before, len(original)
    ) != _without_signature_fields(signed, after, len(original)):
        return "bytes outside the signature differ from the tested file"
    return None


# --- wheels ---------------------------------------------------------------


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not (
        name.endswith("/")
        or path.is_absolute()
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 for character in name)
        or any(part in ("", ".", "..") for part in path.parts)
        or str(path) != name
    )


def _record_rows(data: bytes) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(data.decode("utf-8"))) if row]


def read_wheel(path: Path) -> tuple[list[Entry], str]:
    """The files of a wheel in archive order and the name of its ``RECORD``.

    Refuses a wheel with unsafe or repeated names, or whose ``RECORD`` does not
    list exactly its files with their SHA-256 and size.
    """
    entries: list[Entry] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            # ``orig_filename`` is the name as the archive stores it. Python
            # turns a backslash into a slash on Windows, and cuts a name at a
            # NUL byte everywhere, before it fills in ``filename``.
            stored = info.orig_filename
            if not _safe_name(stored) or info.filename != stored:
                raise RefusedError(f"unsafe path in the wheel: {stored!r}")
            entries.append(
                Entry(
                    name=info.filename,
                    data=archive.read(info),
                    date_time=info.date_time,
                    external_attr=info.external_attr,
                    create_system=info.create_system,
                    compress_type=info.compress_type,
                )
            )
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names) or len({name.lower() for name in names}) != len(
        names
    ):
        raise RefusedError("the wheel repeats a file name")
    records = [name for name in names if re.fullmatch(r"[^/]+\.dist-info/RECORD", name)]
    if len(records) != 1:
        raise RefusedError("the wheel does not have exactly one RECORD")
    record_name = records[0]
    by_name = {entry.name: entry for entry in entries}
    listed: set[str] = set()
    for row in _record_rows(by_name[record_name].data):
        if len(row) != 3 or row[0] in listed or row[0] not in by_name:
            raise RefusedError(
                f"RECORD row does not name one file of the wheel: {row!r}"
            )
        listed.add(row[0])
        if row[0] == record_name:
            if row[1] or row[2]:
                raise RefusedError("RECORD lists a digest for itself")
            continue
        data = by_name[row[0]].data
        if row[1] != record_hash(data) or row[2] != str(len(data)):
            raise RefusedError(f"RECORD does not match {row[0]}")
    if listed != set(names):
        raise RefusedError("RECORD does not list every file of the wheel")
    return entries, record_name


def rewritten_record(original: bytes, changed: dict[str, bytes]) -> bytes:
    """``RECORD`` with the rows of the changed files brought up to date."""
    ending = "\r\n" if b"\r\n" in original else "\n"
    output = io.StringIO()
    writer = csv.writer(output, lineterminator=ending)
    for row in _record_rows(original):
        if row[0] in changed:
            data = changed[row[0]]
            row = [row[0], record_hash(data), str(len(data))]
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def write_wheel(path: Path, entries: list[Entry]) -> None:
    """Write ``entries`` to a new archive at ``path``, in the order given.

    Each member keeps the name, date, attributes and compression its entry
    records. Raises ``FileExistsError`` when ``path`` exists.
    """
    with zipfile.ZipFile(path, "x") as archive:
        for entry in entries:
            info = zipfile.ZipInfo(entry.name, date_time=entry.date_time)
            info.external_attr = entry.external_attr
            info.create_system = entry.create_system
            info.compress_type = entry.compress_type
            archive.writestr(info, entry.data)


def expected_wheel_name(version: str) -> str:
    """Return the only file name a Windows bindings wheel of ``version`` may have.

    Raises ``RefusedError`` when ``version`` is not a dotted number.
    """
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)*", version):
        raise RefusedError(f"not a bindings version: {version!r}")
    return f"skulk_pyo3_bindings-{version}-cp313-cp313-win_amd64.whl"


def _checked_digest(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RefusedError("the expected SHA-256 is malformed")
    return value


def _destination(root: Path, name: str) -> Path:
    return root.joinpath(*PurePosixPath(name).parts)


# --- commands -------------------------------------------------------------


def config() -> None:
    """Require the five Artifact Signing settings in the environment.

    Prints a confirmation when each is present and has the shape of its kind.
    Raises ``RefusedError`` naming what is missing or malformed. Reads the
    environment and nothing else.
    """
    missing = [name for name in SETTINGS if not os.environ.get(name, "").strip()]
    if missing:
        raise RefusedError(
            "Azure Artifact Signing is not set up for this environment yet; missing "
            + ", ".join(missing)
        )
    for name, pattern in SETTINGS.items():
        if not re.fullmatch(pattern, os.environ[name].strip()):
            raise RefusedError(f"{name} does not look like an Artifact Signing value")
    print("Artifact Signing settings are present.")


def unpack(wheel: Path, sha256: str, version: str, target: Path) -> list[str]:
    """Extract the tested wheel and write the catalog of files to sign.

    ``wheel`` must be named for ``version`` and have the digest ``sha256``.
    Creates ``target`` (which must not exist) with the wheel's files under
    ``wheel/`` and ``signing-catalog.txt`` beside them. Returns the archive
    names of the files to sign. Raises ``RefusedError`` for a wheel that is
    not the one named, is inconsistent, has no native library, or is already
    signed; nothing is created in that case.
    """
    if wheel.name != expected_wheel_name(version):
        raise RefusedError(f"expected {expected_wheel_name(version)}, got {wheel.name}")
    if (
        wheel.is_symlink()
        or not wheel.is_file()
        or file_sha256(wheel) != _checked_digest(sha256)
    ):
        raise RefusedError("the wheel does not have the given SHA-256")
    entries, _ = read_wheel(wheel)
    to_sign = [entry.name for entry in entries if is_pe(entry.data)]
    if not to_sign:
        raise RefusedError("the wheel has no native library to sign")
    for entry in entries:
        if entry.name in to_sign and carries_signature(entry.data):
            raise RefusedError(
                f"{entry.name} is already signed; only a tested unsigned wheel is signed"
            )
    target.mkdir(parents=True, exist_ok=False)
    root = target / WHEEL_DIRECTORY
    for entry in entries:
        destination = _destination(root, entry.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(entry.data)
    # The signing action reads these paths relative to the catalog's folder.
    catalog = "".join(
        str(_destination(Path(WHEEL_DIRECTORY), name)) + "\n" for name in to_sign
    )
    (target / CATALOG).write_text(catalog, encoding="utf-8")
    print(f"{len(entries)} files; to sign: {', '.join(to_sign)}")
    return to_sign


def _catalog_paths(unpacked: Path) -> list[Path]:
    lines = (unpacked / CATALOG).read_text(encoding="utf-8").splitlines()
    return [unpacked / line.strip() for line in lines if line.strip()]


def read_signatures(paths: list[Path]) -> dict[str, dict[str, object]]:
    """Return what Windows says about each file's signature, keyed by path.

    Each value holds ``status`` (Windows' own word, such as ``Valid`` or
    ``NotSigned``), ``subject`` (the signer's certificate subject, or empty)
    and ``timestamped``. Runs Windows PowerShell, so it works on Windows only.
    Raises ``RefusedError`` when the answer does not cover exactly ``paths``.
    """
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / "signatures.ps1"
        listing = Path(scratch) / "paths.txt"
        script.write_text(SIGNATURES_PS1, encoding="utf-8")
        listing.write_text(
            "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
        )
        output = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                str(listing),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    readings = {
        item["path"]: item
        for item in json.loads(output)
        if isinstance(item["path"], str)
    }
    if set(readings) != {str(path) for path in paths}:
        raise RefusedError(
            "Windows did not report on exactly the files it was asked about"
        )
    return readings


def common_name(subject: object) -> str:
    """Return the common name (``CN``) from a certificate subject line.

    ``subject`` is the text Windows gives; a quoted name is returned without
    its quotes. Returns an empty string when the subject has no common name.
    """
    match = re.search(r"(?:^|,\s*)CN=(\"[^\"]*\"|[^,]*)", str(subject))
    return match.group(1).strip().strip('"') if match else ""


def signature_problems(
    paths: list[Path], current: dict[str, dict[str, object]]
) -> list[str]:
    """Return why each file's signature is not acceptable, one line per file.

    ``paths`` are the files that must be signed and ``current`` is what
    ``read_signatures`` returned for them. A file passes only when Windows
    calls its signature valid, the signature carries a timestamp, and the
    signer's common name is the publisher's. An empty list means every file
    passed.
    """
    problems: list[str] = []
    for path in paths:
        item = current.get(str(path))
        if item is None:
            problems.append(f"{path.name}: Windows reported nothing")
        elif item.get("status") != "Valid":
            problems.append(f"{path.name}: signature status {item.get('status')}")
        elif not item.get("timestamped"):
            problems.append(f"{path.name}: the signature has no timestamp")
        elif common_name(item.get("subject")) != PUBLISHER:
            problems.append(
                f"{path.name}: signed by {common_name(item.get('subject'))!r}, not {PUBLISHER!r}"
            )
    return problems


def verify(unpacked: Path) -> None:
    """Require every file in the catalog of ``unpacked`` to be validly signed.

    ``unpacked`` is the folder ``unpack`` made. Asks Windows about each listed
    file and prints a confirmation. Raises ``RefusedError`` with every problem
    found, or when run anywhere but Windows. Changes nothing.
    """
    if sys.platform != "win32":
        raise RefusedError("only Windows can judge an Authenticode signature")
    paths = _catalog_paths(unpacked)
    if not paths:
        raise RefusedError("the catalog lists no file")
    problems = signature_problems(paths, read_signatures(paths))
    if problems:
        raise RefusedError("signature check failed:\n" + "\n".join(problems))
    print(f"{len(paths)} file(s) validly signed and timestamped by {PUBLISHER}")


def pack(original: Path, unpacked: Path, output: Path) -> str:
    """Rebuild the wheel from the signed files and write its signing record.

    ``original`` is the tested wheel and ``unpacked`` the folder ``unpack``
    made from it, with its native libraries signed in place. Creates
    ``output`` (which must not exist) holding the rebuilt wheel under the
    original's name and ``signing-record.json``. Prints and returns the
    rebuilt wheel's SHA-256. Raises ``RefusedError``, creating nothing, unless
    each native library is the tested one plus a signature and every other
    file is unchanged.
    """
    entries, record_name = read_wheel(original)
    root = unpacked / WHEEL_DIRECTORY
    on_disk = {
        str(PurePosixPath(*path.relative_to(root).parts))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if on_disk != {entry.name for entry in entries}:
        raise RefusedError(
            "the unpacked wheel no longer has exactly the tested wheel's files"
        )
    signed: dict[str, bytes] = {}
    for entry in entries:
        path = _destination(root, entry.name)
        if path.is_symlink():
            raise RefusedError(f"symbolic link in the unpacked wheel: {entry.name}")
        data = path.read_bytes()
        if is_pe(entry.data):
            problem = signature_only_difference(entry.data, data)
            if problem:
                raise RefusedError(f"{entry.name}: {problem}")
            signed[entry.name] = data
        elif data != entry.data:
            raise RefusedError(
                f"{entry.name} changed, and it is not a file that is signed"
            )
    if not signed:
        raise RefusedError("the wheel has no native library to sign")
    rebuilt = [
        Entry(
            name=entry.name,
            data=rewritten_record(entry.data, signed)
            if entry.name == record_name
            else signed.get(entry.name, entry.data),
            date_time=entry.date_time,
            external_attr=entry.external_attr,
            create_system=entry.create_system,
            compress_type=entry.compress_type,
        )
        for entry in entries
    ]
    output.mkdir(parents=True, exist_ok=False)
    wheel = output / original.name
    write_wheel(wheel, rebuilt)
    read_wheel(wheel)
    digest = file_sha256(wheel)
    by_name = {entry.name: entry for entry in entries}
    record = {
        "schema_version": RECORD_SCHEMA,
        "wheel": wheel.name,
        "tested_sha256": file_sha256(original),
        "signed_sha256": digest,
        "signed_files": [
            {
                "path": name,
                "tested_sha256": sha256_hex(by_name[name].data),
                "signed_sha256": sha256_hex(data),
            }
            for name, data in sorted(signed.items())
        ],
    }
    (output / "signing-record.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(digest)
    return digest


def check(wheel: Path, unsigned: bool) -> None:
    """Require ``wheel`` to be consistent with its ``RECORD`` and signed as asked.

    With ``unsigned`` false, every native library must carry a signature; with
    ``unsigned`` true, none may. Prints what it found and raises
    ``RefusedError`` otherwise. This reads the files on any platform and does
    not judge whether a signature is valid; only ``verify`` does that.
    """
    entries, _ = read_wheel(wheel)
    native = [entry for entry in entries if is_pe(entry.data)]
    if not native:
        raise RefusedError("the wheel has no native library")
    with_signature = [entry.name for entry in native if carries_signature(entry.data)]
    if unsigned and with_signature:
        raise RefusedError(
            "asked for an unsigned wheel, and this one carries a signature: "
            + ", ".join(with_signature)
        )
    if not unsigned and len(with_signature) != len(native):
        missing = [entry.name for entry in native if entry.name not in with_signature]
        raise RefusedError("not signed: " + ", ".join(missing))
    state = (
        "carries no signature"
        if unsigned
        else "carries a signature on every native library"
    )
    print(f"{wheel.name}: consistent with its RECORD, {state}")


def main(arguments: list[str] | None = None) -> None:
    """Run the command named on the command line.

    ``arguments`` replaces ``sys.argv[1:]`` when given. A refusal is printed to
    standard error and ends the process with exit code 2.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config")
    command = commands.add_parser("unpack")
    command.add_argument("--wheel", type=Path, required=True)
    command.add_argument("--sha256", required=True)
    command.add_argument("--version", required=True)
    command.add_argument("--target", type=Path, required=True)
    command = commands.add_parser("verify")
    command.add_argument("--unpacked", type=Path, required=True)
    command = commands.add_parser("pack")
    command.add_argument("--original", type=Path, required=True)
    command.add_argument("--unpacked", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("check")
    command.add_argument("--wheel", type=Path, required=True)
    command.add_argument("--unsigned", action="store_true")
    args = parser.parse_args(arguments)
    try:
        if args.command == "config":
            config()
        elif args.command == "unpack":
            unpack(args.wheel, args.sha256, args.version, args.target)
        elif args.command == "verify":
            verify(args.unpacked)
        elif args.command == "pack":
            pack(args.original, args.unpacked, args.output)
        else:
            check(args.wheel, args.unsigned)
    except RefusedError as refusal:
        print(refusal, file=sys.stderr)
        raise SystemExit(2) from refusal


if __name__ == "__main__":
    main()
