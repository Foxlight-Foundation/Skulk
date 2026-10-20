"""Local, network-free inspection and recovery of public preservation bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from foxlight_model_preservation.assurance import load_retained_card
from foxlight_model_preservation.signing import load_key

from skulk.store.preservation import (
    inspect_recovery,
    load_owner_trust,
    read_recovery_metadata,
    recover_home,
)
from skulk.store.preservation_export import plan_preservation, preserve_installed


class _Arguments(argparse.Namespace):
    command: Literal["inspect", "recover", "plan", "preserve"]
    installed: Path
    artifact_root: list[Path]
    retained_card: Path | None
    key: Path
    passphrase_file: Path | None
    store: Path
    manifest: str
    trust: Path
    historical: bool
    destination: Path
    accept_owner_capture: bool
    allow_repository_code: bool


def main(argv: list[str] | None = None) -> int:
    """Inspect or recover a pinned local bundle without launching a Skulk node.

    The receiver chooses the owner trust file. Registry proof uses only the
    packaged Foxlight root; neither a mirror nor the bundle may replace it.
    Results are JSON, errors go to stderr, and no destination is overwritten.
    """
    parser = argparse.ArgumentParser(prog="skulk-preservation", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "recover"):
        operation = commands.add_parser(command)
        operation.add_argument("manifest", help="exact SHA-256 manifest ID")
        operation.add_argument(
            "--store",
            type=Path,
            required=True,
            help="local preservation store or mounted NAS",
        )
        operation.add_argument(
            "--trust",
            type=Path,
            required=True,
            help="independently authenticated owner trust file",
        )
        operation.add_argument(
            "--historical",
            action="store_true",
            help="explicitly check historical registry publication, allowing expired metadata",
        )
        if command == "recover":
            operation.add_argument(
                "--destination",
                type=Path,
                required=True,
                help="new Skulk home; parent must already exist",
            )
            operation.add_argument(
                "--accept-owner-capture",
                action="store_true",
                help="accept owner-only, incomplete original file evidence, or historical publication as a custom model",
            )
            operation.add_argument(
                "--allow-repository-code",
                action="store_true",
                help="separately authorize executable model repository content for this exact recovery",
            )
    for command in ("plan", "preserve"):
        operation = commands.add_parser(command)
        operation.add_argument(
            "--installed",
            type=Path,
            required=True,
            help="installed base-model directory with a Skulk installed-card record",
        )
        operation.add_argument(
            "--artifact-root",
            type=Path,
            action="append",
            default=[],
            help="additional local root holding exact installed companions",
        )
        operation.add_argument(
            "--retained-card",
            type=Path,
            help="optional original public retained-card profile with offline registry proof",
        )
        if command == "preserve":
            operation.add_argument(
                "--store",
                type=Path,
                required=True,
                help="existing private archive directory; verify its mount before use",
            )
            operation.add_argument(
                "--key",
                type=Path,
                required=True,
                help="protected owner signing key; never included in the archive",
            )
            operation.add_argument(
                "--passphrase-file", type=Path, help="local key passphrase file"
            )
    arguments = parser.parse_args(argv, namespace=_Arguments())
    try:
        if arguments.command in {"plan", "preserve"}:
            retained = (
                load_retained_card(read_recovery_metadata(arguments.retained_card))
                if arguments.retained_card is not None
                else None
            )
            roots = tuple(arguments.artifact_root)
            if arguments.command == "plan":
                print(
                    plan_preservation(
                        arguments.installed,
                        artifact_roots=roots,
                        retained_card=retained,
                    ).model_dump_json(indent=2)
                )
            else:
                passphrase = (
                    read_recovery_metadata(arguments.passphrase_file).rstrip(b"\n")
                    if arguments.passphrase_file is not None
                    else None
                )
                manifest = preserve_installed(
                    arguments.installed,
                    arguments.store,
                    (load_key(arguments.key, passphrase),),
                    artifact_roots=roots,
                    retained_card=retained,
                )
                print(
                    json.dumps(
                        {"manifest_id": manifest, "audience": "private"}, indent=2
                    )
                )
            return 0
        trust = load_owner_trust(arguments.trust)
        mode = "historical" if arguments.historical else "fresh"
        if arguments.command == "inspect":
            result = inspect_recovery(
                arguments.store, arguments.manifest, trust, mode=mode
            )
        else:
            result = recover_home(
                arguments.store,
                arguments.manifest,
                trust,
                arguments.destination,
                mode=mode,
                accept_owner_capture=arguments.accept_owner_capture,
                allow_repository_code=arguments.allow_repository_code,
            )
        print(result.model_dump_json(indent=2))
    except (ValueError, OSError) as error:
        print(f"Preservation stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Preservation interrupted. Check the selected archive or recovery destination before retrying; an already published home is never overwritten.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
