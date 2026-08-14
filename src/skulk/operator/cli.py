"""Local operator commands for designating and pairing a Skulk gateway."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from skulk.operator.pairing import (
    OperatorPairingService,
    PairingInvitationPackage,
    PairingPackageTooLargeError,
)
from skulk.operator.relay import OperatorRelayProvisioning
from skulk.utils.pydantic_ext import FrozenModel

DEFAULT_OPERATOR_API_PORT = 52417


class _PairArguments(FrozenModel):
    """Strictly validated local pairing command arguments."""

    command: Literal["pair"]
    exchange_url: str | None
    cluster_name: str
    valid_for: timedelta | None
    max_pairings: int | None
    qr_output: Path | None


class _ConfigureRelayArguments(FrozenModel):
    """Strictly validated local relay-provisioning command arguments."""

    command: Literal["configure-relay"]
    provisioning_file: Path
    operator_api_port: int
    cluster_name: str


class _InvitationListArguments(FrozenModel):
    """Strictly validated invitation-list arguments."""

    command: Literal["invitations"]
    invitation_command: Literal["list"]


class _InvitationRevokeArguments(FrozenModel):
    """Strictly validated invitation-revocation arguments."""

    command: Literal["invitations"]
    invitation_command: Literal["revoke"]
    invitation_id: UUID


_DURATION_PATTERN = re.compile(r"^([1-9][0-9]*)([mhd])$")


def _parse_invitation_duration(value: str) -> timedelta:
    """Parse a bounded integer minute, hour, or day duration."""

    match = _DURATION_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("duration must be a positive integer plus m, h, or d")
    amount = int(match.group(1))
    unit = match.group(2)
    duration = {
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
    }[unit]
    if duration > timedelta(days=90):
        raise argparse.ArgumentTypeError("duration must not exceed 90 days")
    return duration


def _print_pairing_qr(payload: str) -> None:
    """Render a terminal QR code plus the exact fallback payload."""

    import qrcode
    from qrcode.constants import ERROR_CORRECT_L

    code = qrcode.QRCode(
        border=2,
        error_correction=ERROR_CORRECT_L,
    )
    code.add_data(payload)
    code.make(fit=True)
    code.print_ascii(invert=True)
    print(payload)


def _write_pairing_qr(payload: str, output_path: Path) -> None:
    """Write a permission-restricted QR PNG without replacing an existing file."""

    import qrcode
    from qrcode.constants import ERROR_CORRECT_L

    code = qrcode.QRCode(border=2, error_correction=ERROR_CORRECT_L)
    code.add_data(payload)
    code.make(fit=True)
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            code.make_image().save(output)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local `skulk operator` command group.

    Args:
        argv: Arguments after the `operator` token. Defaults to process args.

    Returns:
        Conventional process exit status.
    """

    parser = argparse.ArgumentParser(prog="skulk operator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair_parser = subparsers.add_parser(
        "pair",
        help="Create a host-authorized phone pairing QR code or invitation.",
    )
    pair_parser.add_argument(
        "--exchange-url",
        help=(
            "Optional direct HTTPS base URL; a configured relay is used by default. "
            "HTTP is accepted only on loopback."
        ),
    )
    pair_parser.add_argument(
        "--cluster-name",
        default="Cluster",
        help="Initial cluster name when designating a gateway for the first time.",
    )
    pair_parser.add_argument(
        "--valid-for",
        type=_parse_invitation_duration,
        help="Create a reusable v3 invitation lasting 1m through 90d.",
    )
    pair_parser.add_argument(
        "--max-pairings",
        type=int,
        choices=range(1, 21),
        metavar="1-20",
        help="Create a reusable v3 invitation with this successful-pairing limit.",
    )
    pair_parser.add_argument(
        "--qr-output",
        type=Path,
        help="Also write a permission-restricted QR PNG without overwriting a file.",
    )
    relay_parser = subparsers.add_parser(
        "configure-relay",
        help="Install generated relay material on the designated gateway.",
    )
    relay_parser.add_argument(
        "--provisioning-file",
        required=True,
        type=Path,
        help="Protected JSON provisioning file received from the relay service.",
    )
    relay_parser.add_argument(
        "--operator-api-port",
        default=DEFAULT_OPERATOR_API_PORT,
        type=int,
        help=(
            "Loopback-only authenticated TLS API port "
            f"(default: {DEFAULT_OPERATOR_API_PORT})."
        ),
    )
    relay_parser.add_argument(
        "--cluster-name",
        default="Cluster",
        help="Initial cluster name when designating a gateway for the first time.",
    )
    invitations_parser = subparsers.add_parser(
        "invitations",
        help="List or revoke reusable pairing invitations.",
    )
    invitation_subparsers = invitations_parser.add_subparsers(
        dest="invitation_command",
        required=True,
    )
    invitation_subparsers.add_parser(
        "list",
        help="List safe invitation status without bearer capabilities.",
    )
    revoke_parser = invitation_subparsers.add_parser(
        "revoke",
        help="Revoke an invitation without disconnecting paired devices.",
    )
    revoke_parser.add_argument("invitation_id", type=UUID)
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    parsed_values = cast(dict[str, object], vars(parsed))
    service = OperatorPairingService.from_default_paths()
    if parsed_values.get("command") == "invitations":
        if parsed_values.get("invitation_command") == "list":
            _InvitationListArguments.model_validate(parsed_values)
            invitations = service.invitations()
            if not invitations:
                print("No reusable pairing invitations.")
                return 0
            print(
                "INVITATION ID                         CREATED                   "
                "EXPIRES                   USE  ACTIVE  STATE"
            )
            for invitation in invitations:
                print(
                    f"{invitation.invitation_id}  "
                    f"{invitation.created_at.isoformat()}  "
                    f"{invitation.expires_at.isoformat()}  "
                    f"{invitation.successful_pairings}/{invitation.max_pairings}  "
                    f"{invitation.active_attempts}       {invitation.state}"
                )
            return 0
        invitation_arguments = _InvitationRevokeArguments.model_validate(parsed_values)
        service.revoke_invitation(invitation_arguments.invitation_id)
        print(f"Revoked pairing invitation {invitation_arguments.invitation_id}.")
        return 0
    if parsed_values.get("command") == "configure-relay":
        arguments = _ConfigureRelayArguments.model_validate(parsed_values)
        try:
            provisioning = OperatorRelayProvisioning.model_validate_json(
                arguments.provisioning_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            # Validation errors may echo the rejected credential-bearing input.
            # The command reports only the safe failure class to stderr.
            parser.error("relay provisioning file is unreadable or invalid")
        configuration = service.configure_relay(
            provisioning,
            operator_api_port=arguments.operator_api_port,
            cluster_name=arguments.cluster_name,
        )
        print(
            "Configured the designated gateway with "
            f"{configuration.lane_count} relay lanes. Restart Skulk to connect."
        )
        return 0

    arguments = _PairArguments.model_validate(parsed_values)
    try:
        invitation_mode = (
            arguments.valid_for is not None or arguments.max_pairings is not None
        )
        package = (
            service.create_invitation(
                lifetime=arguments.valid_for or timedelta(minutes=5),
                max_pairings=arguments.max_pairings or 10,
                exchange_url=arguments.exchange_url,
                cluster_name=arguments.cluster_name,
            )
            if invitation_mode
            else service.create_session(
                exchange_url=arguments.exchange_url,
                cluster_name=arguments.cluster_name,
            )
        )
    except PairingPackageTooLargeError:
        parser.error("relay pairing package is too large for a reliable QR code")
    except ValueError:
        parser.error(
            "pairing requires a configured relay or an explicit --exchange-url"
        )
    print(
        f"Pair with {package.cluster_name} before "
        f"{package.expires_at.isoformat()}."
    )
    print(f"Cluster fingerprint: {package.cluster_fingerprint}")
    payload = package.as_url()
    if isinstance(package, PairingInvitationPackage):
        print(
            "This reusable QR is a secret bearer invitation. "
            f"It permits up to {package.max_pairings} successful pairings."
        )
    print(
        "Treat the terminal QR, fallback payload, and any saved image as secrets."
    )
    _print_pairing_qr(payload)
    if arguments.qr_output is not None:
        try:
            _write_pairing_qr(payload, arguments.qr_output)
        except FileExistsError:
            parser.error("--qr-output refuses to overwrite an existing file")
        except OSError:
            parser.error("--qr-output could not be written")
        print(f"Wrote protected QR image to {arguments.qr_output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
