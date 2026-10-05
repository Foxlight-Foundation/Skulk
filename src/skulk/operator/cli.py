"""Local operator commands for designating and pairing a Skulk gateway."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import yaml

from skulk.operator.pairing import (
    MAXIMUM_ACTIVE_PAIRED_DEVICES,
    OperatorDevice,
    OperatorDeviceNotFoundError,
    OperatorPairingService,
    PairingDeviceLimitError,
    PairingGatewayNotInitializedError,
    PairingInvitationPackage,
    PairingPackageTooLargeError,
    PairingSessionStateError,
)
from skulk.operator.relay import (
    DEFAULT_OPERATOR_API_PORT,
    OperatorRelayAlreadyConfiguredError,
    OperatorRelayConfiguration,
    OperatorRelayProvisioning,
)
from skulk.operator.relay_registration import (
    RelayRegistrationError,
    resolve_registration_origin,
)
from skulk.shared.constants import offline_mode
from skulk.store.config import load_skulk_config
from skulk.utils.pydantic_ext import FrozenModel


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


class _ForgetRelayArguments(FrozenModel):
    """Strictly validated relay-forget arguments."""

    command: Literal["forget-relay"]


class _InvitationListArguments(FrozenModel):
    """Strictly validated invitation-list arguments."""

    command: Literal["invitations"]
    invitation_command: Literal["list"]


class _InvitationRevokeArguments(FrozenModel):
    """Strictly validated invitation-revocation arguments."""

    command: Literal["invitations"]
    invitation_command: Literal["revoke"]
    invitation_id: UUID


class _DeviceListArguments(FrozenModel):
    """Strictly validated paired-device list arguments."""

    command: Literal["devices"]
    device_command: Literal["list"]


class _DeviceRevokeArguments(FrozenModel):
    """Strictly validated paired-device revocation arguments."""

    command: Literal["devices"]
    device_command: Literal["revoke"]
    device_id: UUID


_DEVICE_LIMIT_GUIDANCE = (
    f"this cluster already has {MAXIMUM_ACTIVE_PAIRED_DEVICES} paired devices, "
    "the most it allows; list them with `skulk operator devices list` and revoke "
    "one with `skulk operator devices revoke DEVICE_ID` before pairing another"
)


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


def _device_display_state(device: OperatorDevice, now: datetime) -> str:
    """Name a device's pairing state the way the device limit counts it.

    Args:
        device: Secret-free paired-device projection.
        now: Current UTC time.

    Returns:
        ``revoked``, ``expired`` (the refresh credential lapsed, so the device
        must pair again and no longer occupies a slot), or ``active``.
    """

    if device.state == "revoked":
        return "revoked"
    if device.refresh_expires_at is None or now >= device.refresh_expires_at:
        return "expired"
    return "active"


def _relay_configured_message(configuration: OperatorRelayConfiguration) -> str:
    """Describe a newly configured relay route in its own version's terms.

    Args:
        configuration: The relay configuration that was just persisted.

    Returns:
        The one-line confirmation printed by ``configure-relay``.
    """

    lane_count = configuration.lane_count
    if lane_count is None:
        # Version-two routes keep one signed control connection and open data
        # connections on demand, so there is no fixed lane pool to report.
        return (
            "Configured the designated gateway for on-demand relay "
            "connections. A running Skulk node connects within a few seconds."
        )
    lanes = "relay lane" if lane_count == 1 else "relay lanes"
    return (
        f"Configured the designated gateway with {lane_count} {lanes}. "
        "A running Skulk node connects within a few seconds."
    )


def _register_relay_or_exit(
    parser: argparse.ArgumentParser,
    service: OperatorPairingService,
    cluster_name: str,
) -> None:
    """Register this gateway with the configured relay, or exit with guidance.

    Args:
        parser: Command parser used to report a failure and exit.
        service: Local pairing service for this gateway.
        cluster_name: Name used only when initializing a new gateway.
    """

    try:
        config = load_skulk_config()
    except (OSError, ValueError, yaml.YAMLError):
        parser.error("skulk.yaml could not be read; fix it before pairing")
    connectivity = config.connectivity if config is not None else None
    relay = connectivity.relay if connectivity is not None else None
    try:
        origin = resolve_registration_origin(
            enabled=relay.enabled if relay is not None else True,
            configured_origin=relay.registration_url if relay is not None else None,
            offline=offline_mode(),
        )
        service.register_relay(
            registration_origin=origin,
            operator_api_port=DEFAULT_OPERATOR_API_PORT,
            cluster_name=cluster_name,
        )
    except RelayRegistrationError as exc:
        parser.error(
            f"could not register a relay route: {exc}. Pass --exchange-url for "
            "direct LAN or Tailscale pairing"
        )
    print(
        "Registered this gateway with the relay. A running Skulk node starts "
        "remote access within a few seconds."
    )


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
        help=(
            "Create a reusable v3 invitation with this successful-pairing limit, "
            "reduced to the cluster's free device slots "
            f"(at most {MAXIMUM_ACTIVE_PAIRED_DEVICES} paired devices)."
        ),
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
    subparsers.add_parser(
        "forget-relay",
        help=(
            "Forget this gateway's relay route; phones paired through it lose "
            "remote access."
        ),
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
    devices_parser = subparsers.add_parser(
        "devices",
        help="List or revoke paired operator devices.",
    )
    device_subparsers = devices_parser.add_subparsers(
        dest="device_command",
        required=True,
    )
    device_subparsers.add_parser(
        "list",
        help="List paired devices and the free device slots.",
    )
    device_revoke_parser = device_subparsers.add_parser(
        "revoke",
        help="Revoke one paired device immediately, freeing its slot.",
    )
    device_revoke_parser.add_argument("device_id", type=UUID)
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    parsed_values = cast(dict[str, object], vars(parsed))
    service = OperatorPairingService.from_default_paths()
    if parsed_values.get("command") == "devices":
        if parsed_values.get("device_command") == "list":
            _DeviceListArguments.model_validate(parsed_values)
            capacity = service.pairing_capacity()
            try:
                devices = service.owner_devices().devices
            except PairingGatewayNotInitializedError:
                devices = ()
            print(
                f"{capacity.active_devices} of {capacity.maximum_devices} device "
                f"slots in use; {capacity.available_slots} free."
            )
            if not devices:
                print("No paired devices.")
                return 0
            now = datetime.now(tz=timezone.utc)
            print(
                "DEVICE ID                             PAIRED                    "
                "STATE    NAME"
            )
            for device in devices:
                print(
                    f"{device.device_id}  {device.paired_at.isoformat()}  "
                    f"{_device_display_state(device, now):<7}  {device.name}"
                )
            return 0
        device_arguments = _DeviceRevokeArguments.model_validate(parsed_values)
        try:
            service.owner_revoke_device(device_arguments.device_id)
        except (OperatorDeviceNotFoundError, PairingGatewayNotInitializedError):
            parser.error("paired device was not found")
        except PairingSessionStateError:
            parser.error("pairing state changed concurrently; run the command again")
        print(f"Revoked paired device {device_arguments.device_id}.")
        return 0
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
        try:
            configuration = service.configure_relay(
                provisioning,
                operator_api_port=arguments.operator_api_port,
                cluster_name=arguments.cluster_name,
            )
        except OperatorRelayAlreadyConfiguredError:
            parser.error(
                "this gateway already has a relay route; run "
                "`skulk operator forget-relay` first to replace it"
            )
        print(_relay_configured_message(configuration))
        return 0
    if parsed_values.get("command") == "forget-relay":
        _ForgetRelayArguments.model_validate(parsed_values)
        if not service.forget_relay():
            print("This gateway has no relay route.")
            return 0
        print(
            "Forgot this gateway's relay route; a running Skulk node stops "
            "remote access within a few seconds. Phones paired through it keep "
            "their device slots until revoked (`skulk operator devices revoke`)."
        )
        return 0

    arguments = _PairArguments.model_validate(parsed_values)
    if arguments.exchange_url is None and service.relay_configuration() is None:
        # Check the device limit first so a full cluster does not register a
        # route it cannot use yet.
        if service.pairing_capacity().available_slots == 0:
            parser.error(_DEVICE_LIMIT_GUIDANCE)
        _register_relay_or_exit(parser, service, arguments.cluster_name)
    try:
        invitation_mode = (
            arguments.valid_for is not None or arguments.max_pairings is not None
        )
        package = (
            service.create_invitation(
                lifetime=arguments.valid_for or timedelta(minutes=5),
                max_pairings=arguments.max_pairings,
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
    except PairingDeviceLimitError:
        parser.error(_DEVICE_LIMIT_GUIDANCE)
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
        if (
            arguments.max_pairings is not None
            and package.max_pairings < arguments.max_pairings
        ):
            print(
                f"Reduced from {arguments.max_pairings} because the cluster has "
                f"{package.max_pairings} free device slots "
                f"(at most {MAXIMUM_ACTIVE_PAIRED_DEVICES} paired devices)."
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
