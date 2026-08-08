"""Local operator commands for designating and pairing a Skulk gateway."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from skulk.operator.pairing import (
    OperatorPairingService,
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


class _ConfigureRelayArguments(FrozenModel):
    """Strictly validated local relay-provisioning command arguments."""

    command: Literal["configure-relay"]
    provisioning_file: Path
    operator_api_port: int
    cluster_name: str


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
        help="Create a five-minute host-authorized phone pairing QR code.",
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
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    parsed_values = cast(dict[str, object], vars(parsed))
    service = OperatorPairingService.from_default_paths()
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
        package = service.create_session(
            exchange_url=arguments.exchange_url,
            cluster_name=arguments.cluster_name,
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
    _print_pairing_qr(package.as_url())
    return 0


if __name__ == "__main__":
    sys.exit(main())
