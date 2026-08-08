"""Local operator commands for designating and pairing a Skulk gateway."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Literal

from skulk.operator.pairing import OperatorPairingService
from skulk.utils.pydantic_ext import FrozenModel


class _PairArguments(FrozenModel):
    """Strictly validated local pairing command arguments."""

    command: Literal["pair"]
    exchange_url: str
    cluster_name: str


def _print_pairing_qr(payload: str) -> None:
    """Render a terminal QR code plus the exact fallback payload."""

    import qrcode

    code = qrcode.QRCode(border=2)
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
        required=True,
        help="HTTPS base URL the phone can reach; HTTP is accepted only on loopback.",
    )
    pair_parser.add_argument(
        "--cluster-name",
        default="Cluster",
        help="Initial cluster name when designating a gateway for the first time.",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    arguments = _PairArguments.model_validate(vars(parsed))
    service = OperatorPairingService.from_default_paths()
    package = service.create_session(
        exchange_url=arguments.exchange_url,
        cluster_name=arguments.cluster_name,
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
