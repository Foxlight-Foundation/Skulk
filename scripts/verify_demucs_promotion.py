"""Select all three fixed separator archives before append-only publication."""

import argparse
from pathlib import Path
from typing import final

from scripts.verify_ace_step_promotion import verify_artifacts
from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES


@final
class _Arguments(argparse.Namespace):
    directory: Path = Path()


def main() -> None:
    """Print the verified complete separator inventory with no pin overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    for wheel in verify_artifacts(arguments.directory, packages=DEMUCS_PACKAGES):
        print(wheel)


if __name__ == "__main__":
    main()
