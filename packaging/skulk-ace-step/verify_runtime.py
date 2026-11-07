"""Exercise installed sealed native libraries and wrapper checks, without model weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import unittest
from pathlib import Path
from typing import Literal

import skulk_ace_step


class Arguments(argparse.Namespace):
    """Expected installed package profile; never an accelerator qualification."""

    variant: Literal["cpu", "cuda", "rocm"]


def main() -> None:
    """Require private installed dependencies, then run the dedicated Python3.12 checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "cuda", "rocm"), default="cpu")
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    prefix = Path(sys.prefix).resolve()
    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            "native ACE verification requires its independent Python3.12 runtime"
        )
    for name in (
        "skulk_ace_step",
        "acestep",
        "torch",
        "torchaudio",
        "scipy",
        "soundfile",
    ):
        module = importlib.import_module(name)
        location = module.__file__
        if location is None or not Path(location).resolve().is_relative_to(prefix):
            raise ValueError(
                "native ACE verification imported outside its private installation"
            )
    if importlib.metadata.version(f"skulk-ace-step-{arguments.variant}") != "0.1.0":
        raise ValueError("native ACE verification loaded the wrong installed adapter")
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent / "runtime_checks"), pattern="check_*.py"
    )
    count = suite.countTestCases()
    if count < 101:
        raise ValueError(
            "native ACE verification collected an incomplete wrapper suite"
        )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": "installed native dependency/wrapper/process boundary only; no music model loading or generation",
                "checks": count,
                "variant": arguments.variant,
                "source_revision": skulk_ace_step.SOURCE_REVISION,
                "torch": importlib.metadata.version("torch"),
                "python": sys.version.split()[0],
            }
        )
    )


if __name__ == "__main__":
    main()
