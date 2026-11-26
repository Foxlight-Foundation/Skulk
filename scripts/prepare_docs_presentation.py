"""Apply shared metadata tooling while retaining each channel's product source."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = (
    "docs-canonical.cjs",
    "docs-canonical.d.cts",
    "src/theme/SiteMetadata/index.tsx",
)


def prepare(repository: Path) -> None:
    """Give older stable checkouts the same canonical policy as the workflow."""
    website = repository / "website"
    if not (website / "docusaurus.config.ts").is_file():
        raise ValueError("Expected a product documentation checkout")
    for name in FILES:
        source = ROOT / "website" / name
        destination = website / name
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    prepare(parser.parse_args().repository)
