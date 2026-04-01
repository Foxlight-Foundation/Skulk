from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_GENERATED = REPO_ROOT / "docs" / "generated"
SITE_DIR = REPO_ROOT / "site"
SITE_GENERATED = SITE_DIR / "generated"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def main() -> None:
    # Generate OpenAPI schema + ReDoc HTML
    run(["uv", "run", "python", "scripts/export_openapi.py"])

    # Generate TypeDoc markdown (output goes directly into docs/ tree)
    run(["npm", "run", "docs:typedoc"], cwd=REPO_ROOT / "dashboard-react")

    # Build MkDocs (TypeDoc markdown is processed natively)
    run(["uv", "run", "mkdocs", "build"])

    # Copy OpenAPI generated assets into site output
    if SITE_GENERATED.exists():
        shutil.rmtree(SITE_GENERATED)
    if DOCS_GENERATED.exists():
        shutil.copytree(DOCS_GENERATED, SITE_GENERATED)


if __name__ == "__main__":
    main()
