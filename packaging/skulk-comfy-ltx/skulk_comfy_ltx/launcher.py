"""Start the pinned engine with exactly one verified custom-node search root."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Run Comfy with the worker-selected root and only this wheel's custom nodes.

    The bootstrap executes from an archive-verified installation. Source and
    arguments come from the runner, never a render request. Restricting the
    search function also prevents a same-named operator plugin from passing
    Comfy's folder-name whitelist in a different search root.
    """
    if len(sys.argv) < 2:
        raise ValueError("the trusted launcher requires its selected Comfy checkout")
    checkout = Path(sys.argv[1]).resolve(strict=True)
    library = Path(__file__).resolve(strict=True).parents[1]
    entrypoint = checkout / "main.py"
    sys.path[:0] = [str(library), str(checkout)]
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    # Comfy normally enables parsing before importing folder_paths. Preserve
    # that ordering so its global args contain the runner's GPU and I/O flags.
    import comfy.options

    comfy.options.enable_args_parsing()
    import folder_paths

    original = folder_paths.get_folder_paths

    def selected_paths(folder: str) -> list[str]:
        return [str(library)] if folder == "custom_nodes" else original(folder)

    folder_paths.get_folder_paths = selected_paths
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
